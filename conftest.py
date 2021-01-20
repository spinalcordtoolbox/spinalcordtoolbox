#############################################################################
#
# Configure tests so that pytest downloads testing data every time
# pytest is run from the sct directory
#
# ----------------------------------------------------------------------------
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad, Chris Hammill
#
# About the license: see the file LICENSE.TXT
###############################################################################

import sys
import os
import pathlib
import tempfile
import logging
from typing import Mapping
import functools
from contextlib import contextmanager
import inspect
from hashlib import md5
try:
    import cPickle as pickle # py 3.6 doesn't have cPickle?
except ImportError:
    import pickle

import pytest
from pytest import fixture

from spinalcordtoolbox.utils.sys import sct_dir_local_path, sct_test_path
from spinalcordtoolbox.utils.fs import Lockf
from spinalcordtoolbox.scripts import sct_download_data as downloader


logger = logging.getLogger(__name__)


def node_scoped(fixture):
    """
    Ensure a fixture only runs once per *node* per test run.

    When using pytest-xdist, there are multiple worker processes potentially spread out over on multiple worker machines ("nodes").
    It causes `@fixture(scope="session")` to essentially mean `@fixture(scope="worker")` and this has caused lots of confusion:
      https://github.com/pytest-dev/pytest-xdist/issues/271

    For fixtures that need to generate shared data in-RAM, that is fine. But for fixtures that generate shared data on-disk
    it causes redundant work and possible corruption.
    Since this doesn't seem fixable inside of pytest (https://github.com/pytest-dev/pytest-xdist/issues/271#issuecomment-762453836)
    this provides a workaround that creates fixtures that run once per run per node.

    Usage:

        @fixture(scope="session")
        @node_scoped
        def fixture():
            return make_some_data()
    """
    tmpdir = pathlib.Path(tempfile.gettempdir())
    if 'PYTEST_XDIST_TESTRUNUID' in os.environ:
        # running under xdist
        tmpdir = tmpdir / f"pytest-{os.environ['PYTEST_XDIST_TESTRUNUID']}"
    else:
        # not under xdist
        # this isn't necessarily safe! this *could* be reused. UUIDs aren't, but PIDs are, sometimes.
        # on the other hand, we shouldn't even need a lock in this case, but we'll make one anyway.
        tmpdir = tmpdir / f"pytest-{os.getpid()}"
    fixture_path = tmpdir / f"{fixture.__name__}"
    os.makedirs(fixture_path, exist_ok=True)
    lock = fixture_path / f"lock"
    done = fixture_path / f"done"

    @contextmanager
    def nodelock():
        #
        # xdist means we need to deal with interprocess concurrency, so here we handle it with a lock file:
        # whichever worker gets the lock first generates the fixture, then marks itself done using a second file.
        # Every other worker blocks until at least the first one is done, then they will see the '.done' file and skip on.
        # We don't try to handle cleaning up the lockfile when we're done (which can be very tricky to get right!);
        # because we're running under pytest we can just lean on the master process to clean it up for us.
        #
        # Here we are using [`fcntl.lockf`](https://apenwarr.ca/log/20101213):
        # > in C, use fcntl(), but avoid lockf(), which is not necessarily the same thing.
        # > in python, use fcntl.lockf(), which is the same thing as fcntl() in C.
        #
        # There are several other options:
        # - [`portalocker.Lock`](https://pypi.org/project/portalocker/)
        # - [`posix_ipc.Semaphore`](http://semanchuk.com/philip/posix_ipc/)
        # - ?
        #
        # This is compatible with pytest-xdist's multi-server mode, where it distributes tests out via ssh.
        # Each node has its own fcntl locks, so workers that end up on the same host will only do the work
        # once, while those on different hosts will have at least one of them pick it up.

        with open(lock, "w") as lock_fd:
            with Lockf(lock_fd):
                yield

    # in order to support generator-fixtures, we need to *give* a generator function
    # -- not just a generator: a generator function -- because pytest internally looks
    # at inspect.isgeneratorfunction() to detect it. Otherwise, the lock just locks
    # its *initialization*, i.e. no actual code of the fixture.
    #
    # The difference between a function and a generator function is the 'yield' keyword,
    # so that means we need to have two nearly identical def:s. I don't know how to reduce the duplication here.
    # Maybe, in the non-generator case, wrap it with something that makes it a generator and only do that?
    if inspect.isgeneratorfunction(fixture):

        def _fixture(*args, **kwargs):
            with nodelock():
                if not os.path.exists(done):
                    g = fixture(*args, **kwargs)
                    result = next(g)
                    with open(done, 'wb') as cache:
                        pickle.dump(result, cache)
                    yield g
                else:
                    with open(done, 'rb') as cache:
                        yield pickle.load(cache)
                    g = (e for e in []) # just make an empty generator for below
            # this sleeps here until pytest is done with all the tests
            with nodelock():
                # delete cache
                if os.path.exists(done):
                    os.unlink(done)
                # clean up fixture
                yield from g

    else:

        def _fixture(*args, **kwargs):
            with nodelock():
                if not os.path.exists(done):
                    result = fixture(*args, **kwargs)
                    with open(done, 'wb') as cache:
                        pickle.dump(result, cache)
                else:
                    with open(done, 'rb') as cache:
                        result = pickle.load(cache)
                return result
                # NB: cache isn't deleted here!

    _fixture = functools.wraps(fixture)(_fixture)

    return _fixture


@fixture(scope="session", autouse=True)
# TODO: make this *not* autouse; instead, replace all calls to sct_test_path() with this fixture, so that pytest can understand the dependencies.
@node_scoped
def sct_testing_data(worker_id):
    """ Download sct_testing_data prior to testing. """

    # TODO: merge test_data_integrity() in here

    if not os.path.exists(sct_test_path()):
        downloader.main(['-d', 'sct_testing_data', '-o', sct_test_path()])

    # return the test path so clients know they can use it
    yield sct_test_path()

    # TODO: delete the data ?


@pytest.fixture(scope="session", autouse=True)
def test_data_integrity(request):
    files_checksums = dict()
    for root, _, files in os.walk(sct_test_path()):
        for f in files:
            fname = os.path.join(root, f)
            chksum = checksum(fname)
            files_checksums[fname] = chksum

    request.addfinalizer(lambda: check_testing_data_integrity(files_checksums))


def checksum(fname: os.PathLike) -> str:
    with open(fname, 'rb') as f:
        data = f.read()
    return md5(data).hexdigest()


def check_testing_data_integrity(files_checksums: Mapping[os.PathLike, str]):
    changed = []
    new = []
    missing = []

    after = []

    for root, _, files in os.walk(sct_test_path()):
        for f in files:
            fname = os.path.join(root, f)
            chksum = checksum(fname)
            after.append(fname)

            if fname not in files_checksums:
                logger.warning(f"Discovered new file in sct_testing_data that didn't exist before: {(fname, chksum)}")
                new.append((fname, chksum))

            elif files_checksums[fname] != chksum:
                logger.error(f"Checksum mismatch for test data: {fname}. Got {chksum} instead of {files_checksums[fname]}")
                changed.append((fname, chksum))

    for fname, chksum in files_checksums.items():
        if fname not in after:
            logger.error(f"Test data missing after test:a: {fname}")
            missing.append((fname, chksum))

    assert not changed
    # assert not new
    assert not missing
