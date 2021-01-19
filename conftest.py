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
import logging
from typing import Mapping
from hashlib import md5

import pytest
from pytest import fixture

from spinalcordtoolbox.utils.sys import sct_dir_local_path, sct_test_path
from spinalcordtoolbox.utils.fs import Lockf
from spinalcordtoolbox.scripts import sct_download_data as downloader


logger = logging.getLogger(__name__)


@fixture(scope="session")
def testrun_tmp_path(tmp_path_factory):
    """
    A session-scoped fixture giving the test run's temporary directory.
    Returns *the same* temporary directory -- even between pytest_xdist workers,
    if that is in use.

    Returns a pathlib.Path object.
    """

    if 'PYTEST_XDIST_WORKER' in os.environ:
        # we are a xdist worker process

        # Each worker has an isolated subdir but to coordinate locking for node-scoped fixtures,
        # we need them to share, so create a fixture that counteracts that.
        # This is based on:
        # - https://github.com/pytest-dev/pytest-xdist/blob/cf45eab9771ee271f8ec3eb4d33e23c914c70126/README.rst#making-session-scoped-fixtures-execute-only-once
        # - https://github.com/cloud-custodian/pytest-terraform/blob/17bb7e8f87540333d65ccdd2554464c798036101/pytest_terraform/xdist.py#L91-L95
        return tmp_path_factory.getbasetemp().parent
    else:
        # we're either not a worker process, *or* the user has used `pytest -n 0`
        # and we are the unique worker, which amounts to the same thing.
        return tmp_path_factory.getbasetemp()


@fixture(scope="session", autouse=True)
# TODO: make this *not* autouse; instead, replace all calls to sct_test_path() with this fixture, so that pytest can understand the dependencies.
def sct_testing_data(testrun_tmp_path):
    """ Download sct_testing_data prior to testing. """
    # pytest-xdist breaks session-scoped fixtures: it silently makes them [worker-scoped instead](https://github.com/pytest-dev/pytest-xdist/issues/271).
    #
    # This is a concurrency issue, and an interprocess one at that, so here we handle it with a lock file:
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

    _fixture_name = "sct_testing_data" # TODO: generalize to fixture.__name__
    lock = testrun_tmp_path / f"{_fixture_name}.lock"
    done = testrun_tmp_path / f"{_fixture_name}.done"
    # instead of testrun_tmp_path, per the suggestion on https://pypi.org/project/pytest-xdist/,
    # we could use testrun_uid to make our own distinct tmpdir, or perhaps a POSIX semaphore.
    # this is working for now.

    with open(lock, "w") as lock:
        with Lockf(lock):
            if not os.path.exists(done):
                if not os.path.exists(sct_test_path()):
                    downloader.main(['-d', 'sct_testing_data', '-o', sct_test_path()])
                with open(done, "w"): pass  # write the 'done' flag

    # return the test path so clients
    # TODO: merge test_data_integrity() in here

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
