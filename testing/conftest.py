"""
Configure tests so that pytest downloads testing data every time pytest is run
from the sct directory

Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import os
import logging
from typing import Mapping
from hashlib import md5

import pytest

from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.download import install_named_dataset


logger = logging.getLogger(__name__)


def pytest_sessionstart():
    """Perform actions that must be done prior to test collection."""
    # Use a non-interactive backend so that no GUI plots will interrupt the test suite.
    # (NB: We do this here to ensure it is set before `matplotlib` is first imported.)
    if 'MPLBACKEND' not in os.environ:
        os.environ["MPLBACKEND"] = 'Agg'

    # Download sct_testing_data prior to test collection
    if not os.path.exists(sct_test_path()):
        logger.info("Downloading sct test data")
        install_named_dataset('sct_testing_data', dest_folder=sct_test_path())


@pytest.fixture(autouse=True)
def run_in_tmp_path(tmp_path):
    """
    Temporarily change the working directory to a pytest temporary dir. This is needed to prevent tests from
    cluttering the working directory *or* the sct_testing_data directory with output files.

    Note: Because this uses the `tmp_path` fixture, which generates one temporary directory per test, this allows
          tests to _also_ call `tmp_path` and access the same directory (e.g. for checking outputs).
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(cwd)


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
