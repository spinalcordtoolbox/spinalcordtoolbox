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

from spinalcordtoolbox.utils.sys import sct_dir_local_path, sct_test_path
from spinalcordtoolbox.scripts import sct_download_data as downloader


logger = logging.getLogger(__name__)


def pytest_sessionstart():
    """ Download sct_testing_data prior to test collection. """
    logger.info("Downloading sct test data")
    downloader.main(['-d', 'sct_testing_data', '-o', sct_test_path()])


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
