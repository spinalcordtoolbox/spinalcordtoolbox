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

from spinalcordtoolbox.utils import sct_dir_local_path, sct_test_path
sys.path.append(sct_dir_local_path('scripts'))

import pytest
import sct_download_data as downloader


def pytest_collectstart():
    # TODO [AJ] check integrity eventually
    if not os.path.exists(sct_test_path()):
        print('\nDownloading sct testing data.')
        downloader.main(['-d', 'sct_testing_data', '-o', sct_test_path()])
