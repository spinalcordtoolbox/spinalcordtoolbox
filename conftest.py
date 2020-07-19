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

from spinalcordtoolbox.utils import sct_dir_local_path
sys.path.append(sct_dir_local_path('scripts'))

import pytest
import sct_download_data as downloader


@pytest.fixture(scope='session', autouse=True)
def download_data(request):
    # This is a hack because the capsys fixture can't be used with
    # session scope at the moment.
    # https://github.com/pytest-dev/pytest/issues/2704#issuecomment-603387680
    capmanager = request.config.pluginmanager.getplugin("capturemanager")
    with capmanager.global_and_fixture_disabled():
        print('\nDownloading sct testing data.')
        downloader.main(['-d', 'sct_testing_data'])
