# pytest unit tests for sct_qc

import pytest
import logging

from spinalcordtoolbox.scripts import sct_qc
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_qc_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_qc.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'),
                      '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                      '-p', 'sct_deepseg_sc',
                      '-qc-dataset', 'sct_testing_data', '-qc-subject', 'dummy'])
