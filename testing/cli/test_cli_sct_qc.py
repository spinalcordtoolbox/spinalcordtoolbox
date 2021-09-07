import pytest
import logging

from spinalcordtoolbox.scripts import sct_qc

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_qc_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_qc.main(argv=['-i', 't2/t2.nii.gz', '-s', 't2/t2_seg-manual.nii.gz', '-p', 'sct_deepseg_sc',
                      '-qc-dataset', 'sct_testing_data', '-qc-subject', 'dummy'])
