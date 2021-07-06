import pytest
import logging

from spinalcordtoolbox.scripts import sct_compute_snr

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_snr_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_compute_snr.main(argv=['-i', 'dmri/dwi.nii.gz', '-m', 'dmri/dmri_T0001.nii.gz', '-method', 'diff',
                               '-vol', '0,5'])
