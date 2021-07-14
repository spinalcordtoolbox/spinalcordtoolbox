import pytest
import logging

from spinalcordtoolbox.scripts import sct_compute_mtr

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_mtr_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_compute_mtr.main(argv=['-mt0', 'mt/mt0.nii.gz', '-mt1', 'mt/mt1.nii.gz'])
