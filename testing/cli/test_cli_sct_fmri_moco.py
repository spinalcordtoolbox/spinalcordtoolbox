import pytest
import logging

from spinalcordtoolbox.scripts import sct_fmri_moco

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_fmri_moco_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_fmri_moco.main(argv=['-i', 'fmri/fmri_r.nii.gz', '-g', '4', '-x', 'nn', '-r', '0'])
