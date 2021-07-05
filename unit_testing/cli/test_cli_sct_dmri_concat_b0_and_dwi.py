import pytest
import logging

from spinalcordtoolbox.scripts import sct_dmri_concat_b0_and_dwi

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_dmri_concat_b0_and_dwi_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_dmri_concat_b0_and_dwi.main(argv=['-i', 'dmri/dmri_T0000.nii.gz', 'dmri/dmri.nii.gz',
                                          '-bvec', 'dmri/bvecs.txt', '-bval', 'dmri/bvals.txt',
                                          '-order', 'b0', 'dwi', '-o', 'b0_dwi_concat.nii',
                                          '-obval', 'bvals_concat.txt', '-obvec', 'bvecs_concat.txt'])
