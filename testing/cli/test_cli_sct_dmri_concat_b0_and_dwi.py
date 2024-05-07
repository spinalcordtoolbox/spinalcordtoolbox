# pytest unit tests for sct_dmri_concat_b0_and_dwi

import pytest
import logging

from spinalcordtoolbox.scripts import sct_dmri_concat_b0_and_dwi
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_dmri_concat_b0_and_dwi_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_dmri_concat_b0_and_dwi.main(argv=['-i',
                                          sct_test_path('dmri', 'dmri_T0000.nii.gz'),
                                          sct_test_path('dmri', 'dmri.nii.gz'),
                                          '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                                          '-bval', sct_test_path('dmri', 'bvals.txt'),
                                          '-order', 'b0', 'dwi', '-o', 'b0_dwi_concat.nii',
                                          '-obval', 'bvals_concat.txt', '-obvec', 'bvecs_concat.txt'])
