# pytest unit tests for sct_fmri_compute_tsnr

import pytest
import logging

from spinalcordtoolbox.scripts import sct_fmri_compute_tsnr
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_sct_fmri_compute_tsnr_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_fmri_compute_tsnr.main(argv=['-i', sct_test_path('fmri', 'fmri.nii.gz'),
                                     '-o', 'out_fmri_tsnr.nii.gz'])
