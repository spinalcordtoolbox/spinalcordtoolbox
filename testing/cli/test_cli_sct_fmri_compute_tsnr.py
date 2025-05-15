# pytest unit tests for sct_fmri_compute_tsnr

import pytest
import logging

from spinalcordtoolbox.scripts import sct_fmri_compute_tsnr, sct_maths, sct_deepseg
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_sct_fmri_compute_tsnr_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_fmri_compute_tsnr.main(argv=['-i', sct_test_path('fmri', 'fmri.nii.gz'),
                                     '-o', 'out_fmri_tsnr.nii.gz'])


@pytest.fixture()
def img_mask(tmp_path):
    """Create a binary mask to highlight the spinal cord."""
    # create mean image of all temporal volumes
    fname_in = sct_test_path('fmri', 'fmri.nii.gz')
    fname_mean = str(tmp_path / "fmri_mean.nii.gz")
    sct_maths.main(argv=["-i", fname_in, "-mean", "t", "-o", fname_mean])
    # segment the spinal cord
    # nb: normally we would segment another contrast then bring it to the fmri space, but direct seg is "good enough"
    fname_out = str(tmp_path / "fmri_seg.nii.gz")
    sct_deepseg.main(argv=["spinalcord", "-i", fname_mean, "-o", fname_out])
    return fname_out


def test_sct_sct_fmri_compute_tsnr_with_mask(tmp_path, img_mask):
    """Run the CLI script using a mask image (to test the QC)."""
    sct_fmri_compute_tsnr.main(argv=['-i', sct_test_path('fmri', 'fmri.nii.gz'), '-m', img_mask,
                                     '-o', 'out_fmri_tsnr.nii.gz', '-qc', str(tmp_path)])
