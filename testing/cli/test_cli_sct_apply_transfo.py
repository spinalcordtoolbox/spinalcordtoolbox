# pytest unit tests for sct_apply_transfo

import pytest
import logging

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_apply_transfo
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.parametrize("path_in,path_dest,path_warp,path_out,remaining_args", [
    (sct_test_path('template', 'template', 'PAM50_small_t2.nii.gz'),
     sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('t2', 'warp_template2anat.nii.gz'),
     'PAM50_small_t2_reg.nii', []),
    (sct_test_path('template', 'template', 'PAM50_small_t2.nii.gz'),
     sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('t2', 'warp_template2anat.nii.gz'),
     'PAM50_small_t2_reg-crop1.nii', ['-crop', '1']),
    (sct_test_path('template', 'template', 'PAM50_small_t2.nii.gz'),
     sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('t2', 'warp_template2anat.nii.gz'),
     'PAM50_small_t2_reg-crop2.nii', ['-crop', '2']),
    (sct_test_path('template', 'template', 'PAM50_small_t2.nii.gz'),
     sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('t2', 'warp_template2anat.nii.gz'),
     'PAM50_small_t2_reg-concatWarp.nii', []),
    (sct_test_path('template', 'template', 'PAM50_small_t2.nii.gz'),
     sct_test_path('dmri', 'dmri.nii.gz'),
     sct_test_path('t2', 'warp_template2anat.nii.gz'),
     'PAM50_small_t2_reg-4Dref.nii', []),
    (sct_test_path('dmri', 'dmri.nii.gz'),
     sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('mt', 'warp_t22mt1.nii.gz'),
     'PAM50_small_t2_reg-4Din.nii', []),
    (sct_test_path('t2', 'labels.nii.gz'),
     sct_test_path('mt', 'mt1.nii.gz'),
     sct_test_path('mt', 'warp_t22mt1.nii.gz'),
     'labels_mt1.nii', ['-x', 'nn'])
])
def test_sct_apply_transfo_output_image_attributes(path_in, path_dest, path_warp, path_out, remaining_args):
    """Run the CLI script and verify transformed images have expected attributes."""
    sct_apply_transfo.main(argv=['-i', path_in, '-d', path_dest, '-w', path_warp, '-o', path_out] + remaining_args)

    img_src = Image(path_in)
    img_ref = Image(path_dest)
    img_output = Image(path_out)

    assert img_output.orientation == img_ref.orientation
    assert (img_output.data != 0).any()
    # Only checking the first 3 dimensions because one test involves a 4D volume
    assert img_ref.dim[0:3] == img_output.dim[0:3]
    # Checking the 4th dim (which should be the same as the input image, not the reference image)
    assert img_src.dim[3] == img_output.dim[3]
    # Make sure that integer labels are preserved, but only for NearestNeighbours interp
    if issubclass(img_src.data.dtype.type, np.integer):
        if remaining_args == ['-x', 'nn']:
            assert img_output.data.dtype == img_src.data.dtype
        else:
            assert img_output.data.dtype != img_src.data.dtype
