import pytest
import logging

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_apply_transfo

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
@pytest.mark.parametrize("path_in,path_dest,path_warp,path_out,remaining_args", [
    ('template/template/PAM50_small_t2.nii.gz', 't2/t2.nii.gz', 't2/warp_template2anat.nii.gz',
     'PAM50_small_t2_reg.nii', []),
    ('template/template/PAM50_small_t2.nii.gz', 't2/t2.nii.gz', 't2/warp_template2anat.nii.gz',
     'PAM50_small_t2_reg-crop1.nii', ['-crop', '1']),
    ('template/template/PAM50_small_t2.nii.gz', 't2/t2.nii.gz', 't2/warp_template2anat.nii.gz',
     'PAM50_small_t2_reg-crop2.nii', ['-crop', '2']),
    ('template/template/PAM50_small_t2.nii.gz', 't2/t2.nii.gz', 't2/warp_template2anat.nii.gz',
     'PAM50_small_t2_reg-concatWarp.nii', []),
    ('template/template/PAM50_small_t2.nii.gz', 'dmri/dmri.nii.gz', 't2/warp_template2anat.nii.gz',
     'PAM50_small_t2_reg-4Dref.nii', []),
    ('dmri/dmri.nii.gz', 't2/t2.nii.gz', 'mt/warp_t22mt1.nii.gz', 'PAM50_small_t2_reg-4Din.nii', [])
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
