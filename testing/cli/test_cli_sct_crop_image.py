import pytest
import logging

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_crop_image

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
@pytest.mark.parametrize("path_in,path_out,remaining_args,expected_dim", [
    ('t2/t2.nii.gz', 't2_crop_xyz.nii', ['-xmin', '1', '-xmax', '-3', '-ymin', '2', '-ymax', '10'], (57, 9, 52)),
    ('t2/t2.nii.gz', 't2_crop_mask.nii', ['-m', 't2/t2_seg-manual.nii.gz'], (11, 55, 13)),
    ('t2/t2.nii.gz', 't2_crop_ref.nii', ['-ref', 'mt/mt0.nii.gz'], (37, 55, 34)),
])
def test_sct_crop_image_output_has_expected_dimensions(path_in, path_out, remaining_args, expected_dim):
    """Run the CLI script and verify cropped image has the expected dimensions."""
    sct_crop_image.main(argv=['-i', path_in, '-o', path_out] + remaining_args)
    # The last 5 dimension values [nt, px, py, pz, pt] should remain the same, so grab them from the input image
    expected_dim += Image(path_in).dim[3:8]
    assert Image(path_out).dim == expected_dim
