import os

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
    nx, ny, nz, _, _, _, _, _ = Image(path_out).dim
    assert (nx, ny, nz) == expected_dim
