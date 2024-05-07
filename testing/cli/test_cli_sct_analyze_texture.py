# pytest unit tests for sct_analyze_texture

import pytest
import logging

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_analyze_texture
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_analyze_texture_image_data_within_threshold():
    """Run the CLI script and verify that the output image data is close to a reference image (within threshold)."""
    sct_analyze_texture.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'),
                                   '-m', sct_test_path('t2', 't2_seg-manual.nii.gz'), '-feature', 'contrast',
                                   '-distance', '1', '-ofolder', '.'])

    im_texture = Image('t2_contrast_1_mean.nii.gz')
    im_texture_ref = Image(sct_test_path('t2', 't2_contrast_1_mean_ref.nii.gz'))

    assert np.linalg.norm(im_texture.data - im_texture_ref.data) <= 0.001
