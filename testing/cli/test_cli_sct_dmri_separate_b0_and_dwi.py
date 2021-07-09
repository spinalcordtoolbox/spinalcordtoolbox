import pytest
import logging

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_dmri_separate_b0_and_dwi

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_dmri_separate_b0_and_dwi_image_data_within_threshold():
    """Run the CLI script and verify that the data of the output images is close to references (within threshold)."""
    sct_dmri_separate_b0_and_dwi.main(argv=['-i', 'dmri/dmri.nii.gz', '-bvec', 'dmri/bvecs.txt', '-a', '1', '-r', '0'])

    # check DWI
    ref_dwi = Image('dmri/dwi.nii.gz')
    new_dwi = Image('dmri_dwi.nii.gz')
    norm_img = np.linalg.norm(ref_dwi.data - new_dwi.data)
    assert norm_img < 0.001

    # check b=0
    ref_dwi = Image('dmri/dmri_T0000.nii.gz')
    new_dwi = Image('dmri_b0.nii.gz')
    norm_img = np.linalg.norm(ref_dwi.data - new_dwi.data)
    assert norm_img < 0.001
