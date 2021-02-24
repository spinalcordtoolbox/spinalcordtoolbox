import math
import pytest
import logging

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_detect_pmj

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_detect_pmj_check_euclidean_distance_against_groundtruth():
    """Run the CLI script and verify that euclidean distance (computed against ground truth) is within a certain
    threshold."""
    fname_out = 'PAM50_small_t2_pmj.nii.gz'
    fname_gt = 'template/template/PAM50_small_t2_pmj_manual.nii.gz'
    sct_detect_pmj.main(argv=['-i', 'template/template/PAM50_small_t2.nii.gz', '-c', 't2', '-qc', 'testing-qc'])

    im_pmj = Image(fname_out)
    im_pmj_manual = Image(fname_gt)

    # compute Euclidean distance between predicted and GT PMJ label
    x_true, y_true, z_true = np.where(im_pmj_manual.data == 50)
    x_pred, y_pred, z_pred = np.where(im_pmj.data == 50)
    x_true, y_true, z_true = im_pmj_manual.transfo_pix2phys([[x_true[0], y_true[0], z_true[0]]])[0]
    x_pred, y_pred, z_pred = im_pmj.transfo_pix2phys([[x_pred[0], y_pred[0], z_pred[0]]])[0]

    distance_detection = math.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2 + (z_true - z_pred)**2)
    assert distance_detection < 10.0
