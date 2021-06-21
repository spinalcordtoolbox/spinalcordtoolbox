import pytest
import logging

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_detect_pmj

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_detect_pmj_check_euclidean_distance_against_groundtruth():
    """Run the CLI script and verify that euclidean distances between predicted and ground truth coordinates
    are within a certain threshold."""
    fname_in = 'template/template/PAM50_small_t2.nii.gz'
    fname_out = 'PAM50_small_t2_pmj.nii.gz'
    fname_gt = 'template/template/PAM50_small_t2_pmj_manual.nii.gz'
    sct_detect_pmj.main(argv=['-i', fname_in, '-o', fname_out, '-c', 't2', '-qc', 'testing-qc'])

    im_pmj = Image(fname_out)
    im_pmj_manual = Image(fname_gt)

    # np.where outputs a tuple with an array for each axis: https://stackoverflow.com/q/50646102
    # so, convert ==> `tuple(x_array, y_array, z_array)` -> `list([x1, y1, z1], [x2, y2, z2], ...)`
    label_value = 50
    gt = [[x, y, z] for x, y, z in zip(*np.where(im_pmj_manual.data == label_value))]
    pred = [[x, y, z] for x, y, z in zip(*np.where(im_pmj.data == label_value))]

    # ensure that only one coordinate was predicted
    assert len(gt) == len(pred) == 1

    # transform pixel coordinates to physical coordinates (units in millimeters)
    gt_phys = im_pmj_manual.transfo_pix2phys(gt)
    pred_phys = im_pmj.transfo_pix2phys(pred)

    # ensure prediction is within 10mm of ground truth coordinates
    distances = np.linalg.norm(gt_phys - pred_phys, axis=1)
    assert np.all(distances < 10.0)
