# pytest unit tests for sct_deepseg_gm

import pytest
import logging

from spinalcordtoolbox.image import Image, compute_dice
from spinalcordtoolbox.scripts import sct_deepseg_gm
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_deepseg_gm_check_dice_coefficient_against_groundtruth():
    """Run the CLI script and verify that dice (computed against ground truth) is within a certain threshold."""
    fname_out = 'output.nii.gz'
    fname_gt = sct_test_path('t2s', 't2s_uncropped_gmseg_manual.nii.gz')
    sct_deepseg_gm.main(argv=['-i', sct_test_path('t2s', 't2s_uncropped.nii.gz'),
                              '-o', fname_out, '-qc', 'testing-qc'])
    dice_segmentation = compute_dice(Image(fname_out), Image(fname_gt), mode='3d', zboundaries=False)
    assert dice_segmentation >= 0.85
