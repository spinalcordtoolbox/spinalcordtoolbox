import pytest
import logging

from spinalcordtoolbox.image import Image, compute_dice
from spinalcordtoolbox.scripts import sct_dice_coefficient

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_dice_coefficient_check_output_against_groundtruth():
    """Run the CLI script and verify its output matches a known ground truth value.
    FIXME: The CLI output gets ignored, and the function is instead tested by calling API function directly."""
    path_data = 't2/t2_seg-manual.nii.gz'
    sct_dice_coefficient.main(argv=['-i', path_data, '-d', path_data])

    im_seg_manual = Image(path_data)
    dice_segmentation = compute_dice(im_seg_manual, im_seg_manual, mode='3d', zboundaries=False)

    assert dice_segmentation == 1.0
