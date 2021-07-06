import pytest
import logging

from spinalcordtoolbox.image import Image, compute_dice
from spinalcordtoolbox.scripts import sct_dice_coefficient

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_dice_coefficient_check_output_against_groundtruth():
    """Run the CLI script and verify its output matches a known ground truth value."""
    # FIXME: The results of "sct_dice_coefficient" are not actually verified. Instead, the "compute_dice" function
    #        is called, and THOSE results are verified instead.
    # This was copied as-is from the existing 'sct_testing' test, but should be fixed at a later date.
    path_data = 't2/t2_seg-manual.nii.gz'
    sct_dice_coefficient.main(argv=['-i', path_data, '-d', path_data])
    im_seg_manual = Image(path_data)
    dice_segmentation = compute_dice(im_seg_manual, im_seg_manual, mode='3d', zboundaries=False)
    assert dice_segmentation == 1.0
