import os
import pytest
import logging

from spinalcordtoolbox.image import Image, compute_dice
from spinalcordtoolbox.utils import run_proc
from spinalcordtoolbox.scripts import sct_propseg

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_propseg_check_dice_coefficient_against_groundtruth():
    """Run the CLI script and verify that dice (computed against ground truth) is within a certain threshold."""
    sct_propseg.main(argv=['-i', 't2/t2.nii.gz', '-c', 't2', '-qc', 'testing-qc'])

    # open output segmentation
    im_seg = Image('t2_seg.nii.gz')
    # open ground truth
    im_seg_manual = Image('t2/t2_seg-manual.nii.gz')
    # compute dice coefficient between generated image and image from database
    dice_segmentation = compute_dice(im_seg, im_seg_manual, mode='3d', zboundaries=False)

    # note: propseg does *not* produce the same results across platforms, hence the 0.9 Dice threshold.
    # For more details, see: https://github.com/neuropoly/spinalcordtoolbox/issues/2769
    assert dice_segmentation > 0.9


@pytest.mark.sct_testing
def test_isct_propseg_compatibility():
    # TODO: Move this check to `sct_check_dependencies`. (It was in `sct_testing`, so it is put here for now.)
    status_isct_propseg, output_isct_propseg = run_proc('isct_propseg', verbose=0, raise_exception=False,
                                                        is_sct_binary=True)
    isct_propseg_version = output_isct_propseg.split('\n')[0]
    assert isct_propseg_version == 'sct_propseg - Version 1.1 (2015-03-24)', \
        'isct_propseg does not seem to be compatible with your system or is no up-to-date... Please contact SCT ' \
        'administrators.'


def test_sct_propseg_o_flags(tmp_path):
    argv = ['-i', 'sct_testing_data/t2/t2.nii.gz', '-c', 't2', '-ofolder', str(tmp_path), '-o', 'test_seg.nii.gz']
    sct_propseg.main(argv)
    output_files = sorted([f for f in os.listdir(tmp_path)
                          if os.path.isfile(os.path.join(tmp_path, f))])
    assert output_files == ['t2_centerline.nii.gz', 'test_seg.nii.gz']
