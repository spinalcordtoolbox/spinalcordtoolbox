import os
import logging

import pytest

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from spinalcordtoolbox import __sct_dir__

from spinalcordtoolbox.scripts import sct_register_to_template, sct_apply_transfo

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
@pytest.mark.parametrize("fname_gt, remaining_args", [
    ('template/template/PAM50_small_cord.nii.gz',
     ['-l', 't2/labels.nii.gz', '-t', 'template', '-qc', 'qc-testing', '-param',
      'step=1,type=seg,algo=centermassrot,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=5,metric=MeanSquares']),
    (os.path.join(__sct_dir__, 'data/PAM50/template/PAM50_cord.nii.gz'),
     ['-ldisc', 't2/labels.nii.gz', '-ref', 'subject'])
])
def test_sct_register_to_template_dice_coefficient_against_groundtruth(fname_gt, remaining_args):
    """Run the CLI script and verify transformed images have expected attributes."""
    fname_seg = 't2/t2_seg-manual.nii.gz'
    dice_threshold = 0.9
    sct_register_to_template.main(argv=['-i', 't2/t2.nii.gz', '-s', fname_seg] + remaining_args)

    # apply transformation to binary mask: template --> anat
    sct_apply_transfo.main(argv=[
        '-i', fname_gt,
        '-d', fname_seg,
        '-w', 'warp_template2anat.nii.gz',
        '-o', 'test_template2anat.nii.gz',
        '-x', 'nn',
        '-v', '0'])

    # apply transformation to binary mask: anat --> template
    sct_apply_transfo.main(argv=[
        '-i', fname_seg,
        '-d', fname_gt,
        '-w', 'warp_anat2template.nii.gz',
        '-o', 'test_anat2template.nii.gz',
        '-x', 'nn',
        '-v', '0'])

    # compute dice coefficient between template segmentation warped to anat and segmentation from anat
    im_seg = Image(fname_seg)
    im_template_seg_reg = Image('test_template2anat.nii.gz')
    dice_template2anat = msct_image.compute_dice(im_seg, im_template_seg_reg, mode='3d', zboundaries=True)
    assert dice_template2anat > dice_threshold

    # compute dice coefficient between anat segmentation warped to template and segmentation from template
    im_seg_reg = Image('test_anat2template.nii.gz')
    im_template_seg = Image(fname_gt)
    dice_anat2template = msct_image.compute_dice(im_seg_reg, im_template_seg, mode='3d', zboundaries=True)
    assert dice_anat2template > dice_threshold

