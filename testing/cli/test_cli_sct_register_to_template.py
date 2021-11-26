import os
import shutil
import glob
import logging

import numpy as np
import pytest

from spinalcordtoolbox.image import Image, compute_dice
from spinalcordtoolbox import __sct_dir__

from spinalcordtoolbox.scripts import sct_register_to_template, sct_apply_transfo

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def template_lpi(tmp_path_factory):
    """Change orientation of test data template to LPI."""
    path_out = str(tmp_path_factory.mktemp("tmp_data")/'template_lpi')  # tmp_path_factory is needed for module scope
    shutil.copytree('sct_testing_data/template', path_out)
    for file in glob.glob('sct_testing_data/template_lpi/template/*.nii.gz'):
        nii = Image(file)
        nii.change_orientation('LPI')
        nii.save(file)
    return path_out


def test_sct_register_to_template_non_rpi_template(tmp_path, template_lpi):
    """Test registration with option -ref subject when template is not RPI orientation, causing #3300."""
    # Run registration to template using the RPI template as input file
    sct_register_to_template.main(argv=['-i', 'sct_testing_data/template/template/PAM50_small_t2.nii.gz',
                                        '-s', 'sct_testing_data/template/template/PAM50_small_cord.nii.gz',
                                        '-ldisc', 'sct_testing_data/template/template/PAM50_small_label_disc.nii.gz',
                                        '-c', 't2', '-t', template_lpi, '-ref', 'subject',
                                        '-param', 'step=1,type=seg,algo=centermass',
                                        '-ofolder', str(tmp_path), '-r', '0', '-v', '2'])
    img_orig = Image('sct_testing_data/template/template/PAM50_small_t2.nii.gz')
    img_reg = Image(str(tmp_path/'template2anat.nii.gz'))
    # Check if both images almost overlap. If they are right-left flipped, distance should be above threshold
    assert np.linalg.norm(img_orig.data - img_reg.data) < 1


def test_sct_register_to_template_non_rpi_data(tmp_path, template_lpi):
    """
    Test registration with option -ref subject when data is not RPI orientation.
    This test uses the temporary dataset created in test_sct_register_to_template_non_rpi_template().
    """
    # Run registration to template using the LPI template as input file
    sct_register_to_template.main(argv=['-i', f'{template_lpi}/template/PAM50_small_t2.nii.gz',
                                        '-s', f'{template_lpi}/template/PAM50_small_cord.nii.gz',
                                        '-ldisc', f'{template_lpi}/template/PAM50_small_label_disc.nii.gz',
                                        '-c', 't2', '-t', 'sct_testing_data/template', '-ref', 'subject',
                                        '-param', 'step=1,type=seg,algo=centermass',
                                        '-ofolder', str(tmp_path), '-r', '0', '-v', '2'])
    img_orig = Image(f'{template_lpi}/template/PAM50_small_t2.nii.gz')
    img_reg = Image(str(tmp_path/'template2anat.nii.gz'))
    # Check if both images almost overlap. If they are right-left flipped, distance should be above threshold
    assert np.linalg.norm(img_orig.data - img_reg.data) < 1


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
    dice_template2anat = compute_dice(im_seg, im_template_seg_reg, mode='3d', zboundaries=True)
    assert dice_template2anat > dice_threshold

    # compute dice coefficient between anat segmentation warped to template and segmentation from template
    im_seg_reg = Image('test_anat2template.nii.gz')
    im_template_seg = Image(fname_gt)
    dice_anat2template = compute_dice(im_seg_reg, im_template_seg, mode='3d', zboundaries=True)
    assert dice_anat2template > dice_threshold
