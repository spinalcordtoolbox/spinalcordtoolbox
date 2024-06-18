# pytest unit tests for sct_register_to_template

import os
import shutil
import glob
import logging

import numpy as np
import pytest

from spinalcordtoolbox.image import Image, compute_dice
from spinalcordtoolbox.utils.sys import __sct_dir__
from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.scripts import sct_register_to_template, sct_apply_transfo
from spinalcordtoolbox.labels import compute_mean_squared_error

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def template_lpi(tmp_path_factory):
    """Change orientation of test data template to LPI."""
    path_out = str(tmp_path_factory.mktemp("tmp_data")/'template_lpi')  # tmp_path_factory is needed for module scope
    shutil.copytree(sct_test_path('template'), path_out)
    for file in glob.glob(sct_test_path('template_lpi', 'template', '*.nii.gz')):
        nii = Image(file)
        nii.change_orientation('LPI')
        nii.save(file)
    return path_out


@pytest.fixture(scope="module")
def labels_discs(tmp_path_factory):
    """Create a disc labels file."""
    file_out = os.path.join(str(tmp_path_factory.mktemp("tmp_data")), 'labels_discs.nii.gz')
    im_labels = Image(sct_test_path('t2', 'labels.nii.gz'))
    im_labels.data = np.zeros(im_labels.data.shape)
    im_labels.data[30, 53, 26] = 3
    im_labels.data[31, 34, 26] = 4
    im_labels.data[31, 17, 26] = 5
    im_labels.data[32, 1, 26] = 6
    im_labels.save(file_out)
    return file_out


def test_sct_register_to_template_non_rpi_template(tmp_path, template_lpi):
    """Test registration with option -ref subject when template is not RPI orientation, causing #3300."""
    # Run registration to template using the RPI template as input file
    sct_register_to_template.main(argv=['-i', sct_test_path('template', 'template', 'PAM50_small_t2.nii.gz'),
                                        '-s', sct_test_path('template', 'template', 'PAM50_small_cord.nii.gz'),
                                        '-ldisc', sct_test_path('template', 'template', 'PAM50_small_label_disc.nii.gz'),
                                        '-c', 't2', '-t', template_lpi, '-ref', 'subject',
                                        '-param', 'step=1,type=seg,algo=centermass',
                                        '-ofolder', str(tmp_path), '-r', '0', '-v', '2'])
    img_orig = Image(sct_test_path('template', 'template', 'PAM50_small_t2.nii.gz'))
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
                                        '-c', 't2', '-t', sct_test_path('template'), '-ref', 'subject',
                                        '-param', 'step=1,type=seg,algo=centermass',
                                        '-ofolder', str(tmp_path), '-r', '0', '-v', '2'])
    img_orig = Image(f'{template_lpi}/template/PAM50_small_t2.nii.gz')
    img_reg = Image(str(tmp_path/'template2anat.nii.gz'))
    # Check if both images almost overlap. If they are right-left flipped, distance should be above threshold
    assert np.linalg.norm(img_orig.data - img_reg.data) < 1


@pytest.mark.sct_testing
@pytest.mark.parametrize("fname_gt, remaining_args", [
    (sct_test_path('template', 'template', 'PAM50_small_cord.nii.gz'),
     ['-l', sct_test_path('t2', 'labels.nii.gz'), '-t', sct_test_path('template'), '-qc', 'qc-testing', '-param',
      'step=1,type=seg,algo=centermassrot,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=5,metric=MeanSquares']),
    (os.path.join(__sct_dir__, 'data/PAM50/template/PAM50_cord.nii.gz'),
     ['-ldisc', sct_test_path('t2', 'labels.nii.gz'), '-ref', 'subject'])
])
def test_sct_register_to_template_dice_coefficient_against_groundtruth(fname_gt, remaining_args, tmp_path):
    """Run the CLI script and verify transformed images have expected attributes."""
    fname_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    dice_threshold = 0.9
    sct_register_to_template.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'),
                                        '-s', fname_seg, '-ofolder', str(tmp_path)]
                                  + remaining_args)

    # Straightening files are only generated for `-ref template`. They should *not* exist for `-ref subject`.
    for file in ["straightening.cache", "straight_ref.nii.gz",
                 "warp_straight2curve.nii.gz", "warp_curve2straight.nii.gz"]:
        assert os.path.isfile(tmp_path/file) == (False if 'subject' in remaining_args else True)

    # apply transformation to binary mask: template --> anat
    sct_apply_transfo.main(argv=[
        '-i', fname_gt,
        '-d', fname_seg,
        '-w', str(tmp_path/'warp_template2anat.nii.gz'),
        '-o', str(tmp_path/'test_template2anat.nii.gz'),
        '-x', 'nn',
        '-v', '0'])

    # apply transformation to binary mask: anat --> template
    sct_apply_transfo.main(argv=[
        '-i', fname_seg,
        '-d', fname_gt,
        '-w', str(tmp_path/'warp_anat2template.nii.gz'),
        '-o', str(tmp_path/'test_anat2template.nii.gz'),
        '-x', 'nn',
        '-v', '0'])

    # compute dice coefficient between template segmentation warped to anat and segmentation from anat
    im_seg = Image(fname_seg)
    im_template_seg_reg = Image(str(tmp_path/'test_template2anat.nii.gz'))
    dice_template2anat = compute_dice(im_seg, im_template_seg_reg, mode='3d', zboundaries=True)
    assert dice_template2anat > dice_threshold

    # compute dice coefficient between anat segmentation warped to template and segmentation from template
    im_seg_reg = Image(str(tmp_path/'test_anat2template.nii.gz'))
    im_template_seg = Image(fname_gt)
    dice_anat2template = compute_dice(im_seg_reg, im_template_seg, mode='3d', zboundaries=True)
    assert dice_anat2template > dice_threshold


def test_sct_register_to_template_mismatched_xforms(tmp_path, capsys):
    fname_mismatch = str(tmp_path / "t2_mismatched.nii.gz")
    im_in = Image(sct_test_path('t2', 't2.nii.gz'))
    qform = im_in.header.get_qform()
    qform[1, 3] += 10
    im_in.header.set_qform(qform)
    im_in.save(fname_mismatch)
    with pytest.raises(SystemExit):
        sct_register_to_template.main(argv=['-i', fname_mismatch,
                                            '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                            '-l', sct_test_path('t2', 'labels.nii.gz')])
    assert "Image sform does not match qform" in capsys.readouterr().out


def test_sct_register_to_template_3_labels(tmp_path, labels_discs):
    """Test registration with 3 labels."""
    sct_register_to_template.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'),
                                        '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                        '-ldisc', labels_discs,
                                        '-t', sct_test_path('template'),
                                        '-param', 'step=1,type=seg,algo=centermassrot,metric=MeanSquares:'
                                                  'step=2,type=seg,algo=bsplinesyn,iter=0,metric=MeanSquares',
                                        '-ofolder', str(tmp_path)])
    # Apply transformation to source labels
    sct_apply_transfo.main(argv=['-i', labels_discs,
                                 '-d', str(tmp_path/'anat2template.nii.gz'),
                                 '-w', str(tmp_path/'warp_anat2template.nii.gz'),
                                 '-o', str(tmp_path/'labels_discs_reg.nii.gz'),
                                 '-x', 'label'])
    # Compute pairwise distance between the template label and the registered label, ie: compute distance between dest
    # and src_reg for label of value '2', then '3', etc. and compute the mean square of all distances. The labels
    # should be touching, hence the mean square should be 0.
    im_label_dest = Image(sct_test_path('template', 'template', 'PAM50_small_label_disc.nii.gz'))
    im_label_src_reg = Image(str(tmp_path/'labels_discs_reg.nii.gz'))
    assert compute_mean_squared_error(im_label_dest, im_label_src_reg) == 0
