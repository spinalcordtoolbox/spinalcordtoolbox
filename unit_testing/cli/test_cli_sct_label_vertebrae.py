from pytest_console_scripts import script_runner
import pytest
import logging
import os
import subprocess
import numpy as np
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.scripts.sct_label_vertebrae as sct_label_vertebrae
logger = logging.getLogger(__name__)
import os
import nibabel as nib
from spinalcordtoolbox.image import Image, compute_dice
from spinalcordtoolbox.labels import check_missing_label

@pytest.mark.script_launch_mode('subprocess')
def test_sct_label_vertebrae_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_label_vertebrae')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


@pytest.mark.parametrize("contrast,is_coordinates_gt, pref", [("t2", [2, 17, 34, 48], "t2"),
                                                               ("t1", [2, 17, 35, 55, 68], "t1w")])
def test_sct_label_vertebrae_disc(tmp_path, is_coordinates_gt, contrast, pref):
    d = tmp_path / "sub"
    print(is_coordinates_gt)
    d.mkdir()
    subprocess.run("sct_label_vertebrae -i sct_testing_data/" + contrast+"/" + pref + ".nii.gz "
                   "-s sct_testing_data/" + contrast + "/" + pref + "_seg-manual.nii.gz "
                   "-initfile sct_testing_data/" + contrast + "/init_label_vertebrae.txt -c " + contrast +
                   " -ofolder " + str(d), shell=True)
    nifti_label = Image(str(d) + "/" + pref + "_labels-disc.nii.gz")
    nifti_label = nifti_label.change_orientation("RPI")
    image_label = np.array(nifti_label.data)
    IS_coordinates = np.nonzero(image_label)[2]  # image is AIL
    IS_coordinates.sort()
    print(IS_coordinates)
    assert np.allclose(is_coordinates_gt, IS_coordinates, atol=2)
    assert os.path.isfile(str(d) + "/warp_straight2curve.nii.gz")
    assert os.path.isfile(str(d) + "/warp_curve2straight.nii.gz")
    assert os.path.isfile(str(d) + "/straight_ref.nii.gz")


def test_sct_label_vertebrae_initz_error():
    command = '-i sct_testing_data/t2/t2.nii.gz -s sct_testing_data/t2/t2_seg-manual.nii.gz -c t2 -initz 40'
    with pytest.raises(ValueError):
        sct_label_vertebrae.main(command.split())


def test_sct_label_vertebrae_high_value_warning(caplog):
    command = '-i sct_testing_data/t2/t2.nii.gz -s sct_testing_data/t2/t2_seg-manual.nii.gz -c t2 -initz 40,19 -scale-dist 0.2'
    sct_label_vertebrae.main(command.split())
    assert 'Disc value not included in template.' in caplog.text


def test_sct_label_vertebrae_clean_labels(tmp_path):
    command = '-i sct_testing_data/t2/t2.nii.gz -s sct_testing_data/t2/t2_seg-manual.nii.gz -c t2 -initz 40,3 -clean-labels 1 -ofolder ' + str(os.path.join(str(tmp_path), 'clean'))
    sct_label_vertebrae.main(command.split())
    command = '-i sct_testing_data/t2/t2.nii.gz -s sct_testing_data/t2/t2_seg-manual.nii.gz -c t2 -initz 40,3 -ofolder ' + str(os.path.join(str(tmp_path), 'no_clean'))
    sct_label_vertebrae.main(command.split())
    image_clean = Image(os.path.join(str(tmp_path), 'clean', 't2_seg-manual_labeled.nii.gz'))
    image_no_clean = Image(os.path.join(str(tmp_path), 'no_clean', 't2_seg-manual_labeled.nii.gz'))
    image_seg = Image(os.path.join('sct_testing_data', 't2', 't2_seg-manual.nii.gz'))
    # binarization (because label are between 3 and 6)
    image_clean.data = image_clean.data > 0.5
    image_no_clean.data = image_no_clean.data > 0.5
    dice_clean = compute_dice(image_clean, image_seg)
    dice_no_clean = compute_dice(image_no_clean, image_seg)
    # The cleaned version should be closer to the segmentation
    assert dice_clean >= dice_no_clean


def test_sct_label_vertebrae_consistent_disc(tmp_path):
    """Check that all expected output labeled discs exist"""
    command = '-i sct_testing_data/t2/t2.nii.gz -s sct_testing_data/t2/t2_seg-manual.nii.gz -c t2 -discfile sct_testing_data/t2/labels.nii.gz -ofolder ' + str(tmp_path)
    sct_label_vertebrae.main(command.split())
    ref = Image('sct_testing_data/t2/labels.nii.gz')
    pred = Image(os.path.join(tmp_path, 't2_seg-manual_labeled_discs.nii.gz'))
    fp, fn = check_missing_label(pred, ref)
    assert fp == []
    assert fn == []
