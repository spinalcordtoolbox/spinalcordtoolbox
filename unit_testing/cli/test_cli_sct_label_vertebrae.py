from pytest_console_scripts import script_runner
import pytest
import logging
import os
import subprocess
import nibabel as nib
import numpy as np


logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_label_vertebrae_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_label_vertebrae')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


@pytest.mark.parametrize("is_coordinates_gt", [[2, 17, 34, 48]])
def test_sct_label_vertebrae_disc(tmpdir, is_coordinates_gt):
    p = tmpdir.mkdir("label_test")
    subprocess.check_output("sct_label_vertebrae -i sct_testing_data/t2/t2.nii.gz "
                            "-s sct_testing_data/t2/t2_seg-manual.nii.gz "
                            "-initfile sct_testing_data/t2/init_label_vertebrae.txt -c t2 -ofolder " + str(p), shell=True)
    nifti_label = nib.load(str(p) + "/t2_labels-disc.nii.gz")
    image_label = np.array(nifti_label.dataobj)
    IS_coordinates = np.nonzero(image_label)[1]  # image is AIL
    IS_coordinates.sort()
    assert np.allclose(is_coordinates_gt, IS_coordinates, atol=2)
    assert os.path.isfile(str(p) + "/warp_straight2curve.nii.gz")
    assert os.path.isfile(str(p) + "/warp_curve2straight.nii.gz")
    assert os.path.isfile(str(p) + "/straight_ref.nii.gz")

