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


def test_sct_label_vertebrae_disc():
    subprocess.check_output("sct_label_vertebrae -i sct_testing_data/t2/t2.nii.gz "
                            "-s sct_testing_data/t2/t2_seg-manual.nii.gz "
                            "-initfile sct_testing_data/t2/init_label_vertebrae.txt -c t2 -ofolder label_test",
                            shell=True)
    nifti_label = nib.load("label_test/t2_labels-disc.nii.gz")
    image_label = np.array(nifti_label.dataobj)
    IS_coordinates = np.nonzero(image_label)[1]  # image is AIL
    IS_coordinates.sort()
    real_IS_coordinates = [2, 17, 34, 48]
    assert np.allclose(real_IS_coordinates, IS_coordinates)
    # cleaning
    subprocess.check_output("rm -r label_test", shell=True)

