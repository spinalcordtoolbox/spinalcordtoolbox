from pytest_console_scripts import script_runner
import pytest
import logging
import os
import subprocess
import numpy as np
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.scripts.sct_label_vertebrae as sct_label_vertebrae
logger = logging.getLogger(__name__)


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


#def test_sct_label_vertebrae_high_value_warning(caplog):
 #   command = '-i sct_testing_data/t2/t2.nii.gz -s sct_testing_data/t2/t2_seg-manual.nii.gz -c t2 -initz 40,20'
  #  s,o = sct_label_vertebrae.main(command.split())
   # assert 'Disc value not included in template.' in caplog.text


