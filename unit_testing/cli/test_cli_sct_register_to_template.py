from pytest_console_scripts import script_runner
import shutil
import glob
import logging

import numpy as np
import pytest

import spinalcordtoolbox.scripts.sct_register_to_template as sct_register_to_template
from spinalcordtoolbox.image import Image

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_register_to_template_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_register_to_template')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


@pytest.fixture(scope="module")
def template_lpi(tmp_path_factory):
    """Change orientation of test data template to LPI."""
    path_out = tmp_path_factory.mktemp("tmp_data")/'template_lpi'  # tmp_path_factory is needed for module scope
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
                                        '-param', 'step=1,type=seg,algo=centermass', '-r', '0', '-v', '2'])
    img_orig = Image('sct_testing_data/template/template/PAM50_small_t2.nii.gz')
    img_reg = Image('template2anat.nii.gz')
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
                                        '-param', 'step=1,type=seg,algo=centermass', '-r', '0', '-v', '2'])
    img_orig = Image(f'{template_lpi}/template/PAM50_small_t2.nii.gz')
    img_reg = Image('template2anat.nii.gz')
    # Check if both images almost overlap. If they are right-left flipped, distance should be above threshold
    assert np.linalg.norm(img_orig.data - img_reg.data) < 1
