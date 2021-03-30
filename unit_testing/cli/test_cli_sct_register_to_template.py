from pytest_console_scripts import script_runner
import shutil
import glob
import numpy as np
import pytest
import logging
import spinalcordtoolbox.scripts.sct_register_to_template as sct_register_to_template
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.labels as sct_labels
logger = logging.getLogger(__name__)

# def change_orientation


# @pytest.mark.script_launch_mode('subprocess')
# def test_sct_register_to_template_backwards_compat(script_runner):
#     ret = script_runner.run('sct_testing', '--function', 'sct_register_to_template')
#     logger.debug(f"{ret.stdout}")
#     logger.debug(f"{ret.stderr}")
#     assert ret.success
#     assert ret.stderr == ''


def test_sct_register_to_template_non_rpi_template(tmp_path):
    """Test registration with option -ref subject when template is not RPI orientation, causing #3300."""
    # Change orientation of test data template to LPI
    shutil.copytree('sct_testing_data/template', 'sct_testing_data/template_lpi')
    for file in glob.glob('sct_testing_data/template_lpi/template/*.nii.gz'):
        nii = Image(file)
        nii.change_orientation('LPI')
        nii.save(file)
    # Create label for registration by taking the center of mass of each vertebral level
    sct_labels.label_vertebrae(Image('sct_testing_data/template_lpi/template/PAM50_small_levels.nii.gz'), 0).\
        save('sct_testing_data/template_lpi/template/labels_vert.nii.gz')
    # Run registration to template using the RPI template as input file
    command = '-i sct_testing_data/template/template/PAM50_small_t2.nii.gz ' \
              '-s sct_testing_data/template/template/PAM50_small_cord.nii.gz ' \
              '-ldisc sct_testing_data/template/template/PAM50_small_label_disc.nii.gz ' \
              '-c t2 -t sct_testing_data/template_lpi -ref subject ' \
              '-param step=1,type=seg,algo=centermass -r 0 -v 2'
    sct_register_to_template.main(command.split())
    img_orig = Image('sct_testing_data/template/template/PAM50_small_t2.nii.gz')
    img_reg = Image('template2anat.nii.gz')
    # Check if both images almost overlap. If they are right-left flipped, distance should be above threshold
    # TODO: adjust threshold once fix is found
    assert np.linalg.norm(img_orig.data - img_reg.data) < 1

