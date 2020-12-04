from pytest_console_scripts import script_runner
import pytest
import logging
import subprocess
import os

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_smooth_spinalcord_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_smooth_spinalcord')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


def test_sct_smooth_spinalcord_o_flag():
    command = 'sct_smooth_spinalcord -i sct_testing_data/t2/t2.nii.gz -s sct_testing_data/t2/t2_seg-manual.nii.gz -o test_smooth.nii.gz'
    subprocess.check_output(command, shell=True)
    assert os.path.isfile('test_smooth.nii.gz')
    os.remove('test_smooth.nii.gz')
