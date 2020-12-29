from pytest_console_scripts import script_runner
import pytest
import logging
import subprocess
import os
from spinalcordtoolbox.scripts import sct_smooth_spinalcord as sct_smooth_spinalcord

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_smooth_spinalcord_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_smooth_spinalcord')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


def test_sct_smooth_spinalcord_o_flag(tmp_path):
    command = '-i sct_testing_data/t2/t2.nii.gz -s sct_testing_data/t2/t2_seg-manual.nii.gz -o' + os.path.join(str(tmp_path), "test_smooth.nii")
    sct_smooth_spinalcord.main(command.split())
    assert os.path.isfile(os.path.join(str(tmp_path), "test_smooth.nii"))
