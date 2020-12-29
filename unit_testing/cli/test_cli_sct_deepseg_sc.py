from pytest_console_scripts import script_runner
import pytest
import logging
import os
import subprocess
from spinalcordtoolbox.scripts import sct_deepseg_sc as sct_deepseg_sc

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_deepseg_sc_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_deepseg_sc')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


def test_sct_deepseg_sc_o_flag(tmp_path):
    command = """-i sct_testing_data/t2/t2.nii.gz -c t2 -o""" + str(os.path.join(str(tmp_path), 'test_seg.nii.gz'))
    sct_deepseg_sc.main(command.split())
    assert os.path.isfile(os.path.join(str(tmp_path), 'test_seg.nii.gz'))

