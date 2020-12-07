from pytest_console_scripts import script_runner
import pytest
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_deepseg_sc_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_deepseg_sc')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


def test_sct_deepseg_sc_o_flag(tmp_path):
    command = """sct_deepseg_sc -i sct_testing_data/t2/t2.nii.gz -c t2 -o""" + os.path.join(tmp_path, 'test_seg.nii.gz')
    subprocess.check_output(command, shell=True)
    assert os.path.isfile(os.path.join(tmp_path, 'test_seg.nii.gz'))

