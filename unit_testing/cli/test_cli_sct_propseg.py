from pytest_console_scripts import script_runner
import pytest
import logging
import subprocess
import os
from spinalcordtoolbox.scripts import sct_propseg

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_propseg_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_propseg')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


def test_sct_propseg_o_flag():
    command = """-i sct_testing_data/t2/t2.nii.gz -c t2 -o test_seg.nii.gz"""
    sct_propseg.main(command.split())
    assert os.path.isfile("test_seg.nii.gz")
    os.remove("test_seg.nii.gz")
