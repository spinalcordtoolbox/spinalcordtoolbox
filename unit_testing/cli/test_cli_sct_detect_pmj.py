from pytest_console_scripts import script_runner
import pytest
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_detect_pmj_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_detect_pmj')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''

def test_sct_detect_pmj_o_flag(tmp_path):
    command = 'sct_detect_pmj -i sct_testing_data/template/template//PAM50_small_t2.nii.gz -igt sct_testing_data/template/template/PAM50_small_t2_pmj_manual.nii.gz  -c t2 -o' + os.path.join(tmp_path, 'test_pmj.nii.gz')
    subprocess.check_output(command, shell=True)
    assert os.path.isfile(os.path.join(tmp_path, 'test_pmj.nii.gz'))
