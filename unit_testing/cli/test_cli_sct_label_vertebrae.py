from pytest_console_scripts import script_runner
import pytest
import logging
import subprocess
from spinalcordtoolbox.utils import init_sct, run_proc
logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_label_vertebrae_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_label_vertebrae')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


def test_sct_label_vertebrae_initz_error():
    a = 'sct_label_vertebrae -i sct_testing_data/t2/t2.nii.gz -s sct_testing_data/t2/t2_seg-manual.nii.gz -c t2 -initz 40'
    proc = subprocess.Popen(a.split(' '),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            )
    out, err = proc.communicate()
    assert 'ValueError: --initz takes two arguments: position in superior-inferior direction, label value' in str(err)
