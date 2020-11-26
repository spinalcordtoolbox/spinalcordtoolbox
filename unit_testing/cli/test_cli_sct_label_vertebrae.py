from pytest_console_scripts import script_runner
import pytest
import logging
import sct_label_vertebrae
logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_label_vertebrae_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_label_vertebrae')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


def test_sct_label_vertebrae_initz_error():
    command = '-i sct_testing_data/t2/t2.nii.gz -s sct_testing_data/t2/t2_seg-manual.nii.gz -c t2 -initz 40'
    with pytest.raises(ValueError):
        sct_label_vertebrae.main(command.split())
