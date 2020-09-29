from pytest_console_scripts import script_runner
import pytest
import logging

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_extract_metric_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_extract_metric')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''
