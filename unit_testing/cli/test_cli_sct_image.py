from pytest_console_scripts import script_runner
import pytest
import logging

from spinalcordtoolbox.scripts import sct_image

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_image_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_image')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


@pytest.mark.parametrize("output_format", ('sct', 'fslhd', 'nibabel'))
def test_sct_image_show_header_no_checks(output_format):
    """Run the CLI script without checking results."""
    sct_image.main(argv=['-i', 'sct_testing_data/t2/t2.nii.gz', '-show-header', output_format])
