from pytest_console_scripts import script_runner
import pytest
import logging
import os

from spinalcordtoolbox.scripts import sct_image
from spinalcordtoolbox.utils.sys import sct_test_path

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
    """Run the CLI script without checking results. The rationale for not checking results is
    provided here: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3317#issuecomment-811429547"""
    sct_image.main(argv=['-i', 'sct_testing_data/t2/t2.nii.gz', '-header', output_format])


def test_sct_image_display_warp_check_output_exists():
    """Run the CLI script and check that the warp image file was created."""
    fname_in = 'warp_template2anat.nii.gz'
    fname_out = 'grid_3_resample_' + fname_in
    sct_image.main(argv=['-i', sct_test_path('t2', fname_in), '-display-warp'])
    assert os.path.exists(sct_test_path('t2', fname_out))
