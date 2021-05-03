import pytest
import logging
import os

from spinalcordtoolbox.scripts import sct_smooth_spinalcord

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_smooth_spinalcord_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_smooth_spinalcord.main(argv=['-i', 't2/t2.nii.gz', '-s', 't2/t2_seg-manual.nii.gz', '-smooth', '0,0,5'])


def test_sct_smooth_spinalcord_o_flag(tmp_path):
    argv = ['-i', 'sct_testing_data/t2/t2.nii.gz', '-s', 'sct_testing_data/t2/t2_seg-manual.nii.gz',
               '-o', os.path.join(str(tmp_path), "test_smooth.nii")]
    sct_smooth_spinalcord.main(argv)
    assert os.path.isfile(os.path.join(str(tmp_path), "test_smooth.nii"))

    # Files created in root directory by sct_smooth_spinalcord
    os.unlink('straightening.cache')
    os.unlink('warp_straight2curve.nii.gz')
    os.unlink('warp_curve2straight.nii.gz')
