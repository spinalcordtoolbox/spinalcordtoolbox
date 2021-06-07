from pytest_console_scripts import script_runner
import os
import pytest
import logging

from spinalcordtoolbox.utils import sct_test_path, sct_dir_local_path
from spinalcordtoolbox.scripts import sct_register_multimodal, sct_create_mask

logger = logging.getLogger(__name__)


@pytest.mark.script_launch_mode('subprocess')
def test_sct_register_multimodal_backwards_compat(script_runner):
    ret = script_runner.run('sct_testing', '--function', 'sct_register_multimodal')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''


def test_sct_register_multimodal_mask_files_exist(tmp_path):
    """Run the script without validating results.

    TODO: Write a check that verifies the registration results as part of
    https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3246."""
    fname_mask = str(tmp_path/'mask_mt1.nii.gz')
    sct_create_mask.main(['-i', sct_test_path('mt', 'mt1.nii.gz'),
                          '-p', f"centerline,{sct_test_path('mt', 'mt1_seg.nii.gz')}",
                          '-size', '35mm', '-f', 'cylinder', '-o', fname_mask])
    sct_register_multimodal.main([
        '-i', sct_dir_local_path('data/PAM50/template/', 'PAM50_t2.nii.gz'),
        '-iseg', sct_dir_local_path('data/PAM50/template/', 'PAM50_cord.nii.gz'),
        '-d', sct_test_path('mt', 'mt1.nii.gz'),
        '-dseg', sct_test_path('mt', 'mt1_seg.nii.gz'),
        '-param', 'step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3',
        '-m', fname_mask,
        '-initwarp', sct_test_path('t2', 'warp_template2anat.nii.gz'),
        '-ofolder', str(tmp_path)
    ])

    for path in ["PAM50_t2_reg.nii.gz", "warp_PAM50_t22mt1.nii.gz"]:
        assert os.path.exists(tmp_path/path)

    # Because `-initwarp` was specified (but `-initwarpinv` wasn't) the dest->seg files should NOT exist
    for path in ["mt1_reg.nii.gz", "warp_mt12PAM50_t2.nii.gz"]:
        assert not os.path.exists(tmp_path/path)
