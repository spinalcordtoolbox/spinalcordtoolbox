import numpy as np
from pytest_console_scripts import script_runner
import os
import pytest
import logging

from spinalcordtoolbox.math import dice
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.deepseg_sc.core import deep_segmentation_spinalcord
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
    """
    Run the script without validating results.

    - TODO: Write a check that verifies the registration results as part of
            https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3246.
    - TODO: Parametrize this test to add '-initwarpinv warp_anat2template.nii.gz',
            after the file is added to sct_testing_data:
            https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3407#discussion_r646895013
    """
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


@pytest.mark.parametrize('algo', ['affine', 'bsplinesyn'])
@pytest.mark.parametrize('mask_type', ['cylinder', 'gaussian'])
def test_sct_register_multimodal_with_masks(tmp_path, algo, mask_type):
    """
    Verify that masks are actually applied during registration, and have a positive effect on the accuracy.

    For 'gaussian', ANTs binaries can't handle softmasks natively, so SCT should (in theory) be applying
    the mask directly to the image. Related links:
        * https://github.com/ivadomed/pipeline-hemis/issues/3.
        * https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3075

    TODO: This test is somewhat redundant with the other mask test, so consider ways to refactor.
          (Though, the other test uses MTI data, and uses -initwarp, so it has some distinctions.)
    """
    fname_mask = str(tmp_path/'mask_t2.nii.gz')
    fname_t2 = sct_test_path('t2', 't2.nii.gz')
    fname_t1 = sct_test_path('t1', 't1w.nii.gz')
    fname_t1_reg = str(tmp_path/"t1w_reg.nii.gz")
    fname_warp = str(tmp_path/"warp_t1w2t2.nii.gz")

    sct_create_mask.main(['-i', fname_t2, '-p', f"centerline,{sct_test_path('t2', 't2_centerline-manual.nii.gz')}",
                          '-o', fname_mask, '-f', mask_type])
    sct_register_multimodal.main(['-i', fname_t1, '-d', fname_t2, '-dseg', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                  '-param', f"step=1,type=im,algo={algo},metric=CC", '-m', fname_mask,
                                  '-ofolder', str(tmp_path), '-r', '0', '-v', '2'])

    # If registration was successful, the warping field should be non-empty
    assert np.any(Image(fname_warp).data)

    # As a baseline, failed registation (no translation at all) has a dice score of 0.851. (This was tested using the
    # conditions from https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3075#issuecomment-869083958.)
    # If translation does occur, then the dice scores for every case should be improved above the threshold.
    im_t2_seg, _, _ = deep_segmentation_spinalcord(Image(fname_t2), contrast_type='t2', ctr_algo='svm')
    im_t1_reg_seg, _, _ = deep_segmentation_spinalcord(Image(fname_t1_reg), contrast_type='t1', ctr_algo='svm')
    dice_score_t1_reg = dice(im_t2_seg.data, im_t1_reg_seg.data)
    assert dice_score_t1_reg > 0.87
