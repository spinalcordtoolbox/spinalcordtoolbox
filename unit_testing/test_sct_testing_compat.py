from pytest_console_scripts import script_runner
import pytest
import logging

logger = logging.getLogger(__name__)

functions = [
    'sct_analyze_lesion',
    'sct_analyze_texture',
    'sct_apply_transfo',
    'sct_convert',
    'sct_compute_ernst_angle',
    'sct_compute_hausdorff_distance',
    'sct_compute_mtr',
    'sct_compute_mscc',
    'sct_compute_snr',
    'sct_create_mask',
    'sct_crop_image',
    'sct_dice_coefficient',
    'sct_detect_pmj',
    'sct_dmri_compute_dti',
    'sct_dmri_concat_b0_and_dwi',
    'sct_dmri_concat_bvals',
    'sct_dmri_concat_bvecs',
    'sct_dmri_compute_bvalue',
    'sct_dmri_moco',
    'sct_dmri_separate_b0_and_dwi',
    'sct_dmri_transpose_bvecs',
    'sct_extract_metric',
    'sct_flatten_sagittal',
    'sct_fmri_compute_tsnr',
    'sct_fmri_moco',
    'sct_get_centerline',
    'sct_image',
    'sct_label_utils',
    'sct_label_vertebrae',
    'sct_maths',
    'sct_merge_images',
    'sct_process_segmentation',
    'sct_propseg',
    'sct_qc',
    'sct_register_multimodal',
    'sct_register_to_template',
    'sct_resample',
    'sct_smooth_spinalcord',
    'sct_straighten_spinalcord',
    'sct_warp_template',
    'sct_deepseg_gm',
    'sct_deepseg_lesion',
    'sct_deepseg_sc',
]


@pytest.mark.script_launch_mode('subprocess')
@pytest.mark.parametrize("function", functions)
def test_backwards_compat(script_runner, function):
    ret = script_runner.run('sct_testing', '--function', function)
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success
    assert ret.stderr == ''
