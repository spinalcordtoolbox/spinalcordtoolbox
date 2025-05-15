# pytest unit tests for spinalcordtoolbox.deepseg

import os

import pytest
import warnings
from torch.serialization import SourceChangeWarning

import spinalcordtoolbox as sct
from spinalcordtoolbox.image import Image, compute_dice, add_suffix, check_image_kind
from spinalcordtoolbox.utils.sys import sct_test_path
import spinalcordtoolbox.deepseg.models

from spinalcordtoolbox.scripts import sct_deepseg, sct_resample


def test_install_model():
    """
    Download all models, to allow subsequent tests on the model packages.
    :return:
    """
    for name_model, value in sct.deepseg.models.MODELS.items():
        if value['default']:
            sct.deepseg.models.install_model(name_model)
            # Make sure all files are present after unpacking the model
            assert sct.deepseg.models.is_valid(sct.deepseg.models.folder(name_model))


def test_model_dict():
    """
    Make sure all fields are present in each model.
    :return:
    """
    for key, value in sct.deepseg.models.MODELS.items():
        assert 'url' in value
        assert 'description' in value
        assert 'default' in value


@pytest.mark.parametrize('fname_image, fname_seg_manual, fname_out, task, thr, expected_dice', [
    (sct_test_path('t2s', 't2s_uncropped.nii.gz'),
     sct_test_path('t2s', 't2s_uncropped_gmseg_manual.nii.gz'),
     't2s_uncropped_seg_deepseg.nii.gz',
     'graymatter',
     None,
     0.91),  # Dice for GM is harder than SC seg due to complex GM shape
    (sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('t2', 't2_seg-manual.nii.gz'),
     't2_seg_deepseg.nii.gz',
     'spinalcord',
     None,
     0.95),
    (sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('t2', 't2_seg-deepseg_rootlets.nii.gz'),
     't2_seg_deepseg.nii.gz',
     'rootlets',
     None,
     None),  # no Dice score for rootlets model (we just make sure all the labels are present)
    (sct_test_path('t2', 't2.nii.gz'),  # dummy image since no EPI test data
     None,  # no ground truth, just test if it runs
     't2_seg_deepseg.nii.gz',
     'sc_epi',
     None,
     None),
    (sct_test_path('t2', 't2.nii.gz'),  # dummy image since no MP2RAGE test data
     None,  # no ground truth, just test if it runs
     't2_seg_deepseg.nii.gz',
     'lesion_ms_mp2rage',
     None,
     None),
    (sct_test_path('t2', 't2.nii.gz'),
     None,  # no ground truth, just test if it runs
     't2_seg_deepseg.nii.gz',
     'lesion_ms',
     None,
     None),
])
def test_segment_nifti_binary_seg(fname_image, fname_seg_manual, fname_out, task, thr, expected_dice,
                                  tmp_path):
    """
    Test binary output (produced using values other than `-thr 0`) with sct_deepseg postprocessing CLI arguments.
    """
    # Ignore warnings from ivadomed model source code changing
    warnings.filterwarnings("ignore", category=SourceChangeWarning)
    fname_out = str(tmp_path/fname_out)  # tmp_path for automatic cleanup
    args = [task, '-i', fname_image, '-o', fname_out, '-qc', str(tmp_path/'qc')]
    if thr is not None:
        args.extend(['-thr', str(thr)])
    if 'sc_' in task:
        # TODO: Replace the "general" testing of these arguments with specific tests with specific input data
        args.extend(['-largest', '1', '-fill-holes', '1', '-remove-small', '5mm3'])
    sct_deepseg.main(argv=args)
    # Make sure output file exists
    assert os.path.isfile(fname_out)
    # Compare with ground-truth segmentation if provided
    if fname_seg_manual:
        im_seg = Image(fname_out)
        im_seg_manual = Image(fname_seg_manual)
        output_type = check_image_kind(im_seg_manual)
        if output_type in ['seg', 'softseg']:
            dice_segmentation = compute_dice(im_seg, im_seg_manual, mode='3d', zboundaries=False)
            assert dice_segmentation > expected_dice
        else:
            # Confirm the output type is a labelled segmentation instead
            assert output_type == 'seg-labeled', f"ground truth is unexpected type {output_type}"

            # Get all labels in the segmentation, and those that we expect to see
            expected_labels = {coord.value for coord in im_seg_manual.getCoordinatesAveragedByValue()}
            detected_labels = {coord.value for coord in im_seg.getCoordinatesAveragedByValue()}

            # See if any labels we expected to see are missing
            missing_labels = expected_labels - detected_labels
            if len(missing_labels) > 0:
                pytest.fail(f"Test expected label(s) '{missing_labels}' in segmentation which were not present.")

            # See if any labels we weren't expecting showed up.
            unexpected_labels = detected_labels - expected_labels
            if len(unexpected_labels) > 0:
                warnings.warn(
                    f"Test produced label(s) '{unexpected_labels}' in segmentation which were not expected to appear."
                )


def t2_ax():
    """Generate an approximation of an axially-acquired T2w anat image using resampling."""
    fname_out = os.path.abspath('t2_ax.nii.gz')
    sct_resample.main(argv=["-i", sct_test_path('t2', 't2.nii.gz'), "-o", fname_out,
                            "-mm", "0.8x3x0.8", "-x", "spline"])
    return fname_out


def t2_ax_sc_seg():
    """Generate an approximation of an axially-acquired T2w segmentation using resampling."""
    fname_out = os.path.abspath('t2_ax_sc_seg.nii.gz')
    sct_resample.main(argv=["-i", sct_test_path('t2', 't2_seg-manual.nii.gz'), "-o", fname_out,
                            "-mm", "0.8x3x0.8", "-x", "spline"])
    return fname_out


@pytest.mark.parametrize('fname_image, fnames_seg_manual, fname_out, suffixes, task, thr, expected_dice, extra_args', [
    (sct_test_path('t2', 't2_fake_lesion.nii.gz'),
     [sct_test_path('t2', 't2_fake_lesion_sc_seg.nii.gz'),
      sct_test_path('t2', 't2_fake_lesion_lesion_seg.nii.gz')],
     't2_deepseg.nii.gz',
     ["_sc_seg", "_lesion_seg"],
     'lesion_sci_t2',
     0.5,
     0.95,
     []),
    (t2_ax(),          # Generate axial images on the fly
     [t2_ax_sc_seg(),  # Just test against SC ground truth, because the model generates SC segs well
      None],           # The model performs poorly on our fake t2_ax() image, so skip evaluating on lesion seg
     't2_deepseg.nii.gz',
     ["_sc_seg", "_lesion_seg"],
     'lesion_ms_axial_t2',
     0.5,
     0.94,  # axial model is just barely under .95, so we'll accept .94
     []),
    (sct_test_path('t1', 't1_mouse.nii.gz'),
     [None, None],
     't1_deepseg.nii.gz',
     ["_GM_seg", "_WM_seg"],
     'gm_wm_mouse_t1',
     0.5,
     None,
     []),
    (sct_test_path('t2', 't2.nii.gz'),
     [None, None, None, None, None],
     't2_deepseg.nii.gz',
     ["_step1_canal", "_step1_cord", "_step1_levels", "_step1_output", "_step2_output"],
     'totalspineseg',
     0,
     None,
     []),
    (sct_test_path('t2', 't2.nii.gz'),
     [None, None, None, None, None],
     't2_deepseg.nii.gz',
     ["_step1_canal", "_step1_cord", "_step1_levels", "_step1_output"],
     'totalspineseg',
     0,
     None,
     ["-step1-only", "1"]),
])
def test_segment_nifti_multiclass(fname_image, fnames_seg_manual, fname_out, suffixes, task, thr, expected_dice,
                                  extra_args, tmp_path):
    """
    Uses the locally-installed sct_testing_data
    """
    # Skip mouse test if the file is not present locally
    # (We do not include the file in sct_testing_data as A. the mouse image is large and B. inference time is lengthy.)
    # If testing locally, you can get this file from our internal testing dataset -> copy to sct_testing_data/t1/
    # More info here: https://github.com/spinalcordtoolbox/spinalcordtoolbox/wiki/Testing%253A-Datasets
    if "mouse" in task and not os.path.exists(fname_image):
        pytest.skip("Mouse data must be manually downloaded to run this test.")

    fname_out = str(tmp_path / fname_out)
    sct_deepseg.main([task, '-i', fname_image, '-thr', str(thr), '-o', fname_out, '-qc', str(tmp_path/'qc'),
                      '-largest', '1'] + extra_args)
    # The `-o` argument takes a single filename, even though one (or more!) files might be output.
    # If multiple output files will be produced, `sct_deepseg` will take this singular `-o` and add suffixes to it.
    fnames_out = [add_suffix(fname_out, suffix) for suffix in suffixes]
    for fname_out, fname_seg_manual in zip(fnames_out, fnames_seg_manual):
        # Make sure output file exists
        assert os.path.isfile(fname_out)
        # Compare with ground-truth segmentation if provided
        if fname_seg_manual:
            im_seg = Image(fname_out)
            im_seg_manual = Image(fname_seg_manual)
            dice_segmentation = compute_dice(im_seg, im_seg_manual, mode='3d', zboundaries=False)
            assert dice_segmentation > expected_dice


@pytest.mark.parametrize("qc_plane", ["Axial", "Sagittal"])
def test_deepseg_with_cropped_qc(qc_plane, tmp_path):
    """
    Test that `-qc-seg` cropping works with both Axial and Sagittal QCs.
    """
    fname_out = str(tmp_path / "t2_deepseg.nii.gz")
    sct_deepseg.main(['lesion_sci_t2',
                      '-i', sct_test_path('t2', 't2_fake_lesion.nii.gz'),
                      '-o', fname_out,
                      '-qc', str(tmp_path/'qc'),
                      '-qc-plane', qc_plane,
                      '-qc-seg', sct_test_path('t2', 't2_fake_lesion_sc_seg.nii.gz')])
