# pytest unit tests for spinalcordtoolbox.deepseg

import os

import pytest
import warnings
from torch.serialization import SourceChangeWarning

import spinalcordtoolbox as sct
from spinalcordtoolbox.image import Image, compute_dice, add_suffix, check_image_kind
from spinalcordtoolbox.utils.sys import sct_test_path
import spinalcordtoolbox.deepseg.models

from spinalcordtoolbox.scripts import sct_deepseg


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


@pytest.mark.parametrize('fname_image, fname_seg_manual, fname_out, task, thr', [
    (sct_test_path('t2s', 't2s.nii.gz'),
     sct_test_path('t2s', 't2s_seg-deepseg.nii.gz'),
     't2s_seg_deepseg.nii.gz',
     'seg_sc_t2star',
     0.9),
    (sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('t2', 't2_seg-manual.nii.gz'),
     't2_seg_deepseg.nii.gz',
     'seg_sc_contrast_agnostic',
     None),
    (sct_test_path('t2', 't2.nii.gz'),
     sct_test_path('t2', 't2_seg-deepseg_rootlets.nii.gz'),
     't2_seg_deepseg.nii.gz',
     'seg_spinal_rootlets_t2w',
     None),
])
def test_segment_nifti(fname_image, fname_seg_manual, fname_out, task, thr,
                       tmp_path):
    """
    Uses the locally-installed sct_testing_data
    """
    # Ignore warnings from ivadomed model source code changing
    warnings.filterwarnings("ignore", category=SourceChangeWarning)
    fname_out = str(tmp_path/fname_out)  # tmp_path for automatic cleanup
    args = ['-i', fname_image, '-task', task, '-o', fname_out]
    if thr is not None:
        args.extend(['-thr', str(thr)])
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
            assert dice_segmentation > 0.95
        else:
            assert output_type == 'seg-labeled', f"ground truth is unexpected type {output_type}"
            for coord_out, coord_gt in zip(im_seg.getCoordinatesAveragedByValue(),
                                           im_seg_manual.getCoordinatesAveragedByValue()):
                assert coord_out == coord_gt


@pytest.mark.parametrize('fname_image, fnames_seg_manual, fname_out, suffixes, task, thr', [
    (sct_test_path('t2', 't2.nii.gz'),
     [sct_test_path('t2', 't2_seg-manual.nii.gz'), None],  # TODO: Add ground truth for lesion seg
     't2_deepseg.nii.gz',
     ["_sc_seg", "_lesion_seg"],
     'seg_sc_lesion_t2w_sci',
     0.5),
])
def test_segment_nifti_multiclass(fname_image, fnames_seg_manual, fname_out, suffixes, task, thr,
                                  tmp_path):
    """
    Uses the locally-installed sct_testing_data
    """
    fname_out = str(tmp_path / fname_out)
    sct_deepseg.main(['-i', fname_image, '-task', task, '-thr', str(thr), '-o', fname_out])
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
            assert dice_segmentation > 0.95
