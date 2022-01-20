import os
import logging

import pytest

from spinalcordtoolbox.image import Image, compute_dice
from spinalcordtoolbox.labels import check_missing_label
from spinalcordtoolbox.utils import sct_test_path
import spinalcordtoolbox.scripts.sct_label_vertebrae as sct_label_vertebrae

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_label_vertebrae_consistent_disc(tmp_path):
    """Check that all expected output labeled discs exist"""
    fname_ref = 't2/labels.nii.gz'
    sct_label_vertebrae.main(argv=['-i', 't2/t2.nii.gz', '-s', 't2/t2_seg-manual.nii.gz', '-c', 't2',
                                   '-discfile', fname_ref, '-ofolder', str(tmp_path)])
    ref = Image(fname_ref)
    pred = Image(os.path.join(tmp_path, 't2_seg-manual_labeled_discs.nii.gz'))
    fp, fn = check_missing_label(pred, ref)
    assert fp == []
    assert fn == []


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_label_vertebrae_initfile_qc_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_label_vertebrae.main(argv=['-i', 't2/t2.nii.gz', '-s', 't2/t2_seg-manual.nii.gz', '-c', 't2',
                                   '-initfile', 't2/init_label_vertebrae.txt', '-t', 'template', '-qc', 'testing-qc'])


def test_sct_label_vertebrae_initz_error():
    with pytest.raises(ValueError):
        sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                                  '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                  '-c', 't2', '-initz', '40'])


def test_sct_label_vertebrae_high_value_warning(caplog, tmp_path):
    sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                              '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                              '-c', 't2', '-initz', '40,19',
                              '-ofolder', str(tmp_path)])
    assert 'Disc value not included in template.' in caplog.text


def test_sct_label_vertebrae_clean_labels(tmp_path):
    sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                              '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                              '-c', 't2', '-initz', '40,3', '-clean-labels', '1',
                              '-ofolder',  str(tmp_path/'clean')])
    sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                              '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                              '-c', 't2', '-initz', '40,3', '-ofolder', str(tmp_path/'no_clean')])
    image_clean = Image(os.path.join(str(tmp_path), 'clean', 't2_seg-manual_labeled.nii.gz'))
    image_no_clean = Image(os.path.join(str(tmp_path), 'no_clean', 't2_seg-manual_labeled.nii.gz'))
    image_seg = Image(sct_test_path('t2', 't2_seg-manual.nii.gz'))
    # binarization (because label are between 3 and 6)
    image_clean.data = image_clean.data > 0.5
    image_no_clean.data = image_no_clean.data > 0.5
    dice_clean = compute_dice(image_clean, image_seg)
    dice_no_clean = compute_dice(image_no_clean, image_seg)
    # The cleaned version should be closer to the segmentation
    assert dice_clean >= dice_no_clean


def test_sct_label_vertebrae_disc_discontinuity_center_of_mass_error(tmp_path, caplog):
    # Generate a discontinuity next to an intervertebral disc
    t2_seg = Image(sct_test_path('t2', 't2_seg-manual.nii.gz'))
    t2_seg.data[:, 16, :] = 0
    path_out = str(tmp_path / 't2_seg-large-discontinuity.nii.gz')
    t2_seg.save(path=path_out)

    # Ensure the discontinuity is detected and an interpolated centerline is used instead
    sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'), '-s', path_out, '-c', 't2',
                              '-initfile', sct_test_path('t2', 'init_label_vertebrae.txt'),
                              '-ofolder', str(tmp_path)])
    assert "Using interpolated centerline" in caplog.text
