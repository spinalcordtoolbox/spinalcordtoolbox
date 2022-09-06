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
    with pytest.raises(SystemExit) as excinfo:
        sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                                  '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                  '-c', 't2', '-initz', '40'])
    # The exit code when argparse prints a usage message is 2
    assert excinfo.value.code == 2


# At least for the (default) PAM50 template, there are 3 cases
# when using a high initial disc value:
#
# value=19: the code does the 'superior' direction of the loop,
#   then does two iterations in the 'inferior' direction, and emits
#   a warning for 'Disc value not included in template.'
#
# value=20: the code does the 'superior' direction of the loop,
#   then tries to switch to the 'inferior' direction, but
#   notices that the initial disc was the most 'inferior' disc already.
#   This is arguably not an error but may be surprising,
#   so the code emits an informational log.
#
# value=21 (or more): the initial disc value is outside the
#   template, so the code detects this condition and errors out
#   with a sensible error message.

def test_sct_label_vertebrae_high_value_warning(caplog, tmp_path):
    sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                              '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                              '-c', 't2', '-initz', '40,19',
                              '-ofolder', str(tmp_path)])
    assert 'Disc value not included in template.' in caplog.text


def test_sct_label_vertebrae_initial_disc_no_inferior(caplog, tmp_path):
    caplog.set_level(logging.INFO)
    sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                              '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                              '-c', 't2', '-initz', '40,20',
                              '-ofolder', str(tmp_path)])
    assert 'No disc is inferior to the initial disc.' in caplog.text


def test_sct_label_vertebrae_initial_disc_too_high(capsys, tmp_path):
    with pytest.raises(Exception) as e:
        sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                                  '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                  '-c', 't2', '-initz', '40,21',
                                  '-ofolder', str(tmp_path)])
    assert 'Initial disc is not in template.' in str(e)


def test_sct_label_vertebrae_initial_disc_zero(capsys, tmp_path):
    with pytest.raises(Exception) as e:
        sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                                  '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                  '-c', 't2', '-initz', '40,0',
                                  '-ofolder', str(tmp_path)])
    assert 'Missing label or zero label for initial disc.' in str(e)


def test_sct_label_vertebrae_initial_disc_too_low(capsys, tmp_path):
    with pytest.raises(Exception) as e:
        sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                                  '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                  '-c', 't2', '-initz', '40,-1',
                                  '-ofolder', str(tmp_path)])
    assert 'Initial disc is not in template.' in str(e)


def test_sct_label_vertebrae_clean_labels(tmp_path):
    im_seg = Image(sct_test_path('t2', 't2_seg-manual.nii.gz'))
    dice_score = {}
    for i in [0, 1, 2]:
        sct_label_vertebrae.main(['-i', sct_test_path('t2', 't2.nii.gz'),
                                  '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                  '-c', 't2',
                                  '-initz', '40,3',
                                  '-clean-labels', str(i),
                                  '-ofolder', str(tmp_path/str(i))])
        im_labeled_seg = Image(str(tmp_path/str(i)/'t2_seg-manual_labeled.nii.gz'))
        # binarization (because labels are 2.0, 3.0, 4.0)
        im_labeled_seg.data = im_labeled_seg.data > 0.5
        dice_score[i] = compute_dice(im_labeled_seg, im_seg)
    # the cleaner version should be closer to the segmentation
    assert dice_score[2] >= dice_score[1]
    assert dice_score[1] >= dice_score[0]


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
