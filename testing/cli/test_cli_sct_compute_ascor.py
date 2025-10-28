# pytest unit tests for sct_compute_ascor

import pytest
import logging
import csv

from spinalcordtoolbox.scripts import sct_compute_ascor, sct_crop_image, sct_deepseg, sct_label_utils
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def t2_canal_seg(tmp_path_factory):
    """Mean segmented image for QC report generation."""
    tmp_path = tmp_path_factory.mktemp('t2_canal_seg')
    path_out = str(tmp_path / 't2_canal_seg.nii.gz')
    sct_deepseg.main(argv=['sc_canal_t2', '-i', sct_test_path('t2', 't2.nii.gz'),
                     '-o', path_out, '-qc', str(tmp_path)])
    return path_out


@pytest.fixture(scope='module')
def t2_disc_labels(tmp_path_factory):
    """Point label image identifying the posterior tip of each intervertebral disc."""
    tmp_path = tmp_path_factory.mktemp('t2_canal_seg')
    fname_anat = sct_test_path('t2', 't2.nii.gz')
    fname_labels = str(tmp_path / 't2_disc_labels.nii.gz')
    sct_label_utils.main([
        '-i', fname_anat,
        '-o', fname_labels,
        '-create', ':'.join(["22,52,26,2", "22,33,24,3", "23,16,24,4", "24,1,24,5"])
    ])
    return fname_labels


def test_sct_process_segmentation_check_discfile_perslice(tmp_path, t2_canal_seg):
    """Run sct_compute_ascor with `-discfile` and `-perslice` args."""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_ascor.main(argv=['-i-SC', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                 '-i-canal', t2_canal_seg,
                                 '-vert', '1:10', '-perslice', '1',
                                 '-discfile', sct_test_path('t2', 'labels.nii.gz'), '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        rows = list(reader)
        row = rows[10]
        assert row['Slice (I->S)'] == '10'
        assert row['DistancePMJ'] == ''
        assert row['VertLevel'] == '3'
        assert float(row['aSCOR']) == pytest.approx(0.45142857142857146)


def test_sct_process_segmentation_check_discfile_perlevel(tmp_path, t2_canal_seg, t2_disc_labels):
    """Run sct_compute_ascor with `-discfile` and `-perlevel` args."""
    filename = str(tmp_path / 'tmp_file_out.csv')
    t2_sc_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    start, stop = 1, 5  # Discfile has labels `[2,3,4,5]`, so labeled seg will have regions 1-5
    sct_compute_ascor.main(argv=['-i-SC', t2_sc_seg,
                                 '-i-canal', t2_canal_seg,
                                 '-vert', f"{start}:{stop}", '-perlevel', '1',
                                 '-discfile', t2_disc_labels, '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        rows = list(reader)
        # Ensure that the VertLevel column contains all expected levels
        assert sorted([row['VertLevel'] for row in rows]) == [str(lvl) for lvl in range(start, stop + 1)]


def test_sct_process_segmentation_shape_mismatch(tmp_path, t2_canal_seg):
    """ Run sct_compute_ascor with a shape mismatch between spinal cord and canal segmentations"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    # Crop the spinal canal segmentation to create a shape mismatch
    sct_crop_image.main(argv=['-i', t2_canal_seg,
                        '-zmax', '-2', '-o', str(tmp_path / 't2_canal_seg_crop.nii.gz')])
    with pytest.raises(ValueError) as e:
        sct_compute_ascor.main(argv=['-i-SC', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                     '-i-canal', str(tmp_path / 't2_canal_seg_crop.nii.gz'),
                                     '-discfile', sct_test_path('t2', 'labels.nii.gz'), '-o', filename])
        assert e.value.code == 2


def test_sct_process_segmentation_missing_input_sc(tmp_path, t2_canal_seg):
    """ Run sct_compute_ascor missing spinal cord segmentation input"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    # Crop the spinal canal segmentation to create a shape mismatch
    with pytest.raises(SystemExit) as e:
        sct_compute_ascor.main(argv=['-i-canal', t2_canal_seg,
                                     '-discfile', sct_test_path('t2', 'labels.nii.gz'), '-o', filename])
        assert e.value.code == 2
