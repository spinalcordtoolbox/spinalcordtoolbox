# pytest unit tests for sct_compute_ascor

import pytest
import logging
import csv

from spinalcordtoolbox.scripts import sct_compute_ascor, sct_crop_image, sct_deepseg
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


def test_sct_process_segmentation_check_discfile(tmp_path, t2_canal_seg):
    """ Run sct_compute_ascor with -discfile persclice"""
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
        assert float(row['aSCOR']) == pytest.approx(0.44901418259799436)


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
