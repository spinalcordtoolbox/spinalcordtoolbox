# pytest unit tests for sct_detect_compression

import os
import csv
import pytest
import tempfile
import nibabel
import numpy as np

from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.scripts import sct_detect_compression


def test_sct_detect_compression_check_missing_input_segmentation(tmp_path):
    """ Run sct_detect_compression when missing -s"""
    path_labels = sct_test_path('t2', 'labels.nii.gz')
    with pytest.raises(SystemExit) as e:
        sct_detect_compression.main(argv=['-discfile', path_labels])
        assert e.value.code == 2

def test_sct_detect_compression_check_missing_input_discfile(tmp_path):
    """ Run sct_detect_compression when missing -discfile"""
    path_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    with pytest.raises(SystemExit) as e:
        sct_detect_compression.main(argv=['-s', path_seg])
        assert e.value.code == 2

def test_sct_detect_compression_check_empty_input_segmentation(tmp_path):
    """ Run sct_detect_compression when -s is empty"""
    nii = nibabel.nifti1.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    path_seg = tempfile.NamedTemporaryFile(prefix='seg_empty', suffix='.nii.gz', delete=False).name
    nibabel.save(nii, path_seg)

    path_labels = sct_test_path('t2', 'labels.nii.gz')
    with pytest.raises(ValueError) as e:
        sct_detect_compression.main(argv=['-s', path_seg,
                                          '-discfile', path_labels])
        assert e.value == "Spinal cord segmentation file is empty."

def test_sct_detect_compression_check_empty_input_discfile(tmp_path):
    """ Run sct_detect_compression when -discfile is empty"""
    nii = nibabel.nifti1.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    path_labels = tempfile.NamedTemporaryFile(prefix='seg_empty', suffix='.nii.gz', delete=False).name
    nibabel.save(nii, path_labels)

    path_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    with pytest.raises(ValueError) as e:
        sct_detect_compression.main(argv=['-s', path_seg,
                                          '-discfile', path_labels])
        assert e.value == "Disc file is empty."

def test_sct_detect_compression(tmp_path):
    """ Run sct_detect_compression and check output CSV file"""
    path_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    path_labels = sct_test_path('t2', 'labels.nii.gz')  # contains only labels 3 and 5 --> okay for testing

    filename = str(tmp_path / 'compression_results.csv')
    sct_detect_compression.main(argv=['-s', path_seg,
                                      '-discfile', path_labels,
                                      '-o', filename])

    # Test presence of output CSV file
    assert os.path.isfile(filename)

    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert float(row['Axial slice #']) == 9
        assert float(row['Compression probability']) == pytest.approx(0.009026609893705780)
        assert row['Compression probability category'] == 'no'
        assert float(row['Compression ratio (%)']) == pytest.approx(63.00680776424760)
        assert float(row['CSA (mm2)']) == pytest.approx(78.5321685211387)
        assert float(row['Solidity (%)']) == pytest.approx(96.82139253279520)
        assert float(row['Torsion (degrees)']) == pytest.approx(1.1599432885836200)