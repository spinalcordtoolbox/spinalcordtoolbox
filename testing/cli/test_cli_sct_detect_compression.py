# pytest unit tests for sct_detect_compression

import os
import pytest
import tempfile
import nibabel
import numpy as np
import pandas as pd

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

    df = pd.read_csv(filename)
    assert len(df) == 1  # '-num-of-slices 0' (default) --> 1 slice in total --> 1 row in the CSV file
    row = df.iloc[0]
    assert float(row['Axial slice #']) == 9
    assert float(row['Compression probability']) == pytest.approx(0.009026609893705780)
    assert row['Compression probability category'] == 'no'
    assert float(row['Compression ratio (%)']) == pytest.approx(63.00680776424760)
    assert float(row['CSA (mm2)']) == pytest.approx(78.5321685211387)
    assert float(row['Solidity (%)']) == pytest.approx(96.82139253279520)
    assert float(row['Torsion (degrees)']) == pytest.approx(1.1599432885836200)


def test_sct_detect_compression_num_of_slices(tmp_path):
    """ Run sct_detect_compression with -num-of-slices flag and check output CSV file"""
    path_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    path_labels = sct_test_path('t2', 'labels.nii.gz')  # contains only labels 3 and 5 --> okay for testing

    filename = str(tmp_path / 'compression_results.csv')
    sct_detect_compression.main(argv=['-s', path_seg,
                                      '-discfile', path_labels,
                                      '-num-of-slices', '1',  # 1 slice above and below the disc --> 3 slices in total
                                      '-o', filename])

    # Test presence of output CSV file
    assert os.path.isfile(filename)

    df = pd.read_csv(filename)
    assert len(df) == 3     # '-num-of-slices 1' --> 3 slices in total --> 3 rows in the CSV file
    assert float(df[df['Axial slice #'] == 8]['Compression probability'].values) == pytest.approx(0.011063310989869700)
    assert float(df[df['Axial slice #'] == 9]['Compression probability'].values) == pytest.approx(0.009026609893705780)
    assert float(df[df['Axial slice #'] == 10]['Compression probability'].values) == pytest.approx(0.00944641208649796)
