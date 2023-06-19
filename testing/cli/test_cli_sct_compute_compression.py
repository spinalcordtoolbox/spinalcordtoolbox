# pytest unit tests for sct_compute_compression

import pytest
import logging
import numpy as np
import tempfile
import nibabel
import csv

from spinalcordtoolbox.scripts import sct_compute_compression

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def dummy_3d_compression_label():
    data = np.zeros([32, 32, 81], dtype=np.uint8)
    data[15, 15, 48] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(prefix='compression', suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_3d_vert_label():
    data = np.zeros([32, 32, 81], dtype=np.uint8)
    data[15, 15, 0:10] = 2.0
    data[15, 15, 10:25] = 3.0
    data[15, 15, 25:50] = 4.0
    data[15, 15, 25:50] = 5.0
    data[15, 15, 50:70] = 6.0
    data[15, 15, 70:81] = 7.0
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(prefix='vert', suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_3d_mask_nib():
    data = np.zeros([32, 32, 81], dtype=np.uint8)
    data[9:24, 9:24, :] = 1
    data[9:24, 9:24, 48] = 0
    data[9:24, 10:22, 48] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(prefix='seg', suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_compression_value_against_groundtruth():
    """Run the CLI script and verify that computed mscc value is equivalent to known ground truth value."""
    di, da, db = 6.85, 7.65, 7.02
    mscc = sct_compute_compression.metric_ratio(mi=di, ma=da, mb=db)
    assert mscc == pytest.approx(6.612133606, abs=1e-4)


def test_sct_compute_compression_check_missing_input_segmentation(tmp_path, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression when missing -i"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-l', dummy_3d_compression_label, '-vertfile', dummy_3d_vert_label,
                                           '-o', filename, '-normalize-hc',  '1'])
        assert e.value.code == 2


def test_sct_compute_compression_check_missing_input_l(tmp_path, dummy_3d_mask_nib, dummy_3d_vert_label):
    """ Run sct_compute_mscc when missing -l"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-vertfile', dummy_3d_vert_label, '-o', filename,
                                           '-normalize-hc',  '1'])
        assert e.value.code == 2


def test_sct_compute_compression_check_missing_input_normalize_hc(tmp_path, dummy_3d_mask_nib, dummy_3d_vert_label):
    """ Run sct_compute_mscc when missing -normalize-hc"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-vertfile', dummy_3d_vert_label, '-o', filename])
        assert e.value.code == 2


def test_sct_compute_compression_check_wrong_sex(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression with wrong value for sex"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label, '-vertfile', dummy_3d_vert_label,
                                           '-sex', 'J', '-o', filename, '-normalize-hc',  '1'])
        assert e.value.code == 2


def test_sct_compute_compression_check_wrong_age(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression with wrong age"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label, '-vertfile', dummy_3d_vert_label,
                                           '-age', '20', '-o', filename, '-normalize-hc',  '1'])
        assert e.value.code == 2


def test_sct_compute_compression_check_wrong_metric(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression when specifying a wrong metric"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label, '-vertfile', dummy_3d_vert_label,
                                           '-metric', 'MEAN', '20', '-o', filename, '-normalize-hc',  '0'])
        assert e.value.code == 2


def test_sct_compute_compression_no_normalization(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression without normalization to a database of healthy controls"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label, '-vertfile', dummy_3d_vert_label,
                                       '-normalize-hc',  '0', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert float(row['compression_level']) == 5.0
        assert float(row['diameter_AP_ratio']) == pytest.approx(20.040803711692355)
        assert row['normalized_diameter_AP_ratio'] == ''


def test_sct_compute_compression(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression and check mscc and normalized mscc"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label,
                                       '-vertfile', dummy_3d_vert_label, '-normalize-hc',  '1', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert float(row['compression_level']) == 5.0
        assert float(row['diameter_AP_ratio']) == pytest.approx(12.525502319807725)
        assert float(row['normalized_diameter_AP_ratio']) == pytest.approx(16.985020560800656)


def test_sct_compute_compression_sex_F(tmp_path, dummy_3d_mask_nib, dummy_3d_compression_label, dummy_3d_vert_label):
    """ Run sct_compute_compression and check mscc and normalized mscc"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_compute_compression.main(argv=['-i', dummy_3d_mask_nib, '-l', dummy_3d_compression_label,
                                       '-vertfile', dummy_3d_vert_label, '-normalize-hc',  '1', '-sex', 'F',
                                       '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert float(row['compression_level']) == 5.0
        assert float(row['diameter_AP_ratio']) == pytest.approx(12.525502319807725)
        assert float(row['normalized_diameter_AP_ratio']) == pytest.approx(16.500154219790886)
