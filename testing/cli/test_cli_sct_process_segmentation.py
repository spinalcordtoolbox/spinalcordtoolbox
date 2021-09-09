import pytest
import logging
import numpy as np
import tempfile
import nibabel
import csv

from spinalcordtoolbox.scripts import sct_process_segmentation

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def dummy_3d_mask_nib():
    data = np.zeros([32, 32, 32], dtype=np.uint8)
    data[9:24, 9:24, 9:24] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_3d_pmj_label():
    data = np.zeros([32, 32, 32], dtype=np.uint8)
    data[15, 15, 28] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


def test_sct_process_segmentation_check_pmj(dummy_3d_mask_nib, dummy_3d_pmj_label, tmp_path):
    """ Run sct_process_segmentation with -pmj, -pmj-distance and -pmj-extent and check the results"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_process_segmentation.main(argv=['-i', dummy_3d_mask_nib, '-pmj', dummy_3d_pmj_label,
                                        '-pmj-distance', '8', '-pmj-extent', '4', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert row['Slice (I->S)'] == '18:21'
        assert row['DistancePMJ'] == '8.0'
        assert row['VertLevel'] == ''
        assert row['SUM(length)'] == '4.0'


def test_sct_process_segmentation_missing_pmj_args(dummy_3d_mask_nib, dummy_3d_pmj_label):
    """ Run sct_process_segmentation with PMJ method when missing -pmj or -pmj-distance """
    for args in [['-i', dummy_3d_mask_nib, '-pmj', dummy_3d_pmj_label], ['-i', dummy_3d_mask_nib, '-pmj-distance', '4']]:
        with pytest.raises(SystemExit) as e:
            sct_process_segmentation.main(argv=args)
            assert e.value.code == 2


def test_sct_process_segmentation_check_normalize(dummy_3d_mask_nib, tmp_path):
    """ Run sct_process_segmentation with -normalize and check the results"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    sct_process_segmentation.main(argv=['-i', dummy_3d_mask_nib, '-normalize', 'brain-volume',
                                        '960606.0', 'sex', '0', 'thalamus-volume', '13942.0', '-o', filename])
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert float(row['MEAN(area)']) == pytest.approx(229.66183933280874)


def test_sct_process_segmentation_check_normalize_missing_value(dummy_3d_mask_nib, tmp_path):
    """ Run sct_process_segmentation with -normalize when missing a value"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_process_segmentation.main(argv=['-i', dummy_3d_mask_nib, '-normalize', 'brain-volume',
                                            '960606.0', 'sex', 'thalamus-volume', '13942.0', '-o', filename])
        assert e.value.code == 2


def test_sct_process_segmentation_check_normalize_missing_predictor(dummy_3d_mask_nib, tmp_path):
    """ Run sct_process_segmentation with -normalize when missing a predictor"""
    filename = str(tmp_path / 'tmp_file_out.csv')
    with pytest.raises(SystemExit) as e:
        sct_process_segmentation.main(argv=['-i', dummy_3d_mask_nib, '-normalize',
                                            'sex', '0' 'thalamus-volume', '13942.0', '-o', filename])
        assert e.value.code == 2



@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_process_segmentation_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_process_segmentation.main(argv=['-i', 't2/t2_seg-manual.nii.gz'])
