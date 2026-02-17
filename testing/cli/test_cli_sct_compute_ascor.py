# pytest unit tests for sct_compute_ascor

import pytest
import logging
import csv

from spinalcordtoolbox.image import Image
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
        assert float(row['aSCOR']) == pytest.approx(0.4502665491328947)


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


def test_sct_compute_ascor_exclude_missing_centerline_slices(tmp_path, t2_canal_seg):
    """Run sct_compute_ascor where the SC seg doesn't fully cover the slices of the canal seg."""
    filename = str(tmp_path / 'tmp_file_out.csv')
    # Crop the spinal cord segmentation with `-b 0` so that the array shape is preserved,
    # but so that the spinal cord no longer fully covers the spinal canal (NB: AIL -> SI axis==`y` -> `-ymax`)
    t2_cord_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    t2_cord_seg_crop = str(tmp_path / 't2_cord_seg_crop.nii.gz')
    sct_crop_image.main(argv=['-i', t2_cord_seg, '-ymax', '-2', '-b', '0', '-o', t2_cord_seg_crop])

    # Ensure consistent arguments between both calls
    argv = ['-i-SC', t2_cord_seg_crop,
            '-i-canal', t2_canal_seg,
            # TODO: When I left `-perslice` off, the range showed as 0:54. Is this correct? Should it be 0:53?
            '-perslice', '1',  # Used to ensure specific slices are skipped
            '-discfile', sct_test_path('t2', 'labels.nii.gz'), '-o', filename]

    # When `0`, this should raise a parser error since the SC seg (used as a centerline) doesn't cover the canal seg
    with pytest.raises(SystemExit) as e:
        sct_compute_ascor.main(argv=argv + ['-centerline-exclude-missing', '0'])
    assert e.value.code == 2

    # However, by default (`1`), this should bypass the error and limit aSCOR to only the overlapping slices
    sct_compute_ascor.main(argv=argv)
    last_slice = Image(t2_canal_seg).change_orientation('RPI').data.shape[2] - 1
    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        slices = [int(row['Slice (I->S)']) for row in list(reader)]
        assert last_slice not in slices
