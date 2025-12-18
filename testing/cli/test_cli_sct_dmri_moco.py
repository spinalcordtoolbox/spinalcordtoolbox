# pytest unit tests for sct_dmri_moco

import pytest
import logging

from numpy import allclose, genfromtxt

from spinalcordtoolbox.scripts import (sct_dmri_moco, sct_image, sct_crop_image, sct_create_mask,
                                       sct_deepseg)
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def dmri_mean_seg(tmp_path_factory):
    """Mean segmented image for QC report generation."""
    tmp_path = tmp_path_factory.mktemp('dmri_mean_seg')
    path_out = str(tmp_path / 'dmri_mean_seg.nii.gz')
    sct_deepseg.main(argv=['spinalcord', '-i', sct_test_path('dmri', 'dwi_mean.nii.gz'),
                           '-o', path_out, '-qc', str(tmp_path)])
    return path_out


@pytest.mark.sct_testing
def test_sct_dmri_moco_check_params(tmp_path, tmp_path_qc, dmri_mean_seg):
    """Run the CLI script and validate output moco params."""
    sct_dmri_moco.main(argv=['-i', sct_test_path('dmri', 'dmri.nii.gz'),
                             '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                             '-g', '3', '-x', 'nn', '-r', '0',
                             '-ofolder', str(tmp_path),
                             '-qc', tmp_path_qc, '-qc-seg', dmri_mean_seg])

    lresults = genfromtxt(tmp_path / "moco_params.tsv", skip_header=1, delimiter='\t')
    lgroundtruth = [
        0.001201150922494186,
        3.276041445156287e-05,
        3.276041445156287e-05,
        3.276041445156287e-05,
        0.2046662087725081,
        0.2046662087725081,
        0.2046662087725081,
    ]
    assert allclose(lresults, lgroundtruth)


@pytest.fixture
def dmri_mask(tmp_path):
    """Mask image for testing."""
    path_out = str(tmp_path / 'mask.nii')
    sct_create_mask.main(argv=['-i', sct_test_path('dmri', 'dmri_T0000.nii.gz'),
                               '-p', 'coord,21x17', '-size', '15mm', '-f', 'gaussian',
                               '-o', path_out])

    return path_out


@pytest.mark.sct_testing
def test_sct_dmri_moco_with_mask_check_params(tmp_path, dmri_mask, tmp_path_qc, dmri_mean_seg):
    """Run the CLI script with '-m' option and validate output moco params."""
    sct_dmri_moco.main(argv=['-i', sct_test_path('dmri', 'dmri.nii.gz'),
                             '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                             '-g', '3', '-r', '0',
                             '-m', dmri_mask, '-ofolder', str(tmp_path),
                             '-qc', tmp_path_qc, '-qc-seg', dmri_mean_seg])

    lresults = genfromtxt(tmp_path / "moco_params.tsv", skip_header=1, delimiter='\t')
    lgroundtruth = [
        0.01207862,
        0.01163640,
        0.01163640,
        0.01163640,
        0.16997247,
        0.16997247,
        0.16997247,
    ]
    assert allclose(lresults, lgroundtruth)


@pytest.mark.sct_testing
def test_sct_dmri_moco_dl_no_checks(tmp_path, dmri_mask, tmp_path_qc, dmri_mean_seg):
    """Run the CLI script with '-m' and '-ref' option and using '-dl' algorithm."""
    sct_dmri_moco.main(argv=['-i', sct_test_path('dmri', 'dmri.nii.gz'),
                             '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                             '-ref', sct_test_path('dmri', 'dwi_mean.nii.gz'),
                             '-m', dmri_mask, '-ofolder', str(tmp_path), '-dl',
                             '-qc', tmp_path_qc, '-qc-seg', dmri_mean_seg])
    # NB: We skip checking params because there are no output moco params from DL-module


@pytest.fixture
def dmri_ail_cropped(tmp_path):
    """Reorient image to sagittal for testing another orientation (and crop to save time)."""
    path_out_orient = str(tmp_path / 'dmri_AIL.nii')
    path_out = str(tmp_path / 'dmri_AIL_crop.nii')
    sct_image.main(argv=['-i', sct_test_path('dmri', 'dmri.nii.gz'),
                         '-setorient', 'AIL', '-o', path_out_orient])
    sct_crop_image.main(argv=['-i', path_out_orient, '-zmin', '19', '-zmax', '21', '-o', path_out])

    return path_out


@pytest.mark.sct_testing
def test_sct_dmri_moco_sagittal_no_checks(tmp_path, tmp_path_qc, dmri_mean_seg, dmri_ail_cropped):
    """Run the CLI script, but don't check anything."""
    sct_dmri_moco.main(argv=['-i', dmri_ail_cropped, '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                             '-x', 'nn', '-r', '0',
                             '-ofolder', str(tmp_path), '-qc', tmp_path_qc, '-qc-seg', dmri_mean_seg])
    # NB: We skip checking params because there are no output moco params for sagittal images (*_AIL)


@pytest.mark.parametrize("group_size", [-1, 0, 1.5, 'NaN'])
def test_sct_dmri_moco_invalid_group_values(tmp_path, tmp_path_qc, dmri_mean_seg, group_size):
    """Ensure that invalid group sizes return a parsing error."""
    with pytest.raises(SystemExit) as e:
        sct_dmri_moco.main(argv=['-i', sct_test_path('dmri', 'dmri.nii.gz'),
                                 '-bvec', sct_test_path('dmri', 'bvecs.txt'),
                                 '-g', str(group_size), '-x', 'nn', '-r', '0',
                                 '-ofolder', str(tmp_path),
                                 '-qc', tmp_path_qc, '-qc-seg', dmri_mean_seg])
    assert e.value.code == 2
