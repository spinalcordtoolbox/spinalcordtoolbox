import numpy as np
import pytest
import logging
import skimage
import tempfile

import nibabel

from spinalcordtoolbox.scripts import sct_compute_snr

logger = logging.getLogger(__name__)

# Declare the signal intensity in the object created in the dummy_*() functions. This object will be used to compute
# the SNR using various methods.
SIGNAL_OBJECT = 1000


def dummy_3d_data():
    """Create 3d image with object in the middle and Rayleigh noise distribution. Outputs a nibabel object."""
    data = np.ones([32, 32, 32], dtype=np.float)
    # Add an object with representative intensity in the middle of the image
    data[9:24, 9:24, 9:24] = SIGNAL_OBJECT
    # Add Gaussian noise with unit variance on two separate images
    data1 = skimage.util.random_noise(data, mode='gaussian', clip=False, mean=0, var=1)
    data2 = skimage.util.random_noise(data, mode='gaussian', clip=False, mean=0, var=1)
    # Compute the square root of the sum of squares to obtain a Rayleigh (equivalent to Chi) distribution. This
    # distribution is a more realistic representation of noise in magnitude MRI data, which is obtained by combining
    # imaginary and real channels (each having Gaussian distribution).
    data = np.sqrt(data1**2 + data2**2)
    return data


@pytest.fixture(scope="session")
def dummy_3d_nib():
    nii = nibabel.nifti1.Nifti1Image(dummy_3d_data(), np.eye(4))
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_4d_nib():
    # Create 4D volume. We need sufficient volumes to compute reliable standard deviation along the 4th dimension.
    data = np.stack([dummy_3d_data() for i in range(50)], axis=3)
    affine = np.eye(4)
    nib = nibabel.nifti1.Nifti1Image(data, affine)
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    nibabel.save(nib, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_3d_mask_nib():
    data = np.zeros([32, 32, 32], dtype=np.uint8)
    data[9:24, 9:24, 9:24] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.fixture(scope="session")
def dummy_3d_mask_noise_nib():
    data = np.zeros([32, 32, 32], dtype=np.uint8)
    data[0:5, 0:5, 0:32] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


@pytest.mark.parametrize('method', ['diff', 'mult'])
def test_sct_compute_snr_check_dimension(dummy_3d_nib, method):
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_3d_nib, '-m', dummy_3d_nib, '-method', method, '-vol', '0,5'])


def test_sct_compute_snr_check_dimension_mask(dummy_4d_nib):
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_4d_nib, '-m', dummy_4d_nib, '-method', 'mult'])


def test_sct_compute_snr_check_dimension_mask_noise(dummy_3d_nib, dummy_4d_nib):
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_3d_nib, '-m', dummy_3d_nib, '-m-noise', dummy_4d_nib,
                                   '-method', 'single'])


def test_sct_compute_snr_check_vol_param(dummy_4d_nib, dummy_3d_nib):
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_4d_nib, '-m', dummy_3d_nib, '-m-noise', dummy_3d_nib,
                                   '-vol', '0,1,2', '-method', 'single'])


def test_sct_compute_snr_missing_mask(dummy_4d_nib, dummy_3d_nib):
    with pytest.raises(RuntimeError):
        sct_compute_snr.main(argv=['-i', dummy_4d_nib, '-m', dummy_3d_nib, '-method', 'single'])


def test_sct_compute_snr_mult(dummy_4d_nib, dummy_3d_mask_nib):
    filename = tempfile.NamedTemporaryFile(prefix='snr_mult_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_4d_nib, '-m', dummy_3d_mask_nib, '-method', 'mult', '-o', filename])
    with open(filename, "r") as f:
        snr = float(f.read())
    # We need a large tolerance because of the randomization
    assert snr == pytest.approx(np.sqrt(2*SIGNAL_OBJECT**2), rel=0.05)


def test_sct_compute_snr_mult_vol(dummy_4d_nib, dummy_3d_mask_nib):
    filename = tempfile.NamedTemporaryFile(prefix='snr_mult_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_4d_nib, '-m', dummy_3d_mask_nib, '-method', 'mult', '-vol', '0:40', '-o', filename])
    with open(filename, "r") as f:
        snr = float(f.read())
    # We need a large tolerance because of the randomization
    assert snr == pytest.approx(np.sqrt(2*SIGNAL_OBJECT**2), rel=0.05)


def test_sct_compute_snr_diff(dummy_4d_nib, dummy_3d_mask_nib):
    filename = tempfile.NamedTemporaryFile(prefix='snr_diff_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_4d_nib, '-m', dummy_3d_mask_nib, '-method', 'diff', '-vol', '0,1', '-o', filename])
    with open(filename, "r") as f:
        snr = float(f.read())
    # We need a large tolerance because of the randomization
    assert snr == pytest.approx(np.sqrt(2*SIGNAL_OBJECT**2), rel=0.05)


def test_sct_compute_snr_diff_vol_versus_not_vol(dummy_4d_nib, dummy_3d_mask_nib):
    """Make sure that if vol is not specified, it uses the first 2 volumes"""
    filename = tempfile.NamedTemporaryFile(prefix='snr_diff_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_4d_nib, '-m', dummy_3d_mask_nib, '-method', 'diff', '-vol', '0,1', '-o', filename])
    filename_no_vol = tempfile.NamedTemporaryFile(prefix='snr_diff_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_4d_nib, '-m', dummy_3d_mask_nib, '-method', 'diff', '-o', filename_no_vol])
    with open(filename, "r") as f:
        snr = float(f.read())
    with open(filename_no_vol, "r") as f:
        snr_no_vol = float(f.read())
    # We need a large tolerance because of the randomization
    assert snr == snr_no_vol


def test_sct_compute_snr_single_3d(dummy_3d_nib, dummy_3d_mask_nib, dummy_3d_mask_noise_nib):
    filename = tempfile.NamedTemporaryFile(prefix='snr_single_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_3d_nib, '-m', dummy_3d_mask_nib, '-m-noise', dummy_3d_mask_noise_nib, '-method', 'single',
              '-o', filename])
    with open(filename, "r") as f:
        snr = float(f.read())
    # We need a large tolerance because of the randomization
    # TODO: Need to figure out what the problem is with the strong bias (~30% less than the "real" SNR)
    assert snr == pytest.approx(np.sqrt(2*SIGNAL_OBJECT**2), rel=0.5)


def test_sct_compute_snr_single_4d(dummy_4d_nib, dummy_3d_mask_nib, dummy_3d_mask_noise_nib):
    filename = tempfile.NamedTemporaryFile(prefix='snr_single_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_4d_nib, '-m', dummy_3d_mask_nib, '-m-noise', dummy_3d_mask_noise_nib, '-method', 'single',
              '-vol', '0', '-o', filename])
    with open(filename, "r") as f:
        snr = float(f.read())
    # We need a large tolerance because of the randomization
    # TODO: Need to figure out what the problem is with the strong bias (~30% less than the "real" SNR)
    assert snr == pytest.approx(np.sqrt(2*SIGNAL_OBJECT**2), rel=0.5)


def test_sct_compute_snr_single_3d_mask_object(dummy_3d_nib, dummy_3d_mask_nib):
    """Compute noise statistics in the object ROI. In this case, noise distribution is considered Gaussian,
    not Rayleigh"""
    filename = tempfile.NamedTemporaryFile(prefix='snr_single_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_3d_nib, '-m', dummy_3d_mask_nib, '-m-noise', dummy_3d_mask_nib, '-method', 'single',
              '-rayleigh', 0, '-o', filename])
    with open(filename, "r") as f:
        snr = float(f.read())
    # We need a large tolerance because of the randomization
    assert snr == pytest.approx(np.sqrt(2*SIGNAL_OBJECT**2), rel=0.05)
