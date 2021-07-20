import numpy as np
import pytest
import logging
import skimage
import tempfile

import nibabel

from spinalcordtoolbox.scripts import sct_compute_snr

logger = logging.getLogger(__name__)

SIGNAL_OBJECT = 100

def dummy_3d_data():
    """Create 3d image with object in the middle and Rayleigh noise distribution. Outputs a nibabel object."""
    data = np.ones([32, 32, 32], dtype=np.float)
    # Add a 5x5x5 object with representative intensity in the middle of the image
    data[14:19, 14:19, 14:19] = SIGNAL_OBJECT
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
    data[14:19, 14:19, 14:19] = 1
    nii = nibabel.nifti1.Nifti1Image(data, np.eye(4))
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    nibabel.save(nii, filename)
    return filename


def test_sct_compute_snr_check_dimension(dummy_3d_nib):
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_3d_nib, '-m', dummy_3d_nib, '-method', 'diff', '-vol', '0,5'])
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_3d_nib, '-m', dummy_3d_nib, '-method', 'mult', '-vol', '0,5'])


def test_sct_compute_snr_check_dimension_mask(dummy_4d_nib):
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_4d_nib, '-m', dummy_4d_nib, '-method', 'mult'])


def test_sct_compute_snr_mult(dummy_4d_nib, dummy_3d_mask_nib):
    filename = tempfile.NamedTemporaryFile(prefix='snr_mult_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_4d_nib, '-m', dummy_3d_mask_nib, '-method', 'mult', '-o', filename])
    with open(filename, "r") as f:
        snr = float(f.read())
    # We need a large tolerance because of the randomization
    assert snr == pytest.approx(np.sqrt(2*SIGNAL_OBJECT**2), abs=5)


def test_sct_compute_snr_mult_vol(dummy_4d_nib, dummy_3d_mask_nib):
    filename = tempfile.NamedTemporaryFile(prefix='snr_mult_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_4d_nib, '-m', dummy_3d_mask_nib, '-method', 'mult', '-vol', '0:40', '-o', filename])
    with open(filename, "r") as f:
        snr = float(f.read())
    # We need a large tolerance because of the randomization
    assert snr == pytest.approx(np.sqrt(2*SIGNAL_OBJECT**2), abs=5)


def test_sct_compute_snr_diff(dummy_4d_nib, dummy_3d_mask_nib):
    filename = tempfile.NamedTemporaryFile(prefix='snr_diff_', suffix='.txt', delete=False).name
    sct_compute_snr.main(
        argv=['-i', dummy_4d_nib, '-m', dummy_3d_mask_nib, '-method', 'diff', '-vol', '0,1', '-o', filename])
    with open(filename, "r") as f:
        snr = float(f.read())
    # We need a large tolerance because of the randomization
    assert snr == pytest.approx(np.sqrt(2*SIGNAL_OBJECT**2), abs=5)

#
# @pytest.mark.sct_testing
# @pytest.mark.usefixtures("run_in_sct_testing_data_dir")
# def test_sct_compute_snr_against_groundtruth():
#     """Run the CLI script and check SNR against a ground truth value."""
#     fname_out = "computed_snr.txt"
#     sct_compute_snr.main(argv=['-i', 'dmri/dwi.nii.gz', '-m', 'dmri/dmri_T0001.nii.gz', '-method', 'diff',
#                                '-vol', '0,5', '-o', fname_out])
#     with open(fname_out, "r") as f:
#         snr = float(f.read())
#     assert snr == pytest.approx(2.432321811697386)
