import numpy as np
import pytest
import logging
import skimage
import tempfile

import nibabel

from spinalcordtoolbox.scripts import sct_compute_snr

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def dummy_3dimage():
    data = np.ones([32, 32, 32])
    # Add a 5x5x5 object with representative intensity in the middle of the image
    data[14:19, 14:19, 14:19] = 100
    # Add Gaussian noise
    data = skimage.util.random_noise(data, mode='gaussian', mean=0, var=1)
    # Compute the square root of the sum of squares to obtain a Rayleigh (equivalent to Chi) distribution. This
    # distribution is a more realistic representation of noise in magnitude MRI data, which is obtained by combining
    # imaginary and real channels (each having Gaussian distribution).
    data = np.sqrt(data**2 + data**2)
    affine = np.eye(4)
    nib = nibabel.nifti1.Nifti1Image(data, affine)
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    nibabel.save(nib, filename.name)
    return filename.name


@pytest.fixture(scope="session")
# TODO: use dummy_3dimage to construct this 4d image
def dummy_4dimage():
    data = np.ones([16, 16, 16, 16])
    affine = np.eye(4)
    nib = nibabel.nifti1.Nifti1Image(data, affine)
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    nibabel.save(nib, filename.name)
    return filename.name


def test_sct_compute_snr_check_dimension(dummy_3dimage):
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_3dimage, '-m', dummy_3dimage, '-method', 'diff', '-vol', '0,5'])
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_3dimage, '-m', dummy_3dimage, '-method', 'mult', '-vol', '0,5'])

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
