import numpy as np
import pytest
import logging
import tempfile

import nibabel

from spinalcordtoolbox.scripts import sct_compute_snr

logger = logging.getLogger(__name__)


# @pytest.fixture(scope="session")
def dummy_3dimage():
    data = np.ones([16, 16, 16])
    affine = np.eye(4)
    nib = nibabel.nifti1.Nifti1Image(data, affine)
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    nibabel.save(nib, filename.name)
    return filename.name


def dummy_4dimage():
    data = np.ones([16, 16, 16, 16])
    affine = np.eye(4)
    nib = nibabel.nifti1.Nifti1Image(data, affine)
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    nibabel.save(nib, filename.name)
    return filename.name


def dummy_5dimage():
    data = np.ones([16, 16, 16, 16, 3])
    affine = np.eye(4)
    nib = nibabel.nifti1.Nifti1Image(data, affine)
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    nibabel.save(nib, filename.name)
    return filename.name


@pytest.mark.parametrize('dummy_image', [dummy_3dimage(), dummy_5dimage()])
def test_sct_compute_snr_check_dimension(dummy_image):
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_image, '-m', dummy_3dimage(), '-method', 'diff', '-vol', '0,5'])
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_image, '-m', dummy_3dimage(), '-method', 'mult', '-vol', '0,5'])


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_snr_against_groundtruth(): 
    """Run the CLI script and check SNR against a ground truth value.""" 
    fname_out = "computed_snr.txt"
    sct_compute_snr.main(argv=['-i', 'dmri/dwi.nii.gz', '-m', 'dmri/dmri_T0001.nii.gz', '-method', 'diff', 
                               '-vol', '0,5', '-o', fname_out])
    with open(fname_out, "r") as f:
        snr = float(f.read())
    assert snr == pytest.approx(2.432321811697386)
