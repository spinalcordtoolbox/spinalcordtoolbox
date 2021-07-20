import numpy as np
import pytest
import logging
import tempfile

import nibabel

from spinalcordtoolbox.scripts import sct_compute_snr
from spinalcordtoolbox.utils import SCTArgumentParser

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def dummy_3dimage():
    """
    :return: a Nifti1Image
    """
    data = np.ones([16, 16, 16])
    affine = np.eye(4)
    nib = nibabel.nifti1.Nifti1Image(data, affine)
    filename = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    nibabel.save(nib, filename.name)
    return filename.name


def test_sct_compute_snr_check_dimension(dummy_3dimage):
    with pytest.raises(ValueError):
        sct_compute_snr.main(argv=['-i', dummy_3dimage, '-m', dummy_3dimage, '-method', 'diff', '-vol', '0,5'])


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
