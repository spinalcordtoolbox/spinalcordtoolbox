import pytest
import logging

from spinalcordtoolbox.scripts import sct_compute_snr

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_snr_against_groundtruth(): 
    """Run the CLI script and check SNR against a ground truth value.""" 
    fname_out = "computed_snr.txt"
    sct_compute_snr.main(argv=['-i', 'dmri/dwi.nii.gz', '-m', 'dmri/dmri_T0001.nii.gz', '-method', 'diff', 
                               '-vol', '0,5', '-o', fname_out])
    with open(fname_out, "r") as f:
         snr = int(f.read())
    assert snr == pytest.approx(2.432321811697386)
