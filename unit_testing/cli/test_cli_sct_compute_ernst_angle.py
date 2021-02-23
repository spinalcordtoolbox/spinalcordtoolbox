import pytest
import logging

from spinalcordtoolbox.scripts import sct_compute_ernst_angle

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_ernst_angle_value_against_groundtruth():
    """Run the CLI script and verify that computed ernst angle is equivalent to known ground truth value."""
    fname_out = 'ernst_angle.txt'
    sct_compute_ernst_angle.main(argv=['-tr', '2000', '-t1', '850', '-o', fname_out])
    with open(fname_out, 'r') as f:
        angle_result = float(f.read())
    assert angle_result == pytest.approx(84.543553255)
