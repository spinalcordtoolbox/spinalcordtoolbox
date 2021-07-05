import numpy as np

import pytest
import logging

from spinalcordtoolbox.scripts import sct_extract_metric

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_extract_metric_against_groundtruth():
    """Verify that computed values are equivalent to known ground truth values."""
    fname_out = 'quantif_mtr.csv'
    sct_extract_metric.main(argv=['-i', 'mt/mtr.nii.gz', '-f', 'mt/label/atlas', '-method', 'wa', '-l', '51',
                                  '-z', '1:2', '-o', fname_out])
    results = np.genfromtxt(fname_out, skip_header=1, delimiter=',')
    results = results[~np.isnan(results)]  # Remove non-numeric fields such as filename, SCT version, etc.
    assert results == pytest.approx([176.1532, 32.6404, 11.4573], abs=0.001)  # Size[vox], WA, and STD respectively
