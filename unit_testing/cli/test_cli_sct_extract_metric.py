import csv
import numpy as np

import pytest
import logging

from spinalcordtoolbox.scripts import sct_extract_metric

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_extract_metric_weighted_average_value_against_groundtruth():
    """Verify that computed WA value is equivalent to known ground truth value."""
    fname_out = 'quantif_mtr.csv'
    sct_extract_metric.main(argv=['-i', 'mt/mtr.nii.gz', '-f', 'mt/label/atlas', '-method', 'wa', '-l', '51',
                                  '-z', '1:2', '-o', fname_out])
    with open(fname_out, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        mtr_result = np.float([row['WA()'] for row in reader][0])

    assert mtr_result == pytest.approx(32.6404)
