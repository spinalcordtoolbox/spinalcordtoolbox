import pytest
import logging

from spinalcordtoolbox.scripts import sct_compute_hausdorff_distance

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_hausdorff_distance_values_against_threshold():
    """Run the CLI script and verify computed distances are all within a given threshold.
    TODO: Distances are all 0.0. Is this test checking anything useful?"""
    sct_compute_hausdorff_distance.main(argv=['-i', 't2s/t2s_gmseg_manual.nii.gz', '-d', 't2s/t2s_gmseg_manual.nii.gz'])

    with open('hausdorff_distance.txt', 'r') as f:
        hausdorff_distance_lst = []
        for line in f.readlines():
            if line.startswith('Slice'):
                hausdorff_distance_lst.append(float(line.split(': ')[1].split(' -')[0]))

    assert max(hausdorff_distance_lst) <= 1.0
