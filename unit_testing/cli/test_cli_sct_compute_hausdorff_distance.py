import pytest
import logging

from spinalcordtoolbox.scripts import sct_compute_hausdorff_distance

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_compute_hausdorff_distance_null_values():
    """Run the CLI script and verify computed distances between identical images are all zero."""
    # TODO: Test distances between non-identical images`
    sct_compute_hausdorff_distance.main(argv=['-i', 't2s/t2s_gmseg_manual.nii.gz', '-d', 't2s/t2s_gmseg_manual.nii.gz'])

    with open('hausdorff_distance.txt', 'r') as f:
        hausdorff_distance_lst = []
        for line in f.readlines():
            if line.startswith('Slice'):
                hausdorff_distance_lst.append(float(line.split(': ')[1].split(' -')[0]))

    assert max(hausdorff_distance_lst) <= 1.0
