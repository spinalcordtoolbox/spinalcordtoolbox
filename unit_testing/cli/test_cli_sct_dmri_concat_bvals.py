import pytest
import logging

from spinalcordtoolbox.scripts import sct_dmri_concat_bvals

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_dmri_concat_bvals_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_dmri_concat_bvals.main(argv=['-i', 'dmri/bvals.txt', 'dmri/bvals.txt', '-o', 'bvals_concat.txt'])
