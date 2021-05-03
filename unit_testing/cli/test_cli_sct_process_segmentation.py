import pytest
import logging

from spinalcordtoolbox.scripts import sct_process_segmentation

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_process_segmentation_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_process_segmentation.main(argv=['-i', 't2/t2_seg-manual.nii.gz'])
