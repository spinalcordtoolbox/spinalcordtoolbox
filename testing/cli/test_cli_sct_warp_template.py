import pytest
import logging

from spinalcordtoolbox.scripts import sct_warp_template

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_warp_template_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_warp_template.main(argv=['-d', 'mt/mt1.nii.gz', '-w', 'mt/warp_template2mt.nii.gz', '-a', '0',
                                 '-t', 'template', '-qc', 'testing-qc'])
