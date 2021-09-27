import pytest
import logging

from spinalcordtoolbox.scripts import sct_concat_transfo

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_concat_transfo_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_concat_transfo.main(argv=['-w', 't2/warp_template2anat.nii.gz', 'mt/warp_t22mt1.nii.gz',
                                  '-d', 'template/template/PAM50_small_t2.nii.gz'])
