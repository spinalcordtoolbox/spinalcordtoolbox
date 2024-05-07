# pytest unit tests for sct_concat_transfo

import pytest
import logging

from spinalcordtoolbox.scripts import sct_concat_transfo
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_concat_transfo_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_concat_transfo.main(argv=['-w',
                                  sct_test_path('t2', 'warp_template2anat.nii.gz'),
                                  sct_test_path('mt', 'warp_t22mt1.nii.gz'),
                                  '-d', sct_test_path('template', 'template', 'PAM50_small_t2.nii.gz')])
