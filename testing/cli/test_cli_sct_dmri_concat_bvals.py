# pytest unit tests for sct_dmri_concat_bvals

import pytest
import logging

from spinalcordtoolbox.scripts import sct_dmri_concat_bvals
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_dmri_concat_bvals_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_dmri_concat_bvals.main(argv=['-i',
                                     sct_test_path('dmri', 'bvals.txt'),
                                     sct_test_path('dmri', 'bvals.txt'),
                                     '-o', 'bvals_concat.txt'])
