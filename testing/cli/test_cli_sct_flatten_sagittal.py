# pytest unit tests for sct_flatten_sagittal

import pytest
import logging

from spinalcordtoolbox.scripts import sct_flatten_sagittal
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_flatten_sagittal_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_flatten_sagittal.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'),
                                    '-s', sct_test_path('t2', 't2_seg-manual.nii.gz')])
