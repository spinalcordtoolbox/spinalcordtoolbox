# pytest unit tests for sct_fmri_moco

import pytest
import logging

from spinalcordtoolbox.scripts import sct_fmri_moco
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_fmri_moco_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_fmri_moco.main(argv=['-i', sct_test_path('fmri', 'fmri_r.nii.gz'), '-g', '4', '-x', 'nn', '-r', '0'])
