# pytest unit tests for sct_straighten_spinalcord

import pytest
import logging

from spinalcordtoolbox.scripts import sct_straighten_spinalcord
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
def test_sct_straighten_spinalcord_no_checks(tmp_path_qc):
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_straighten_spinalcord.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'),
                                         '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                         '-qc', tmp_path_qc])
