import logging

import pytest

from spinalcordtoolbox.scripts import sct_get_centerline

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_get_centerline_no_checks():
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    sct_get_centerline.main(argv=['-i', 't2s/t2s.nii.gz', '-c', 't2s'])
    # Note: There actually used to be a check here, but it was replaced by 'test_get_centerline_optic' in
    # 'unit_testing/test_centerline.py'. For more details, see:
    #    * https://github.com/neuropoly/spinalcordtoolbox/pull/2774/commits/5e6bd57abf6bcf825cd110e0d74b8e465d298409
    #    * https://github.com/neuropoly/spinalcordtoolbox/pull/2774#discussion_r450546434
