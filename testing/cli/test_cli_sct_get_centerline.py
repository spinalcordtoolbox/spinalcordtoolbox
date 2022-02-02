import logging
import os

import pytest

from spinalcordtoolbox.scripts import sct_get_centerline

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
@pytest.mark.parametrize('use_o_arg', [pytest.param(True, id="custom_filename"),
                                       pytest.param(False, id="default_filename")])
def test_sct_get_centerline_output_file_exists(use_o_arg, tmp_path):
    """Run the CLI script without checking results.
    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)"""
    if use_o_arg:
        output_path = os.path.join(tmp_path, 't2s_centerline.nii.gz')
        sct_get_centerline.main(argv=['-i', 't2s/t2s.nii.gz', '-c', 't2s',
                                      '-qc', str(tmp_path), '-o', output_path])
    else:
        output_path = os.path.join('t2s', 't2s_centerline.nii.gz')
        sct_get_centerline.main(argv=['-i', 't2s/t2s.nii.gz', '-c', 't2s',
                                      '-qc', str(tmp_path)])
    assert os.path.exists(output_path)
    # Note: There actually used to be a check here, but it was replaced by 'test_get_centerline_optic' in
    # 'unit_testing/test_centerline.py'. For more details, see:
    #    * https://github.com/neuropoly/spinalcordtoolbox/pull/2774/commits/5e6bd57abf6bcf825cd110e0d74b8e465d298409
    #    * https://github.com/neuropoly/spinalcordtoolbox/pull/2774#discussion_r450546434
