import logging
import os

import numpy as np
import pytest

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_get_centerline

logger = logging.getLogger(__name__)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_get_centerline_output_file_exists(tmp_path):
    """This test checks the output file using default usage of the CLI script.

    TODO: Check the results. (This test replaces the 'sct_testing' test, which did not implement any checks.)
    Note: There actually used to be a check here, but it was replaced by 'test_get_centerline_optic' in
    'unit_testing/test_centerline.py'. For more details, see:
       * https://github.com/neuropoly/spinalcordtoolbox/pull/2774/commits/5e6bd57abf6bcf825cd110e0d74b8e465d298409
       * https://github.com/neuropoly/spinalcordtoolbox/pull/2774#discussion_r450546434"""
    sct_get_centerline.main(argv=['-i', 't2s/t2s.nii.gz', '-c', 't2s', '-qc', str(tmp_path)])
    for file in [os.path.join('t2s', 't2s_centerline.nii.gz'), os.path.join('t2s', 't2s_centerline.csv')]:
        assert os.path.exists(file)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
@pytest.mark.parametrize('ext', ["", ".nii.gz"])
def test_sct_get_centerline_output_file_exists_with_o_arg(tmp_path, ext):
    """This test checks the '-o' argument with and without an extension to
    ensure that the correct output file is created either way."""
    sct_get_centerline.main(argv=['-i', 't2s/t2s.nii.gz', '-c', 't2s', '-qc', str(tmp_path),
                                  '-o', os.path.join(tmp_path, 't2s_centerline'+ext)])
    for file in [os.path.join('t2s', 't2s_centerline.nii.gz'), os.path.join('t2s', 't2s_centerline.csv')]:
        assert os.path.exists(file)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_get_centerline_soft_output_file_exists(tmp_path):
    """This test checks the output soft centerline using default usage of the CLI script."""
    sct_get_centerline.main(argv=['-i', 't2/t2_seg-manual.nii.gz', '-method', 'fitseg', '-centerline-soft', '1', '-o',
                                  't2/t2_seg_centerline_soft.nii.gz', '-qc', str(tmp_path)])
    for file in [os.path.join('t2', 't2_seg_centerline_soft.nii.gz'), os.path.join('t2', 't2_seg_centerline_soft.csv')]:
        assert os.path.exists(file)


@pytest.mark.sct_testing
@pytest.mark.usefixtures("run_in_sct_testing_data_dir")
def test_sct_get_centerline_soft_output_is_equal_to_one(tmp_path):
    """This test checks that the sum of the output intensities of the soft centerline is equal to 1 on all slices. using default usage of the CLI
    script."""
    # TODO - check if it is necessary to run this command again. Maybe, we can reuse the output of the previous test.
    sct_get_centerline.main(argv=['-i', 't2/t2_seg-manual.nii.gz', '-method', 'fitseg', '-centerline-soft', '1', '-o',
                                  't2/t2_seg_centerline_soft.nii.gz', '-qc', str(tmp_path)])

    # Read nii file
    im = Image(os.path.join('t2', 't2_seg_centerline_soft.nii.gz'))
    # Sum soft centerline intensities in the (x,y) plane, across all slices
    sum_over_slices = np.apply_over_axes(np.sum, im.data, [0, 2]).flatten()
    # Test if the summation for each slice is equal to 1
    assert (sum_over_slices == 1).all()
