#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.straightening

from spinalcordtoolbox.straightening import SpinalCordStraightener
from spinalcordtoolbox.utils import sct_test_path

VERBOSE = 0  # Set to 2 to save images, 0 otherwise


def test_straighten(tmp_path):
    """Test straightening with default params"""
    fname_t2 = sct_test_path('t2', 't2.nii.gz')  # sct_download_data -d sct_testing_data
    fname_t2_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    sc_straight = SpinalCordStraightener(fname_t2, fname_t2_seg, path_output=str(tmp_path))
    sc_straight.accuracy_results = True
    sc_straight.straighten()
    assert sc_straight.mse_straightening < 0.8
    assert sc_straight.max_distance_straightening < 1.2
