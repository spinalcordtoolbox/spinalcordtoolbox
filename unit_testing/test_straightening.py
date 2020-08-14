#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.straightening


from __future__ import absolute_import

import os, sys

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))
from spinalcordtoolbox.straightening import SpinalCordStraightener
import sct_utils as sct
from spinalcordtoolbox.utils import sct_test_path

VERBOSE = 0  # Set to 2 to save images, 0 otherwise


# noinspection 801,PyShadowingNames
def test_straighten():
    """Test straightening with default params"""
    fname_t2 = sct_test_path('t2', 't2.nii.gz')  # sct_download_data -d sct_testing_data
    fname_t2_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')
    sc_straight = SpinalCordStraightener(fname_t2, fname_t2_seg)
    sc_straight.accuracy_results = True
    sc_straight.straighten()
    assert sc_straight.mse_straightening < 0.8
    assert sc_straight.max_distance_straightening < 1.2
