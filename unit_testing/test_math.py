#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.math


from __future__ import absolute_import
import sys
import os
import pytest
import numpy as np
import datetime

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))
import spinalcordtoolbox as sct

from create_test_data import dummy_blob


VERBOSE = int(os.getenv('SCT_VERBOSE', 0))
DUMP_IMAGES = bool(os.getenv('SCT_DEBUG_IMAGES', False))

# Generate a list of dummy images with single pixel in the middle
list_im = [
    # test area
    (dummy_blob(size_arr=(9, 9, 9), coordvox=(4, 4, 4))),
    ]

# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('im', list_im)
def test_morphomath(im):

    if DUMP_IMAGES:
        im.save('tmp_dummy_im_'+datetime.now().strftime("%Y%m%d%H%M%S%f")+'.nii.gz')

    # cube (only asserting along one dimension for convenience)
    data_dil = sct.math.morphomath(im.data, filter='dilation', size=1, shape='cube')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 0, 1, 0, 0]))
    data_dil = sct.math.morphomath(im.data, filter='dilation', size=2, shape='cube')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 0, 0]))
    data_dil = sct.math.morphomath(im.data, filter='dilation', size=3, shape='cube')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 1, 0]))

    # cube (only asserting along one dimension for convenience)
    data_dil = sct.math.morphomath(im.data, filter='dilation', size=0, shape='ball')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 0, 1, 0, 0]))
    data_dil = sct.math.morphomath(im.data, filter='dilation', size=1, shape='ball')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 1, 0]))
    data_dil = sct.math.morphomath(im.data, filter='dilation', size=2, shape='ball')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([1, 1, 1, 1, 1]))

    # square in xy plane
    data_dil = sct.math.morphomath(im.data, filter='dilation', size=1, shape='disk', dim=1)
    assert np.array_equal(data_dil[2:7, 4, 4], np.array([0, 1, 1, 1, 0]))
    assert np.array_equal(data_dil[4, 4, 2:7], np.array([0, 1, 1, 1, 0]))
    data_dil = sct.math.morphomath(im.data, filter='dilation', size=1, shape='disk', dim=2)
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 1, 0]))
    assert np.array_equal(data_dil[2:7, 4, 4], np.array([0, 1, 1, 1, 0]))
