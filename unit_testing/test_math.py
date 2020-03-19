#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.math


from __future__ import absolute_import
import os
import numpy as np
import datetime

import spinalcordtoolbox as sct
import spinalcordtoolbox.math
from spinalcordtoolbox.testing.create_test_data import dummy_blob


VERBOSE = int(os.getenv('SCT_VERBOSE', 0))
DUMP_IMAGES = bool(os.getenv('SCT_DEBUG_IMAGES', False))


# noinspection 801,PyShadowingNames
def test_dilate():

    # Create dummy image with single pixel in the middle
    im = dummy_blob(size_arr=(9, 9, 9), coordvox=(4, 4, 4))
    if DUMP_IMAGES:
        im.save('tmp_dummy_im_'+datetime.now().strftime("%Y%m%d%H%M%S%f")+'.nii.gz')

    # cube (only asserting along one dimension for convenience)
    data_dil = sct.math.dilate(im.data, size=1, shape='cube')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 0, 1, 0, 0]))
    data_dil = sct.math.dilate(im.data, size=2, shape='cube')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 0, 0]))
    data_dil = sct.math.dilate(im.data, size=3, shape='cube')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 1, 0]))

    # ball (only asserting along one dimension for convenience)
    data_dil = sct.math.dilate(im.data, size=0, shape='ball')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 0, 1, 0, 0]))
    data_dil = sct.math.dilate(im.data, size=1, shape='ball')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 1, 0]))
    data_dil = sct.math.dilate(im.data, size=2, shape='ball')
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([1, 1, 1, 1, 1]))

    # square in xy plane
    data_dil = sct.math.dilate(im.data, size=1, shape='disk', dim=1)
    assert np.array_equal(data_dil[2:7, 4, 4], np.array([0, 1, 1, 1, 0]))
    assert np.array_equal(data_dil[4, 4, 2:7], np.array([0, 1, 1, 1, 0]))
    data_dil = sct.math.dilate(im.data, size=1, shape='disk', dim=2)
    assert np.array_equal(data_dil[4, 2:7, 4], np.array([0, 1, 1, 1, 0]))
    assert np.array_equal(data_dil[2:7, 4, 4], np.array([0, 1, 1, 1, 0]))

    # test with Image as input
    im_dil = sct.math.dilate(im, size=1, shape='cube')
    assert np.array_equal(im_dil.data[4, 2:7, 4], np.array([0, 0, 1, 0, 0]))


# noinspection 801,PyShadowingNames
def test_erode():
    # Create dummy image with single pixel
    im = dummy_blob(size_arr=(9, 9, 9), coordvox=(4, 4, 4))
    # Dilate it
    im_dil = sct.math.dilate(im, size=1, shape='ball')
    # Erode it
    im_dil_erode = sct.math.erode(im_dil, size=1, shape='ball')
    assert np.array_equal(np.where(im_dil_erode.data), (np.array([4]), np.array([4]), np.array([4])))
    # Test with data as input
    data_dil_erode = sct.math.erode(im_dil.data, size=1, shape='ball')
    assert np.array_equal(np.where(data_dil_erode), (np.array([4]), np.array([4]), np.array([4])))
