#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.centerline

# TODO: check derivatives

from __future__ import absolute_import

import os
import pytest
import numpy as np

from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline, find_and_sort_coord, round_and_clip
from spinalcordtoolbox.image import Image
import sct_utils as sct

from create_test_data import dummy_centerline_small

VERBOSE = 0


# Generate a list of fake centerlines for testing different algorithms
im_centerlines = [(dummy_centerline_small(size_arr=(41, 7, 9), subsampling=1, orientation='SAL'), 2.),
                  (dummy_centerline_small(size_arr=(9, 9, 9), subsampling=3), 3.),
                  (dummy_centerline_small(size_arr=(9, 9, 9), subsampling=1, hasnan=True), 2.),
                  (dummy_centerline_small(size_arr=(30, 20, 50), subsampling=1), 3.),
                  (dummy_centerline_small(size_arr=(30, 20, 50), subsampling=5), 4.),
                  (dummy_centerline_small(size_arr=(30, 20, 50), dilate_ctl=2, subsampling=3, orientation='AIL'), 3.)]


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_polyfit(img_ctl, expected):
    """Test centerline fitting using polyfit"""
    deg = 3
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, _ = get_centerline(img_sub, algo_fitting='polyfit', param=ParamCenterline(degree=deg),
                                         minmax=False, verbose=VERBOSE)

    assert np.linalg.norm(find_and_sort_coord(img) - find_and_sort_coord(img_out)) < expected
    # check arr_out and arr_out_deriv only if input orientation is RPI (because the output array is always in RPI)
    if img.orientation == 'RPI':
        assert np.linalg.norm(find_and_sort_coord(img) - arr_out) < expected


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_bspline(img_ctl, expected):
    """Test centerline fitting using bspline"""
    deg = 3
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, _ = get_centerline(img_sub, algo_fitting='bspline', param=ParamCenterline(degree=deg),
                                         minmax=False, verbose=VERBOSE)
    assert np.linalg.norm(find_and_sort_coord(img) - find_and_sort_coord(img_out)) < expected


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_linear(img_ctl, expected):
    """Test centerline fitting using linear interpolation"""
    deg = 3
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, _ = get_centerline(img_sub, algo_fitting='linear', param=ParamCenterline(degree=deg),
                                         minmax=False, verbose=VERBOSE)
    assert np.linalg.norm(find_and_sort_coord(img) - find_and_sort_coord(img_out)) < expected


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_nurbs(img_ctl, expected):
    """Test centerline fitting using nurbs"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    # Here we need a try/except because nurbs crashes with too few points.
    try:
        img_out, arr_out, _ = get_centerline(img_sub, algo_fitting='nurbs', minmax=False, verbose=VERBOSE)
        assert np.linalg.norm(find_and_sort_coord(img) - find_and_sort_coord(img_out)) < expected
    except ArithmeticError as e:
        print(e)


# noinspection 801,PyShadowingNames
def test_get_centerline_optic():
    """Test extraction of metrics aggregation across slices: All slices by default"""
    fname_t2 = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2.nii.gz')  # install: sct_download_data -d sct_testing_data
    img_t2 = Image(fname_t2)
    # Add non-numerical values at the top corner of the image for testing purpose
    img_t2.change_type('float32')
    img_t2.data[0, 0, 0] = np.nan
    img_t2.data[1, 0, 0] = np.inf
    img_out, arr_out, _ = get_centerline(img_t2, algo_fitting='optic', param=ParamCenterline(contrast='t2'),
                                         minmax=False, verbose=VERBOSE)
    # Open ground truth segmentation and compare
    fname_t2_seg = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2_seg.nii.gz')
    img_seg_out, arr_seg_out, _ = get_centerline(Image(fname_t2_seg), algo_fitting='bspline', minmax=False,
                                                 verbose=VERBOSE)
    assert np.linalg.norm(find_and_sort_coord(img_seg_out) - find_and_sort_coord(img_out)) < 3.5


def test_round_and_clip():
    arr = round_and_clip(np.array([-0.2, 3.00001, 2.99999, 49]), clip=[0, 41])
    assert np.all(arr == np.array([0,  3,  3, 40]))  # Check element-wise equality between the two arrays
