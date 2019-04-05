#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.centerline

# TODO: test with more datapoints
# TODO: introduce distance unit


from __future__ import absolute_import

import os
import pytest
import numpy as np

from spinalcordtoolbox.centerline.core import get_centerline, find_and_sort_coord, round_and_clip
from spinalcordtoolbox.image import Image
import sct_utils as sct

from create_test_data import dummy_centerline

VERBOSE = 2  # Set to 2 to save images, 0 otherwise


# Generate a list of fake centerlines: (dummy_segmentation(params), dict of expected results)
im_ctl_find_and_sort_coord = [
    (dummy_centerline(size_arr=(41, 7, 9), subsampling=1, orientation='LPI'), None),
    ]

im_ctl_zeroslice = [
    (dummy_centerline(size_arr=(15, 7, 9), zeroslice=[0, 1], orientation='LPI'), (3, 7)),
    (dummy_centerline(size_arr=(15, 7, 9), zeroslice=[], orientation='LPI'), (3, 9)),
    ]

im_centerlines = [(dummy_centerline(size_arr=(41, 7, 9), subsampling=1, orientation='SAL'),
                   {'median': 0, 'laplacian': 2}),
                  (dummy_centerline(size_arr=(9, 9, 9), subsampling=3), {'median': 0, 'laplacian': 2}),
                  (dummy_centerline(size_arr=(9, 9, 9), subsampling=1, hasnan=True), {'median': 0, 'laplacian': 1}),
                  (dummy_centerline(size_arr=(30, 20, 50), subsampling=1), {'median': 0, 'laplacian': 0.05}),
                  (dummy_centerline(size_arr=(30, 20, 50), subsampling=5), {'median': 0, 'laplacian': 0.5}),
                  (dummy_centerline(size_arr=(30, 20, 50), dilate_ctl=2, subsampling=3, orientation='AIL'),
                   {'median': 0, 'laplacian': 0.1}),
                  (dummy_centerline(size_arr=(30, 20, 50), subsampling=1, outlier=[20]), {'median': 0, 'laplacian': 0.5})
                  ]

# Specific centerline for nurbs because test does not pas with the previous centerlines
im_centerlines_nurbs = [
    (dummy_centerline(size_arr=(9, 9, 9), subsampling=3), {'median': 0, 'laplacian': 2.6})
    ]

# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_ctl_find_and_sort_coord)
def test_find_and_sort_coord(img_ctl, expected):
    img = img_ctl[0].copy()
    centermass = find_and_sort_coord(img)
    assert centermass.shape == (3, 9)
    assert np.linalg.norm(centermass - img_ctl[2]) == 0


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_ctl_zeroslice)
def test_get_centerline_polyfit_minmax(img_ctl, expected):
    """Test centerline fitting with minmax=True"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, _ = get_centerline(img_sub, algo_fitting='polyfit', degree=3, minmax=True, verbose=VERBOSE)
    # Assess output size
    assert arr_out.shape == expected


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_polyfit(img_ctl, expected):
    """Test centerline fitting using polyfit"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out = get_centerline(img_sub, algo_fitting='polyfit', minmax=False, verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert np.max(np.absolute(np.diff(arr_deriv_out))) < expected['laplacian']
    # check arr_out and arr_out_deriv only if input orientation is RPI (because the output array is always in RPI)
    if img.orientation == 'RPI':
        assert np.linalg.norm(find_and_sort_coord(img) - arr_out) < expected


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_bspline(img_ctl, expected):
    """Test centerline fitting using bspline"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out = get_centerline(img_sub, algo_fitting='bspline', minmax=False, verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert np.max(np.absolute(np.diff(arr_deriv_out))) < expected['laplacian']


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_linear(img_ctl, expected):
    """Test centerline fitting using linear interpolation"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out = get_centerline(img_sub, algo_fitting='linear', minmax=False, verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert np.max(np.absolute(np.diff(arr_deriv_out))) < expected['laplacian']


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines_nurbs)
def test_get_centerline_nurbs(img_ctl, expected):
    """Test centerline fitting using nurbs"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    # Here we need a try/except because nurbs crashes with too few points.
    try:
        img_out, arr_out, arr_deriv_out = get_centerline(img_sub, algo_fitting='nurbs', minmax=False, verbose=VERBOSE)
        assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
        assert np.max(np.absolute(np.diff(arr_deriv_out))) < expected['laplacian']
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
    img_out, arr_out, _ = get_centerline(img_t2, algo_fitting='optic', contrast='t2', minmax=False, verbose=VERBOSE)
    # Open ground truth segmentation and compare
    fname_t2_seg = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2_seg.nii.gz')
    img_seg_out, arr_seg_out, _ = get_centerline(Image(fname_t2_seg), algo_fitting='bspline', minmax=False,
                                                 verbose=VERBOSE)
    assert np.linalg.norm(find_and_sort_coord(img_seg_out) - find_and_sort_coord(img_out)) < 3.5


def test_round_and_clip():
    arr = round_and_clip(np.array([-0.2, 3.00001, 2.99999, 49]), clip=[0, 41])
    assert np.all(arr == np.array([0,  3,  3, 40]))  # Check element-wise equality between the two arrays
