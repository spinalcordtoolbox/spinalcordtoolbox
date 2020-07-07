#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.centerline


from __future__ import print_function, absolute_import

import os
import sys
import pytest
import numpy as np
import nibabel

import spinalcordtoolbox as sct
from spinalcordtoolbox import __sct_dir__
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline, find_and_sort_coord, round_and_clip
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.testing.create_test_data import dummy_centerline
import spinalcordtoolbox.math

sys.path.append(os.path.join(__sct_dir__, 'scripts'))
from sct_utils import init_sct


init_sct(log_level=2)  # Set logger in debug mode
VERBOSE = 0  # Set to 2 to save images, 0 otherwise


# Generate a list of fake centerlines: (dummy_segmentation(params), dict of expected results)
im_ctl_find_and_sort_coord = [
    (dummy_centerline(size_arr=(41, 7, 9), subsampling=1, orientation='LPI'), None),
    ]

im_ctl_zeroslice = [
    (dummy_centerline(size_arr=(15, 7, 9), zeroslice=[0, 1], orientation='LPI'), (3, 7)),
    (dummy_centerline(size_arr=(15, 7, 9), zeroslice=[], orientation='LPI'), (3, 9)),
    ]

im_centerlines = [
    (dummy_centerline(size_arr=(41, 7, 9), subsampling=1, orientation='SAL'),
     {'median': 0, 'rmse': 0.4, 'laplacian': 2},
     {}),
    (dummy_centerline(size_arr=(41, 7, 9), pixdim=(0.5, 0.5, 10), subsampling=1, orientation='SAL'),
     {'median': 0, 'rmse': 0.3, 'laplacian': 2},
     {}),
    (dummy_centerline(size_arr=(9, 9, 9), subsampling=3),
     {'median': 0, 'rmse': 0.3, 'laplacian': 0.5, 'norm': 2},
     {'exclude_polyfit': True}),  # excluding polyfit because of poorly conditioned fitting
    (dummy_centerline(size_arr=(9, 9, 9), subsampling=1, hasnan=True),
     {'median': 0, 'rmse': 0.3, 'laplacian': 2, 'norm': 1.5},
     {}),
    # (dummy_centerline(size_arr=(30, 20, 9), subsampling=1, outlier=[5]),
    #  {'median': 0, 'rmse': 1, 'laplacian': 5, 'norm': 13.5},
    #  {}),
    (dummy_centerline(size_arr=(30, 20, 50), subsampling=1),
     {'median': 0, 'rmse': 0.3, 'laplacian': 0.5, 'norm': 2.1},
     {}),
    (dummy_centerline(size_arr=(30, 20, 50), subsampling=1, outlier=[20]),
     {'median': 0, 'rmse': 0.8, 'laplacian': 70, 'norm': 14},
     {'exclude_nurbs': True}),
    (dummy_centerline(size_arr=(30, 20, 50), subsampling=3, dilate_ctl=2, orientation='AIL'),
     {'median': 0, 'rmse': 0.25, 'laplacian': 0.2},
     {}),
    (dummy_centerline(size_arr=(30, 20, 50), subsampling=5),
     {'median': 0, 'rmse': 0.3, 'laplacian': 0.5, 'norm': 3.6},
     {}),
    (dummy_centerline(size_arr=(30, 20, 50), subsampling=10),
     {'median': 0, 'rmse': 0.1, 'laplacian': 0.5, 'norm': 3.8},
     {}),
]

param_optic = [
    ({'fname_image': 'sct_testing_data/t2/t2.nii.gz',
      'contrast': 't2',
      'fname_centerline-optic': 'sct_testing_data/t2/t2_centerline-optic.nii.gz'}),
    ({'fname_image': 'sct_testing_data/t2s/t2s.nii.gz',
      'contrast': 't2s',
      'fname_centerline-optic': 'sct_testing_data/t2s/t2s_centerline-optic.nii.gz'}),
    ({'fname_image': 'sct_testing_data/dmri/dwi_mean.nii.gz',
      'contrast': 'dwi',
      'fname_centerline-optic': 'sct_testing_data/dmri/dwi_mean_centerline-optic.nii.gz'}),
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
    img_out, arr_out, _, _ = get_centerline(
        img_sub, ParamCenterline(algo_fitting='polyfit', degree=3, minmax=True), verbose=VERBOSE)
    # Assess output size
    assert arr_out.shape == expected


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected,params', im_centerlines)
def test_get_centerline_polyfit(img_ctl, expected, params):
    """Test centerline fitting using polyfit"""
    if 'exclude_polyfit':
        return
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out, fit_results = get_centerline(
        img_sub, ParamCenterline(algo_fitting='polyfit', minmax=False), verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert np.max(np.absolute(np.diff(arr_deriv_out))) < expected['laplacian']
    # check arr_out only if input orientation is RPI (because the output array is always in RPI)
    if img.orientation == 'RPI':
        assert np.linalg.norm(find_and_sort_coord(img) - arr_out) < expected['norm']


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected,params', im_centerlines)
def test_get_centerline_bspline(img_ctl, expected, params):
    """Test centerline fitting using bspline"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out, fit_results = get_centerline(
        img_sub, ParamCenterline(algo_fitting='bspline', minmax=False), verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert fit_results.rmse < expected['rmse']
    assert fit_results.laplacian_max < expected['laplacian']


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected,params', im_centerlines)
def test_get_centerline_linear(img_ctl, expected, params):
    """Test centerline fitting using linear interpolation"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out, fit_results = get_centerline(
        img_sub, ParamCenterline(algo_fitting='linear', minmax=False), verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert fit_results.laplacian_max < expected['laplacian']


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected,params', im_centerlines)
def test_get_centerline_nurbs(img_ctl, expected, params):
    """Test centerline fitting using nurbs"""
    if 'exclude_nurbs':
        return
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, arr_deriv_out, fit_results = get_centerline(
        img_sub, ParamCenterline(algo_fitting='nurbs', minmax=False), verbose=VERBOSE)
    assert np.median(find_and_sort_coord(img) - find_and_sort_coord(img_out)) == expected['median']
    assert fit_results.laplacian_max < expected['laplacian']


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('params', param_optic)
def test_get_centerline_optic(params):
    """Test centerline extraction with optic"""
    # TODO: add assert on the output .csv files for more precision
    im = Image(params['fname_image'])
    # Add non-numerical values at the top corner of the image for testing purpose
    im.change_type('float32')
    im.data[0, 0, 0] = np.nan
    im.data[1, 0, 0] = np.inf
    im_centerline, arr_out, _, _ = get_centerline(
        im, ParamCenterline(algo_fitting='optic', contrast=params['contrast'], minmax=False), verbose=VERBOSE)
    # Compare with ground truth centerline
    assert np.all(im_centerline.data == Image(params['fname_centerline-optic']).data)


def test_round_and_clip():
    arr = round_and_clip(np.array([-0.2, 3.00001, 2.99999, 49]), clip=[0, 41])
    assert np.all(arr == np.array([0,  3,  3, 40]))  # Check element-wise equality between the two arrays
