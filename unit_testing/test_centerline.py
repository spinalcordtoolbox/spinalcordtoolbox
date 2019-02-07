#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.centerline

# TODO: check derivatives

from __future__ import absolute_import

import os
import pytest
import itertools
import numpy as np
import nibabel as nib

from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline, find_and_sort_coord, round_and_clip
from spinalcordtoolbox.image import Image
import sct_utils as sct

VERBOSE = 0


@pytest.fixture(scope="session")
def dummy_centerline_small(size_arr=(9, 9, 9), subsampling=1, dilate_ctl=0, hasnan=False, orientation='RPI'):
    """
    Create a dummy Image centerline of small size. Return the full and sub-sampled version along z.
    :param size_arr: tuple: (nx, ny, nz)
    :param subsampling: int >=1. Subsampling factor along z. 1: no subsampling. 2: centerline defined every other z.
    :param dilate_ctl: Dilation of centerline. E.g., if dilate_ctl=1, result will be a square of 3x3 per slice.
                         if dilate_ctl=0, result will be a single pixel per slice.
    :param hasnan: Bool: Image has non-numerical values: nan, inf. In this case, do not subsample.
    :param orientation:
    :return:
    """
    from numpy import poly1d, polyfit
    nx, ny, nz = size_arr
    # define polynomial-based centerline within X-Z plane, located at y=ny/4
    x = np.array([round(nx/4.), round(nx/2.), round(3*nx/4.)])
    z = np.array([0, round(nz/2.), nz-1])
    p = poly1d(polyfit(z, x, deg=3))
    data = np.zeros((nx, ny, nz))
    # Loop across dilation of centerline. E.g., if dilate_ctl=1, result will be a square of 3x3 per slice.
    for ixiy_ctl in itertools.product(range(-dilate_ctl, dilate_ctl+1, 1), range(-dilate_ctl, dilate_ctl+1, 1)):
        data[p(range(nz)).astype(np.int) + ixiy_ctl[0], round(ny / 4.) + ixiy_ctl[1], range(nz)] = 1
    # generate Image object with RPI orientation
    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())
    # subsample data
    img_sub = img.copy()
    img_sub.data = np.zeros((nx, ny, nz))
    for iz in range(0, nz, subsampling):
        img_sub.data[..., iz] = data[..., iz]
    # Add non-numerical values at the top corner of the image
    if hasnan:
        img.data[0, 0, 0] = np.nan
        img.data[1, 0, 0] = np.inf
    # Update orientation
    img.change_orientation(orientation)
    img_sub.change_orientation(orientation)
    return img, img_sub


# Generate a list of fake centerlines for testing different algorithms
im_centerlines = [(dummy_centerline_small(size_arr=(41, 7, 9), subsampling=1, orientation='SAL'), 2.),
                  (dummy_centerline_small(size_arr=(9, 9, 9), subsampling=3), 3.),
                  (dummy_centerline_small(size_arr=(9, 9, 9), subsampling=1, hasnan=True), 2.),
                  (dummy_centerline_small(size_arr=(30, 20, 50), subsampling=1), 3.),
                  (dummy_centerline_small(size_arr=(30, 20, 50), subsampling=5), 3.),
                  (dummy_centerline_small(size_arr=(30, 20, 50), dilate_ctl=2, subsampling=3, orientation='AIL'), 3.)]


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_polyfit(img_ctl, expected):
    """Test centerline fitting using polyfit"""
    deg = 3
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    img_out, arr_out, _ = get_centerline(img_sub, algo_fitting='polyfit', param=ParamCenterline(degree=deg),
                                         verbose=VERBOSE)

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
                                         verbose=VERBOSE)
    assert np.linalg.norm(find_and_sort_coord(img) - find_and_sort_coord(img_out)) < expected


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_nurbs(img_ctl, expected):
    """Test centerline fitting using nurbs"""
    img, img_sub = [img_ctl[0].copy(), img_ctl[1].copy()]
    # Here we need a try/except because nurbs crashes with too few points.
    try:
        img_out, arr_out, _ = get_centerline(img_sub, algo_fitting='nurbs', verbose=VERBOSE)
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
                                         verbose=VERBOSE)
    # Open ground truth segmentation and compare
    fname_t2_seg = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2_seg.nii.gz')
    img_seg_out, arr_seg_out, _ = get_centerline(Image(fname_t2_seg), algo_fitting='bspline', verbose=VERBOSE)
    assert np.linalg.norm(find_and_sort_coord(img_seg_out) - find_and_sort_coord(img_out)) < 3.5


def test_round_and_clip():
    arr = round_and_clip(np.array([-0.2, 3.00001, 2.99999, 49]), clip=[0, 41])
    assert np.all(arr == np.array([0,  3,  3, 41]))  # Check element-wise equality between the two arrays
