#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.centerline

# TODO: test various orientations, length, pix dim
# TODO: create synthetic centerline using polynomial functions.

from __future__ import absolute_import

import os
import pytest
import tempfile
import numpy as np
import nibabel as nib

from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline
from spinalcordtoolbox.image import Image

os.chdir(tempfile.gettempdir())
print("\nOuptut folder:\n" + os.path.abspath(os.curdir) + "\n")
verbose = 2


@pytest.fixture(scope="session")
def dummy_centerline_small(size_arr=(9, 9, 9), orientation='RPI'):
    """
    Create a dummy Image centerline of small size. Return the full and sub-sampled version along z.
    """
    from numpy import poly1d, polyfit
    nx, ny, nz = size_arr
    # define polynomial-based centerline within X-Z plane, located at y=ny/4
    x = np.array([round(nx/4.), round(nx/2.), round(3*nx/4.)])
    z = np.array([0, round(nz/2.), nz-1])
    p = poly1d(polyfit(z, x, deg=3))
    data = np.zeros((nx, ny, nz))
    data[p(range(nz)).astype(np.int), round(ny / 4.), range(nz)] = 1
    # generate Image object
    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())
    img.change_orientation(orientation)
    img.data = data  # overwrite data with RPI orientation
    # define sub data
    img_sub = img.copy()
    img_sub.data = np.zeros((nx, ny, nz))
    for iz in range(0, nz, 3):
        img_sub.data[..., iz] = data[..., iz]
    return img, img_sub


# noinspection 801,PyShadowingNames
def test_get_centerline_polyfit(dummy_centerline_small):
    """Test centerline fitting using polyfit"""
    img, img_sub = dummy_centerline_small
    for deg in [3, 5]:
        # All points
        img_out, arr_out = get_centerline(img, algo_fitting='polyfit', param=ParamCenterline(degree=deg), verbose=verbose)
        assert np.linalg.norm(np.where(img.data) - arr_out) < 1
        # Sparse points
        img_out, arr_out = get_centerline(img_sub, algo_fitting='polyfit', param=ParamCenterline(degree=deg), verbose=verbose)
        assert np.linalg.norm(np.where(img.data) - arr_out) < 3.5


# noinspection 801,PyShadowingNames
def test_get_centerline_sinc(dummy_centerline_small):
    """Test centerline fitting using polyfit"""
    img, img_sub = dummy_centerline_small
    # All points
    img_out, arr_out = get_centerline(img, algo_fitting='sinc', verbose=verbose)
    assert np.linalg.norm(np.where(img.data) - arr_out) < 0.1
    # Sparse points
    img_out, arr_out = get_centerline(img_sub, algo_fitting='sinc', verbose=verbose)
    assert np.linalg.norm(np.where(img.data) - arr_out) < 3.5


# noinspection 801,PyShadowingNames
def test_get_centerline_bspline(dummy_centerline_small):
    """Test centerline fitting using polyfit"""
    img, img_sub = dummy_centerline_small
    # All points
    img_out, arr_out = get_centerline(img, algo_fitting='bspline', verbose=verbose)
    assert np.linalg.norm(np.where(img.data) - arr_out) < 0.8
    # Sparse points
    img_out, arr_out = get_centerline(img_sub, algo_fitting='bspline', param=ParamCenterline(degree=2), verbose=verbose)
    assert np.linalg.norm(np.where(img.data) - arr_out) < 2.5


# noinspection 801,PyShadowingNames
def test_get_centerline_nurbs(dummy_centerline_small):
    """Test centerline fitting using nurbs"""
    img, img_sub = dummy_centerline_small
    # All points
    img_out, arr_out = get_centerline(img, algo_fitting='nurbs', verbose=verbose)
    assert np.linalg.norm(np.where(img.data) - arr_out) < 0.8
    # Sparse points
    # TODO: fix error below and un-mask it
    # img_out, arr_out = get_centerline(img_sub, algo_fitting='nurbs')
    # assert np.linalg.norm(np.where(img.data) - arr_out) < 5.5


# noinspection 801,PyShadowingNames
def test_get_centerline_optic(dummy_centerline_small):
    """Test extraction of metrics aggregation across slices: All slices by default"""
    img, img_sub = dummy_centerline_small
    # All points
    img_out, arr_out = get_centerline(img, algo_fitting='optic', param=ParamCenterline(contrast='t2'),
                                      verbose=verbose)
    assert np.linalg.norm(np.where(img.data) - arr_out) < 10.5  # this is obviously a dummy quantitative test, given that
    #  Optic model was not trained on this synthetic data.
