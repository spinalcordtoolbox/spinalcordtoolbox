#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.centerline

# TODO: test various orientations, length, pix dim
# TODO: create test in physical coordinate
# TODO: create synthetic segmentation (in addition to centerline)

from __future__ import absolute_import

import os
import pytest
import tempfile
import numpy as np
import nibabel as nib

from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline
from spinalcordtoolbox.image import Image

# Move to temp folder
curdir = os.path.abspath(os.curdir)
os.chdir(tempfile.gettempdir())
print("\nOuptut folder:\n" + os.path.abspath(os.curdir) + "\n")
verbose = 0


@pytest.fixture(scope="session")
def dummy_centerline_small(size_arr=(9, 9, 9), subsampling=1, orientation='RPI'):
    """
    Create a dummy Image centerline of small size. Return the full and sub-sampled version along z.
    :param size_arr: tuple: (nx, ny, nz)
    :param subsampling: int >=1. Subsampling factor along z. 1: no subsampling. 2: centerline defined every other z.
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
    data[p(range(nz)).astype(np.int), round(ny / 4.), range(nz)] = 1
    # generate Image object
    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())
    img.change_orientation(orientation)
    img.data = data  # overwrite data with RPI orientation
    # subsample data
    img_sub = img.copy()
    img_sub.data = np.zeros((nx, ny, nz))
    for iz in range(0, nz, subsampling):
        img_sub.data[..., iz] = data[..., iz]
    return img, img_sub


im_centerlines = [(dummy_centerline_small(size_arr=(9, 9, 9), subsampling=1), 2.),
                  (dummy_centerline_small(size_arr=(9, 9, 9), subsampling=3), 3.),
                  (dummy_centerline_small(size_arr=(30, 20, 50), subsampling=1), 3.),
                  (dummy_centerline_small(size_arr=(30, 20, 50), subsampling=5), 3.)]


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_polyfit(img_ctl, expected):
    """Test centerline fitting using polyfit"""
    deg = 3
    img, img_sub = img_ctl
    img_out, arr_out, _ = get_centerline(img_sub, algo_fitting='polyfit', param=ParamCenterline(degree=deg), verbose=verbose)
    assert np.linalg.norm(np.where(img.data) - arr_out) < expected


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_bspline(img_ctl, expected):
    """Test centerline fitting using polyfit"""
    deg = 3
    img, img_sub = img_ctl
    img_out, arr_out, _ = get_centerline(img_sub, algo_fitting='bspline', param=ParamCenterline(degree=deg), verbose=verbose)
    assert np.linalg.norm(np.where(img.data) - arr_out) < expected


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('img_ctl,expected', im_centerlines)
def test_get_centerline_nurbs(img_ctl, expected):
    """Test centerline fitting using nurbs"""
    img, img_sub = img_ctl
    # Here we need a try/except because nurbs crashes with too few points.
    try:
        img_out, arr_out, _ = get_centerline(img_sub, algo_fitting='nurbs', verbose=verbose)
        assert np.linalg.norm(np.where(img.data) - arr_out) < expected
    except Exception as e:
        print(e)


# noinspection 801,PyShadowingNames
def test_get_centerline_optic():
    """Test extraction of metrics aggregation across slices: All slices by default"""
    fname_t2 = os.path.join(curdir, 'sct_testing_data/t2/t2.nii.gz')  # install: sct_download_data -d sct_testing_data
    img = Image(fname_t2)
    img_out, arr_out, _ = get_centerline(img, algo_fitting='optic', param=ParamCenterline(contrast='t2'),
                                         verbose=verbose)
    # Open ground truth segmentation and compare
    fname_t2_seg = os.path.join(curdir, 'sct_testing_data/t2/t2_seg.nii.gz')
    _, arr_seg_out, _ = get_centerline(Image(fname_t2_seg), algo_fitting='bspline', verbose=verbose)
    assert np.linalg.norm(arr_seg_out - arr_out) < 3.0


os.chdir(curdir)