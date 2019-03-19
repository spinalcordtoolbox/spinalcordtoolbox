# -*- coding: utf-8
# Collection of functions to create data for testing

import numpy as np
import itertools
import nibabel as nib

from spinalcordtoolbox.image import Image


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