# -*- coding: utf-8
# Collection of functions to create data for testing

import numpy as np
from datetime import datetime
import itertools
import nibabel as nib
from skimage.transform import rotate

from spinalcordtoolbox.image import Image


def dummy_centerline(size_arr=(9, 9, 9), subsampling=1, dilate_ctl=0, hasnan=False, zeroslice=[], orientation='RPI'):
    """
    Create a dummy Image centerline of small size. Return the full and sub-sampled version along z.
    :param size_arr: tuple: (nx, ny, nz)
    :param subsampling: int >=1. Subsampling factor along z. 1: no subsampling. 2: centerline defined every other z.
    :param dilate_ctl: Dilation of centerline. E.g., if dilate_ctl=1, result will be a square of 3x3 per slice.
                         if dilate_ctl=0, result will be a single pixel per slice.
    :param hasnan: Bool: Image has non-numerical values: nan, inf. In this case, do not subsample.
    :param zeroslice: list int: zero all slices listed in this param
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
    arr_ctl = np.array([p(range(nz)).astype(np.int),
                        [round(ny / 4.)] * len(range(nz)),
                        range(nz)], dtype='uint8')
    # Loop across dilation of centerline. E.g., if dilate_ctl=1, result will be a square of 3x3 per slice.
    for ixiy_ctl in itertools.product(range(-dilate_ctl, dilate_ctl+1, 1), range(-dilate_ctl, dilate_ctl+1, 1)):
        data[(arr_ctl[0] + ixiy_ctl[0]).tolist(),
             (arr_ctl[1] + ixiy_ctl[1]).tolist(),
             arr_ctl[2].tolist()] = 1
    # Zero specified slices
    if zeroslice is not []:
        data[:, :, zeroslice] = 0

    # Create image with default orientation LPI
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
    return img, img_sub, arr_ctl


def dummy_segmentation(size_arr=(256, 256, 256), shape='ellipse', angle=15, a=50.0, b=30.0):
    """Create a dummy Image with a ellipse or ones running from top to bottom in the 3rd dimension, and rotate the image
    to make sure that compute_csa and compute_shape properly estimate the centerline angle.
    :param size_arr: tuple: (nx, ny, nz)
    :param shape: {'rectangle', 'ellipse'}
    :param angle: int: in deg
    :param a: float: 1st radius
    :param b: float: 2nd radius
    :return: fname_seg: filename of 3D binary image
    """
    nx, ny, nz = size_arr
    data = np.random.random((nx, ny, nz)) * 0.
    xx, yy = np.mgrid[:nx, :ny]
    # loop across slices and add an ellipse of axis length a and b
    # a, b = 50.0, 30.0  # radius of the ellipse (in pix size). Theoretical CSA: 4712.4
    for iz in range(nz):
        if shape == 'rectangle':  # theoretical CSA: (a*2+1)(b*2+1)
            data[:, :, iz] = ((abs(xx - nx / 2) <= a) & (abs(yy - ny / 2) <= b)) * 1
        if shape == 'ellipse':
            data[:, :, iz] = (((xx - nx / 2) / a) ** 2 + ((yy - ny / 2) / b) ** 2 <= 1) * 1
    # swap x-z axes (to make a rotation within y-z plane)
    data_swap = data.swapaxes(0, 2)
    # rotate by 15 deg, and re-grid using linear interpolation
    data_swap_rot = rotate(data_swap, angle, resize=False, center=None, order=1, mode='constant', cval=0,
                           clip=False, preserve_range=False)
    # swap back
    data_rot = data_swap_rot.swapaxes(0, 2)
    # Crop to avoid rotation edge issues
    # data_rot_crop = data_rot[..., 25:nz-25]
    # remove 5 to assess SCT stability if incomplete segmentation
    # data_rot_crop[..., data_rot_crop.shape[2]-5:] = 0
    xform = np.eye(4)
    for i in range(3):
        xform[i][i] = 0.1  # adjust voxel dimension to get realistic spinal cord size (important for some functions)
    nii = nib.nifti1.Nifti1Image(data_rot.astype('float32'), xform)
    # For debugging add .save() at the end of the command below
    img = Image(nii.get_data(), hdr=nii.header, orientation="RPI", dim=nii.header.get_data_shape(),
                absolutepath='tmp_dummy_seg_'+datetime.now().strftime("%Y%m%d%H%M%S%f")+'.nii.gz')
    return img
