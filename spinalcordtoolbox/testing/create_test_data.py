# -*- coding: utf-8
# Collection of functions to create data for testing

import numpy as np
import numpy.matlib
from numpy.polynomial import Polynomial as P
from datetime import datetime
import itertools
from skimage.transform import rotate

from random import uniform
import copy

import nibabel as nib

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.centerline.curve_fitting import bspline, polyfit_1d
from sct_image import concat_data

# TODO: retrieve os.environ['SCT_DEBUG']
DEBUG = False  # Save img_sub


def dummy_blob(size_arr=(9, 9, 9), pixdim=(1, 1, 1), coordvox=None):
    """
    Create an image with a non-null voxels at coordinates specified by coordvox.
    :param size_arr:
    :param pixdim:
    :param coordvox: If None: will create a single voxel in the middle of the FOV.
      If tuple: (x,y,z): Create single voxel at specified coordinate
      If list of tuples: [(x1,y1,z1), (x2,y2,z2)]: Create multiple voxels.
    :return: Image object
    """
    # nx, ny, nz = size_arr
    data = np.zeros(size_arr)
    # if not specified, voxel coordinate is set at the middle of the volume
    if coordvox is None:
        coordvox = tuple([round(i / 2) for i in size_arr])
    elif isinstance(coordvox, list):
        for icoord in coordvox:
            data[icoord] = 1
    elif isinstance(coordvox, tuple):
        data[coordvox] = 1
    else:
        ValueError("Wrong type for coordvox")
    # Create image with default orientation LPI
    affine = np.eye(4)
    affine[0:3, 0:3] = affine[0:3, 0:3] * pixdim
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())
    return img


def dummy_centerline(size_arr=(9, 9, 9), pixdim=(1, 1, 1), subsampling=1, dilate_ctl=0, hasnan=False, zeroslice=[],
                     outlier=[], orientation='RPI', debug=False):
    """
    Create a dummy Image centerline of small size. Return the full and sub-sampled version along z. Voxel resolution
    on fully-sampled data is 1x1x1 mm (so, 2x undersampled data along z would have resolution of 1x1x2 mm).
    :param size_arr: tuple: (nx, ny, nz)
    :param pixdim: tuple: (px, py, pz)
    :param subsampling: int >=1. Subsampling factor along z. 1: no subsampling. 2: centerline defined every other z.
    :param dilate_ctl: Dilation of centerline. E.g., if dilate_ctl=1, result will be a square of 3x3 per slice.
                         if dilate_ctl=0, result will be a single pixel per slice.
    :param hasnan: Bool: Image has non-numerical values: nan, inf. In this case, do not subsample.
    :param zeroslice: list int: zero all slices listed in this param
    :param outlier: list int: replace the current point with an outlier at the corner of the image for the slices listed
    :param orientation:
    :param debug: Bool: Write temp files
    :return:
    """
    nx, ny, nz = size_arr
    # create regularized curve, within X-Z plane, located at y=ny/4, passing through the following points:
    x = np.array([round(nx/4.), round(nx/2.), round(3*nx/4.)])
    z = np.array([0, round(nz/2.), nz-1])
    # we use bspline (instead of poly) in order to avoid bad extrapolation at edges
    # see: https://github.com/neuropoly/spinalcordtoolbox/pull/2754
    xfit, _ = bspline(z, x, range(nz), 10)
    # p = P.fit(z, x, 3)
    # p = np.poly1d(np.polyfit(z, x, deg=3))
    data = np.zeros((nx, ny, nz))
    arr_ctl = np.array([xfit.astype(np.int),
                        [round(ny / 4.)] * len(range(nz)),
                        range(nz)], dtype=np.uint16)
    # Loop across dilation of centerline. E.g., if dilate_ctl=1, result will be a square of 3x3 per slice.
    for ixiy_ctl in itertools.product(range(-dilate_ctl, dilate_ctl+1, 1), range(-dilate_ctl, dilate_ctl+1, 1)):
        data[(arr_ctl[0] + ixiy_ctl[0]).tolist(),
             (arr_ctl[1] + ixiy_ctl[1]).tolist(),
             arr_ctl[2].tolist()] = 1
    # Zero specified slices
    if zeroslice is not []:
        data[:, :, zeroslice] = 0
    # Add outlier
    if outlier is not []:
        # First, zero all the slice
        data[:, :, outlier] = 0
        # Then, add point in the corner
        data[0, 0, outlier] = 1
    # Create image with default orientation LPI
    affine = np.eye(4)
    affine[0:3, 0:3] = affine[0:3, 0:3] * pixdim
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
    if debug:
        img_sub.save('tmp_dummy_seg_'+datetime.now().strftime("%Y%m%d%H%M%S%f")+'.nii.gz')
    return img, img_sub, arr_ctl


def dummy_segmentation(size_arr=(256, 256, 256), pixdim=(1, 1, 1), dtype=np.float64, orientation='LPI',
                       shape='rectangle', angle_RL=0, angle_AP=0, angle_IS=0, radius_RL=5.0, radius_AP=3.0,
                       degree=2, interleaved=False, zeroslice=[], debug=False):
    """Create a dummy Image with a ellipse or ones running from top to bottom in the 3rd dimension, and rotate the image
    to make sure that compute_csa and compute_shape properly estimate the centerline angle.
    :param size_arr: tuple: (nx, ny, nz)
    :param pixdim: tuple: (px, py, pz)
    :param dtype: Numpy dtype.
    :param orientation: Orientation of the image. Default: LPI
    :param shape: {'rectangle', 'ellipse'}
    :param angle_RL: int: angle around RL axis (in deg)
    :param angle_AP: int: angle around AP axis (in deg)
    :param angle_IS: int: angle around IS axis (in deg)
    :param radius_RL: float: 1st radius. With a, b = 50.0, 30.0 (in mm), theoretical CSA of ellipse is 4712.4
    :param radius_AP: float: 2nd radius
    :param degree: int: degree of polynomial fit
    :param interleaved: bool: create a dummy segmentation simulating interleaved acquisition
    :param zeroslice: list int: zero all slices listed in this param
    :param debug: Write temp files for debug
    :return: img: Image object
    """
    # Initialization
    padding = 15  # Padding size (isotropic) to avoid edge effect during rotation
    # Create a 3d array, with dimensions corresponding to x: RL, y: AP, z: IS
    nx, ny, nz = [int(size_arr[i] * pixdim[i]) for i in range(3)]
    data = np.zeros((nx, ny, nz))
    xx, yy = np.mgrid[:nx, :ny]

    # Create a dummy segmentation using polynomial function
    # create regularized curve, within Y-Z plane (A-P), located at x=nx/2:
    x = [round(nx / 2.)] * len(range(nz))
    # and passing through the following points:
    #y = np.array([round(ny / 4.), round(ny / 2.), round(3 * ny / 4.)])  # oblique curve (changing AP points across SI)
    y = [round(ny / 2.), round(ny / 2.), round(ny / 2.)]               # straight curve (same location of AP across SI)
    z = np.array([0, round(nz / 2.), nz - 1])
    # we use poly (instead of bspline) in order to allow change of scalar for each term of polynomial function
    p = np.polynomial.Polynomial.fit(z, y, deg=degree)

    # create two polynomial fits, by choosing random scalar for each term of both polynomial functions and then
    # interleave these two fits (one for odd slices, second one for even slices)
    if interleaved:
        p_even = copy.copy(p)
        p_odd = copy.copy(p)
        # choose random scalar for each term of polynomial function
        # even slices
        p_even.coef = [element * uniform(0.5, 1) for element in p_even.coef]
        # odd slices
        p_odd.coef = [element * uniform(0.5, 1) for element in p_odd.coef]
        # performs two polynomial fits - one will serve for even slices, second one for odd slices
        yfit_even = np.round(p_even(range(nz)))
        yfit_odd = np.round(p_odd(range(nz)))

        # combine even and odd polynomial fits
        yfit = np.zeros(nz)
        yfit[0:nz:2] = yfit_even[0:nz:2]
        yfit[1:nz:2] = yfit_odd[1:nz:2]
    # IF INTERLEAVED=FALSE, perform only one polynomial fit without modification of term's scalars
    else:
        yfit = np.round(p(range(nz)))   # has to be rounded for correct float -> int conversion in next step

    yfit = yfit.astype(np.int)
    # loop across slices and add object
    for iz in range(nz):
        if shape == 'rectangle':  # theoretical CSA: (a*2+1)(b*2+1)
            data[:, :, iz] = ((abs(xx - x[iz]) <= radius_RL) & (abs(yy - yfit[iz]) <= radius_AP)) * 1
        if shape == 'ellipse':
            data[:, :, iz] = (((xx - x[iz]) / radius_RL) ** 2 + ((yy - yfit[iz]) / radius_AP) ** 2 <= 1) * 1

    # Pad to avoid edge effect during rotation
    data = np.pad(data, padding, 'reflect')

    # ROTATION ABOUT IS AXIS
    # rotate (in deg), and re-grid using linear interpolation
    data_rotIS = rotate(data, angle_IS, resize=False, center=None, order=1, mode='constant', cval=0, clip=False,
                        preserve_range=False)

    # ROTATION ABOUT RL AXIS
    # Swap x-z axes (to make a rotation within y-z plane, because rotate will apply rotation on the first 2 dims)
    data_rotIS_swap = data_rotIS.swapaxes(0, 2)
    # rotate (in deg), and re-grid using linear interpolation
    data_rotIS_swap_rotRL = rotate(data_rotIS_swap, angle_RL, resize=False, center=None, order=1, mode='constant',
                                   cval=0, clip=False, preserve_range=False)
    # swap back
    data_rotIS_rotRL = data_rotIS_swap_rotRL.swapaxes(0, 2)

    # ROTATION ABOUT AP AXIS
    # Swap y-z axes (to make a rotation within x-z plane)
    data_rotIS_rotRL_swap = data_rotIS_rotRL.swapaxes(1, 2)
    # rotate (in deg), and re-grid using linear interpolation
    data_rotIS_rotRL_swap_rotAP = rotate(data_rotIS_rotRL_swap, angle_AP, resize=False, center=None, order=1,
                                         mode='constant', cval=0, clip=False, preserve_range=False)
    # swap back
    data_rot = data_rotIS_rotRL_swap_rotAP.swapaxes(1, 2)

    # Crop image (to remove padding)
    data_rot_crop = data_rot[padding:nx+padding, padding:ny+padding, padding:nz+padding]

    # Zero specified slices
    if zeroslice is not []:
        data_rot_crop[:, :, zeroslice] = 0

    # Create nibabel object
    xform = np.eye(4)
    for i in range(3):
        xform[i][i] = 1  # in [mm]
    nii = nib.nifti1.Nifti1Image(data_rot_crop.astype('float32'), xform)
    # resample to desired resolution
    nii_r = resample_nib(nii, new_size=pixdim, new_size_type='mm', interpolation='linear')
    # Create Image object. Default orientation is LPI.
    # For debugging add .save() at the end of the command below
    img = Image(nii_r.get_data(), hdr=nii_r.header, dim=nii_r.header.get_data_shape())
    # Update orientation
    img.change_orientation(orientation)
    if debug:
        img.save('tmp_dummy_seg_'+datetime.now().strftime("%Y%m%d%H%M%S%f")+'.nii.gz')
    return img

def dummy_segmentation_4d(vol_num=10, create_bvecs=False, size_arr=(256, 256, 256), pixdim=(1, 1, 1), dtype=np.float64,
                          orientation='LPI', shape='rectangle', angle_RL=0, angle_AP=0, angle_IS=0, radius_RL=5.0,
                          radius_AP=3.0, degree=2, interleaved=False, zeroslice=[], debug=False):
    """
    Create a dummy 4D segmentation (dMRI/fMRI) and dummy bvecs file (optional)
    :param vol_num: int: number of volumes in 4D data
    :param create_bvecs: bool: create dummy bvecs file (necessary e.g. for sct_dmri_moco)
    other parameters are same as in dummy_segmentation function
    :return: Image object
    """

    img_list = []

    # Loop across individual volumes of 4D data
    for volume in range(0,vol_num):
        # set debug=True in line below for saving individual volumes into individual nii files
        img_list.append(dummy_segmentation(size_arr=size_arr, pixdim=pixdim, dtype=dtype, orientation=orientation,
                                           shape=shape, angle_RL=angle_RL, angle_AP=angle_AP, angle_IS=angle_IS,
                                           radius_RL=radius_RL, radius_AP=radius_AP, degree=degree, zeroslice=zeroslice,
                                           interleaved=interleaved, debug=False))

    # Concatenate individual 3D images into 4D data
    img_4d = concat_data(img_list, 3)
    if debug:
        out_name = datetime.now().strftime("%Y%m%d%H%M%S%f")
        file_4d_data = 'tmp_dummy_4d_' + out_name + '.nii.gz'
        img_4d.save(file_4d_data, verbose=0)

    # Create a dummy bvecs file (necessary e.g. for sct_dmri_moco)
    if create_bvecs:
        n_b0 = 1                # number of b0
        n_dwi = vol_num-n_b0    # number of dwi
        bvecs_dummy = ['', '', '']
        bvec_b0 = np.array([[0.0, 0.0, 0.0]] * n_b0)
        bvec_dwi = np.array([[uniform(0,1), uniform(0,1), uniform(0,1)]] * n_dwi)
        bvec = np.concatenate((bvec_b0,bvec_dwi),axis=0)
        # Concatenate bvecs
        for i in (0, 1, 2):
            bvecs_dummy[i] += ' '.join(str(v) for v in map(lambda n: '%.16f' % n, bvec[:, i]))
            bvecs_dummy[i] += ' '
        bvecs_concat = '\n'.join(str(v) for v in bvecs_dummy)  # transform list into lines of strings
        if debug:
            new_f = open('tmp_dummy_4d_' + out_name + '.bvec', 'w')
            new_f.write(bvecs_concat)
            new_f.close()

    return img_4d
