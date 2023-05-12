# Collection of functions to create data for testing

import numpy as np
from datetime import datetime
import itertools

import nibabel as nib

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.curve_fitting import bspline

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
    # see: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2754
    xfit, _ = bspline(z, x, range(nz), 10)
    # p = P.fit(z, x, 3)
    # p = np.poly1d(np.polyfit(z, x, deg=3))
    data = np.zeros((nx, ny, nz))
    arr_ctl = np.array([xfit.astype(int),
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
