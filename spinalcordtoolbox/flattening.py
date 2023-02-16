#########################################################################################
#
# Module for spinal cord flattening in different planes.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import logging

import numpy as np
from skimage import transform, img_as_float

from spinalcordtoolbox.image import change_type
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline

logger = logging.getLogger(__name__)


def flatten_sagittal(im_anat, im_centerline, verbose):
    """
    Flatten a 3D volume using the segmentation, such that the spinal cord is centered in the R-L medial plane.

    :param im_anat:
    :param im_centerline:
    :param verbose:
    :return:
    """
    # re-oriente to RPI
    orientation_native = im_anat.orientation
    im_anat.change_orientation("RPI")
    im_centerline.change_orientation("RPI")
    nx, ny, nz, nt, px, py, pz, pt = im_anat.dim

    # smooth centerline and return fitted coordinates in voxel space
    _, arr_ctl, _, _ = get_centerline(im_centerline, param=ParamCenterline(), verbose=verbose)
    x_centerline_fit, y_centerline_fit, z_centerline = arr_ctl

    # Extend the centerline by copying values below zmin and above zmax to avoid discontinuities
    zmin, zmax = z_centerline.min().astype(int), z_centerline.max().astype(int)
    x_centerline_extended = np.concatenate([np.ones(zmin) * x_centerline_fit[0],
                                            x_centerline_fit,
                                            np.ones(nz - zmax) * x_centerline_fit[-1]])

    # change type to float32 and scale between -1 and 1 as requested by img_as_float(). See #1790, #2069
    im_anat_flattened = change_type(im_anat, np.float32)
    min_data, max_data = np.min(im_anat_flattened.data), np.max(im_anat_flattened.data)
    im_anat_flattened.data = 2 * im_anat_flattened.data / (max_data - min_data) - 1

    # loop and translate each axial slice, such that the flattened centerline is centered in the medial plane (R-L)
    for iz in range(nz):
        # compute translation along x (R-L)
        translation_x = x_centerline_extended[iz] - np.round(nx / 2.0)
        # apply transformation to 2D image with linear interpolation
        # tform = tf.SimilarityTransform(scale=1, rotation=0, translation=(translation_x, 0))
        tform = transform.SimilarityTransform(translation=(0, translation_x))
        # important to force input in float to skikit image, because it will output float values
        img = img_as_float(im_anat_flattened.data[:, :, iz])
        img_reg = transform.warp(img, tform)
        im_anat_flattened.data[:, :, iz] = img_reg

    # change back to native orientation
    im_anat_flattened.change_orientation(orientation_native)

    return im_anat_flattened
