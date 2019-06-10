#!/usr/bin/env python
#########################################################################################
#
# Flatten spinal cord in sagittal plane.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2014-06-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import, division

import sys

import numpy as np
from skimage import transform, img_as_float

import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from msct_parser import Parser


# Default parameters
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.interp = 'sinc'  # final interpolation
        self.remove_temp_files = 1  # remove temporary files
        self.verbose = 1


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
                                            np.ones(nz-zmax) * x_centerline_fit[-1]])

    # change type to float32 and scale between -1 and 1 as requested by img_as_float(). See #1790, #2069
    im_anat_flattened = msct_image.change_type(im_anat, np.float32)
    min_data, max_data = np.min(im_anat_flattened.data), np.max(im_anat_flattened.data)
    im_anat_flattened.data = 2 * im_anat_flattened.data/(max_data - min_data) - 1

    # loop and translate each axial slice, such that the flattened centerline is centered in the medial plane (R-L)
    for iz in range(nz):
        # compute translation along x (R-L)
        translation_x = x_centerline_extended[iz] - np.round(nx/2.0)
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


def main(fname_anat, fname_centerline, verbose):
    """
    Main function
    :param fname_anat:
    :param fname_centerline:
    :param verbose:
    :return:
    """
    # load input images
    im_anat = Image(fname_anat)
    im_centerline = Image(fname_centerline)

    # flatten sagittal
    im_anat_flattened = flatten_sagittal(im_anat, im_centerline, verbose)

    # save output
    fname_out = sct.add_suffix(fname_anat, '_flatten')
    im_anat_flattened.save(fname_out)

    sct.display_viewer_syntax([fname_anat, fname_out])


def get_parser():
    param_default = Param()
    parser = Parser(__file__)
    parser.usage.set_description("""Flatten the spinal cord such within the medial sagittal plane. Useful to make nice 
    pictures. Output data has suffix _flatten. Output type is float32 (regardless of input type) to minimize loss of 
    precision during conversion.""")
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='Input volume.',
                      mandatory=True,
                      example='t2.nii.gz')
    parser.add_option(name='-s',
                      type_value='image_nifti',
                      description='Spinal cord segmentation or centerline.',
                      mandatory=True,
                      example='t2_seg.nii.gz')
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='0: no verbose (default), 1: min verbose, 2: verbose + figures',
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value=str(param_default.verbose))
    parser.add_option(name='-h',
                      type_value=None,
                      description='Display this help',
                      mandatory=False)

    return parser


if __name__ == "__main__":
    sct.init_sct()
    # initialize parameters
    param = Param()
    param_default = Param()

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_anat = arguments['-i']
    fname_centerline = arguments['-s']
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # call main function
    main(fname_anat, fname_centerline, verbose)
