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

import sys, os

import numpy as np
from skimage import transform, img_as_float, img_as_uint

import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from msct_parser import Parser
from sct_straighten_spinalcord import smooth_centerline

# Default parameters
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        self.interp = 'sinc'  # final interpolation
        self.deg_poly = 10  # maximum degree of polynomial function for fitting centerline.
        self.remove_temp_files = 1  # remove temporary files
        self.verbose = 1


#=======================================================================================================================
# main
#=======================================================================================================================
def main(fname_anat, fname_centerline, degree_poly, centerline_fitting, interp, remove_temp_files, verbose):

    # load input image
    im_anat = Image(fname_anat)
    nx, ny, nz, nt, px, py, pz, pt = im_anat.dim
    # re-oriente to RPI
    orientation_native = im_anat.orientation
    im_anat.change_orientation("RPI")

    # load centerline
    im_centerline = Image(fname_centerline).change_orientation("RPI")

    # smooth centerline and return fitted coordinates in voxel space
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
        im_centerline, algo_fitting=centerline_fitting, type_window='hanning', window_length=50,
        nurbs_pts_number=3000, phys_coordinates=False, verbose=verbose, all_slices=True)

    # compute translation for each slice, such that the flattened centerline is centered in the medial plane (R-L) and
    # avoid discontinuity in slices where there is no centerline (in which case, simply copy the translation of the
    # closest Z).
    # first, get zmin and zmax spanned by the centerline (i.e. with non-zero values)
    indz_centerline = np.where([np.sum(im_centerline.data[:, :, iz]) for iz in range(nz)])[0]
    zmin, zmax = indz_centerline[0], indz_centerline[-1]
    # then, extend the centerline by copying values below zmin and above zmax
    x_centerline_extended = np.concatenate([np.ones(zmin) * x_centerline_fit[0],
                                            x_centerline_fit,
                                            np.ones(nz-zmax) * x_centerline_fit[-1]])

    # change type to float32 and scale between -1 and 1 as requested by img_as_float(). See #1790, #2069
    im_anat_flattened = msct_image.change_type(im_anat, np.float32)
    min_data, max_data = np.min(im_anat_flattened.data), np.max(im_anat_flattened.data)
    im_anat_flattened.data = 2 * im_anat_flattened.data/(max_data - min_data) - 1

    # loop across slices and apply translation
    for iz in range(nz):
        # compute translation along x (R-L)
        translation_x = x_centerline_extended[iz] - np.round(nx/2.0)
        # apply transformation to 2D image with linear interpolation
        # tform = tf.SimilarityTransform(scale=1, rotation=0, translation=(translation_x, 0))
        tform = transform.SimilarityTransform(translation=(0, translation_x))
        # important to force input in float to skikit image, because it will output float values
        img = img_as_float(im_anat_flattened.data[:, :, iz])
        img_reg = transform.warp(img, tform)
        im_anat_flattened.data[:, :, iz] = img_reg  # img_as_uint(img_reg)

    # change back to native orientation
    im_anat_flattened.change_orientation(orientation_native)
    # save output
    fname_out = sct.add_suffix(fname_anat, '_flatten')
    im_anat_flattened.save(fname_out)

    sct.display_viewer_syntax([fname_anat, fname_out])


def get_parser():
    param_default = Param()
    parser = Parser(__file__)
    parser.usage.set_description("""Flatten the spinal cord in the sagittal plane (to make nice pictures). Output data
    type is float32 (regardless of input type) to minimize loss of precision during conversion.""")
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
    parser.add_option(name='-x',
                      type_value='multiple_choice',
                      description='Final interpolation.',
                      mandatory=False,
                      example=['nearestneighbour', 'trilinear', 'sinc'],
                      default_value=str(param_default.interp))
    parser.add_option(name='-d',
                      type_value='int',
                      description='Degree of fitting polynome.',
                      mandatory=False,
                      default_value=param_default.deg_poly)
    parser.add_option(name='-f',
                      type_value='multiple_choice',
                      description='Fitting algorithm.',
                      mandatory=False,
                      example=['hanning', 'nurbs'],
                      default_value='hanning')
    parser.add_option(name='-r',
                      type_value='multiple_choice',
                      description='Removes the temporary folder and debug folder used for the algorithm at the end of execution',
                      mandatory=False,
                      default_value=str(param_default.remove_temp_files),
                      example=['0', '1'])
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


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # initialize parameters
    param = Param()
    param_default = Param()

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_anat = arguments['-i']
    fname_centerline = arguments['-s']
    degree_poly = arguments['-d']
    centerline_fitting = arguments['-f']
    interp = arguments['-x']
    remove_temp_files = arguments['-r']
    verbose = int(arguments['-v'])

    # call main function
    main(fname_anat, fname_centerline, degree_poly, centerline_fitting, interp, remove_temp_files, verbose)
