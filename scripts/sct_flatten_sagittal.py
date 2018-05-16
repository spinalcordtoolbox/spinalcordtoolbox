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


import sys, os

import numpy as np

import sct_utils as sct
from msct_nurbs import NURBS
from msct_image import Image
from msct_parser import Parser
from sct_straighten_spinalcord import smooth_centerline
from skimage import transform, img_as_float, img_as_uint

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

    # Display arguments
    sct.printv('\nCheck input arguments...')
    sct.printv('  Input volume ...................... ' + fname_anat)
    sct.printv('  Centerline ........................ ' + fname_centerline)
    sct.printv('')

    # load input image
    im_anat = Image(fname_anat)
    nx, ny, nz, nt, px, py, pz, pt = im_anat.dim
    # re-oriente to RPI
    orientation_native = im_anat.change_orientation('RPI')

    # load centerline
    im_centerline = Image(fname_centerline)
    im_centerline.change_orientation('RPI')

    # smooth centerline and return fitted coordinates in voxel space
    x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(
        im_centerline, algo_fitting='hanning', type_window='hanning', window_length=50,
        nurbs_pts_number=3000, phys_coordinates=False, verbose=verbose, all_slices=True)

    # compute translation for each slice, such that the flattened centerline is centered in the medial plane (R-L) and
    # avoid discontinuity in slices where there is no centerline (in which case, simply copy the translation of the
    # closest Z).
    # first, get zmin and zmax spanned by the centerline (i.e. with non-zero values)
    indz_centerline = np.where([np.sum(im_centerline.data[:, :, iz]) for iz in range(nz)])[0]
    zmin, zmax = indz_centerline[0], indz_centerline[-1]
    # then, extend the centerline by padding values below zmin and above zmax
    x_centerline_extended = np.concatenate([np.ones(zmin) * x_centerline_fit[0], x_centerline_fit, np.ones(nz-zmax-1) * x_centerline_fit[-1]])

    # loop across slices and apply translation
    im_anat_flattened = im_anat.copy()
    im_anat_flattened.changeType('uint16')  # force uint16 because outputs are converted using img_as_uint()
    for iz in range(nz):
        # compute translation along x (R-L)
        translation_x = x_centerline_extended[iz] - round(nx/2.0)
        # apply transformation to 2D image with linear interpolation
        # tform = tf.SimilarityTransform(scale=1, rotation=0, translation=(translation_x, 0))
        tform = transform.SimilarityTransform(translation=(0, translation_x))
        # important to force input in float to skikit image, because it will output float values
        img = img_as_float(im_anat.data[:, :, iz])
        img_reg = transform.warp(img, tform)
        im_anat_flattened.data[:, :, iz] = img_as_uint(img_reg)

    # change back to native orientation
    im_anat_flattened.change_orientation(orientation_native)
    # save output
    fname_out = sct.add_suffix(fname_anat, '_flatten')
    im_anat_flattened.setFileName(fname_out)
    im_anat_flattened.save()

    sct.display_viewer_syntax([fname_anat, fname_out])


def b_spline_centerline(x_centerline, y_centerline, z_centerline):
    """Give a better fitting of the centerline than the method 'spline_centerline' using b-splines"""

    points = [[x_centerline[n], y_centerline[n], z_centerline[n]] for n in range(len(x_centerline))]

    nurbs = NURBS(3, len(z_centerline) * 3, points, nbControl=None, verbose=2)  # for the third argument (number of points), give at least len(z_centerline)
    # (len(z_centerline)+500 or 1000 is ok)
    P = nurbs.getCourbe3D()
    x_centerline_fit = P[0]
    y_centerline_fit = P[1]

    return x_centerline_fit, y_centerline_fit


def polynome_centerline(x_centerline, y_centerline, z_centerline):
    """Fit polynomial function through centerline"""

    # Fit centerline in the Z-X plane using polynomial function
    sct.printv('\nFit centerline in the Z-X plane using polynomial function...')
    coeffsx = numpy.polyfit(z_centerline, x_centerline, deg=5)
    polyx = numpy.poly1d(coeffsx)
    x_centerline_fit = numpy.polyval(polyx, z_centerline)

    # Fit centerline in the Z-Y plane using polynomial function
    sct.printv('\nFit centerline in the Z-Y plane using polynomial function...')
    coeffsy = numpy.polyfit(z_centerline, y_centerline, deg=5)
    polyy = numpy.poly1d(coeffsy)
    y_centerline_fit = numpy.polyval(polyy, z_centerline)

    return x_centerline_fit, y_centerline_fit


def get_parser():
    param_default = Param()
    parser = Parser(__file__)
    parser.usage.set_description("""Flatten the spinal cord in the sagittal plane (to make nice pictures).""")
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
                      example=['polynome', 'nurbs'],
                      default_value='nurbs')
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
