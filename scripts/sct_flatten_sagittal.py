#!/usr/bin/env python
#########################################################################################
#
# Flatten spinal cord in sagittal plane.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad, Francisco Perdigon Romero
# Modified: 2018-05-13
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys, os
import numpy as np


import sct_utils as sct
from msct_nurbs import NURBS
from sct_image import get_orientation_3d
from msct_image import Image
from msct_parser import Parser



# Default parameters
class Param:
    # The constructor
    def __init__(self):
        self.debug = 0
        #self.interp = 'sinc'  # final interpolation
        #self.deg_poly = 10  # maximum degree of polynomial function for fitting centerline.
        #self.remove_temp_files = 1  # remove temporary files
        self.verbose = 1


#=======================================================================================================================
# main
#=======================================================================================================================
def main(fname_anat, fname_centerline, verbose):

    # extract path of the script
    path_script = os.path.dirname(__file__) + '/'

    # Parameters for debug mode
    if param.debug == 1:
        sct.printv('\n*** WARNING: DEBUG MODE ON ***\n')
        path_sct_data = os.environ.get("SCT_TESTING_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__))), "testing_data")
        fname_anat = path_sct_data + '/t2/t2.nii.gz'
        fname_centerline = path_sct_data + '/t2/t2_seg.nii.gz'

    # extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)

    # Display arguments
    sct.printv('\nCheck input arguments...')
    sct.printv('  Input volume ...................... ' + fname_anat)
    sct.printv('  Centerline ........................ ' + fname_centerline)
    sct.printv('')

    # Get input image orientation
    im_anat = Image(fname_anat)
    anat_ima_orientation_original = get_orientation_3d(im_anat)
    im_centerline = Image(fname_centerline)


    # Process centerline
    #==========================================================================================
    # For now faltten sagittal only support the OptiC centerline extraction
    # TODO: Add support for probabilistic centerline
    # TODO: Add support for centerline extraction from spinal cord segmentation image
    # TODO: Add support for NURB centerline adjustment
    # TODO: Add support for polynomial centerline adjustment
    # TODO: Add support for other sagittal flatten methods like 'trilinear', 'sinc'
    # TODO: Add 4D support


    # Process flatten sagittal
    #==========================================================================================
    im_flatten_sag = flatten_sagittal_linear(im_anat, im_centerline)

    # Generate output file (in current folder)
    sct.printv('\nGenerate output file (in current folder)...')
    im_flatten_sag.change_orientation(orientation=anat_ima_orientation_original)
    fname_out = os.path.join(path_anat,file_anat + '_flatten' + ext_anat)
    im_flatten_sag.setFileName(fname_out)
    im_flatten_sag.save()

    sct.display_viewer_syntax([fname_out])


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
    coeffsx = np.polyfit(z_centerline, x_centerline, deg=5)
    polyx = np.poly1d(coeffsx)
    x_centerline_fit = np.polyval(polyx, z_centerline)

    # Fit centerline in the Z-Y plane using polynomial function
    sct.printv('\nFit centerline in the Z-Y plane using polynomial function...')
    coeffsy = np.polyfit(z_centerline, y_centerline, deg=5)
    polyy = np.poly1d(coeffsy)
    y_centerline_fit = np.polyval(polyy, z_centerline)

    return x_centerline_fit, y_centerline_fit


def flatten_sagittal_linear(im_anat, im_centerline):
    """"
    Flatten sagittal linear method
    @:param im_anat: Anatomical image. File type Image. Must be oriented in RPI
    @:param im_centerline: Centerline image. File type Image. Must be oriented in RPI
    @:return im_flatten_sag_data: flatten sagittal image. File type Image. RPI oriented
    """

    # Reorient input data into RL PA IS orientation
    im_anat.change_orientation(orientation='RPI')
    im_centerline.change_orientation(orientation='RPI')

    sct.printv('\n Doing flatten sagittal...')

    X = np.zeros(im_centerline.data.shape[-1], dtype=np.int16)
    Y = np.zeros(im_centerline.data.shape[-1], dtype=np.int16)
    for z_axis in range(im_centerline.data.shape[-1]):
        X[z_axis], Y[z_axis] = im_centerline.data[:, :, z_axis].nonzero()

    im_x, im_y, im_z = im_anat.data.shape

    imf_x = 2 * im_x + 1  # With these value for sure the loop will not traspass the max index

    im_flatten_sag_data = np.zeros((imf_x, im_y, im_z), dtype=im_anat.data.dtype)

    for z_axis in range(len(X)):
        x_start = int(imf_x / 2) - X[z_axis]
        x_end = x_start + im_x
        im_flatten_sag_data[x_start: x_end, :, z_axis] = im_anat.data[:, :, z_axis]

    im_flatten_sag = im_anat.copy()
    im_flatten_sag.orientation = 'RPI'
    im_flatten_sag.data = im_flatten_sag_data

    sct.printv('\n Done')
    return im_flatten_sag

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
                      description='Centerline. Recommended sct_get_centerline',
                      mandatory=True,
                      example='centerline.nii.gz')
    # parser.add_option(name='-c',
    #                   type_value=None,
    #                   description='Centerline.',
    #                   mandatory=False,
    #                   deprecated_by='-s')
    # parser.add_option(name='-x',
    #                   type_value='multiple_choice',
    #                   description='Final interpolation.',
    #                   mandatory=False,
    #                   example=['nearestneighbour', 'trilinear', 'sinc'],
    #                   default_value=str(param_default.interp))
    # parser.add_option(name='-d',
    #                   type_value='int',
    #                   description='Degree of fitting polynome.',
    #                   mandatory=False,
    #                   default_value=param_default.deg_poly)
    # parser.add_option(name='-f',
    #                   type_value='multiple_choice',
    #                   description='Fitting algorithm.',
    #                   mandatory=False,
    #                   example=['polynome', 'nurbs'],
    #                   default_value='nurbs')
    # parser.add_option(name='-r',
    #                   type_value='multiple_choice',
    #                   description='Removes the temporary folder and debug folder used for the algorithm at the end of execution',
    #                   mandatory=False,
    #                   default_value=str(param_default.remove_temp_files),
    #                   example=['0', '1'])
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='1: display on, 0: display off (default)',
                      mandatory=False,
                      example=['0', '1'],
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
    #degree_poly = arguments['-d']
    #centerline_fitting = arguments['-f']
    #interp = arguments['-x']
    #remove_temp_files = arguments['-r']
    verbose = int(arguments['-v'])

    # call main function
    #main(fname_anat, fname_centerline, degree_poly, centerline_fitting, interp, remove_temp_files, verbose)
    main(fname_anat, fname_centerline, verbose)
