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

# TODO: remove FSL dependency

import os
import getopt
import sys
import commands
import nibabel
import numpy
from shutil import move
import sct_utils as sct
from msct_nurbs import NURBS
from sct_utils import fsloutput
from sct_image import get_orientation_3d, set_orientation
from msct_image import Image
from sct_image import split_data, concat_data
from msct_parser import Parser


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

    # extract path of the script
    path_script = os.path.dirname(__file__) + '/'

    # Parameters for debug mode
    if param.debug == 1:
        sct.printv('\n*** WARNING: DEBUG MODE ON ***\n')
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
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
    input_image_orientation = get_orientation_3d(im_anat)

    # Reorient input data into RL PA IS orientation
    im_centerline = Image(fname_centerline)
    im_anat_orient = set_orientation(im_anat, 'RPI')
    im_anat_orient.setFileName('tmp.anat_orient.nii')
    im_centerline_orient = set_orientation(im_centerline, 'RPI')
    im_centerline_orient.setFileName('tmp.centerline_orient.nii')

    # Open centerline
    #==========================================================================================
    sct.printv('\nGet dimensions of input centerline...')
    nx, ny, nz, nt, px, py, pz, pt = im_centerline_orient.dim
    sct.printv('.. matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz))
    sct.printv('.. voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm')

    sct.printv('\nOpen centerline volume...')
    data = im_centerline_orient.data

    X, Y, Z = (data > 0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)

    # loop across z and associate x,y coordinate with the point having maximum intensity
    x_centerline = [0 for iz in range(min_z_index, max_z_index + 1, 1)]
    y_centerline = [0 for iz in range(min_z_index, max_z_index + 1, 1)]
    z_centerline = [iz for iz in range(min_z_index, max_z_index + 1, 1)]

    # Two possible scenario:
    # 1. the centerline is probabilistic: each slices contains voxels with the probability of containing the centerline [0:...:1]
    # We only take the maximum value of the image to aproximate the centerline.
    # 2. The centerline/segmentation image contains many pixels per slice with values {0,1}.
    # We take all the points and approximate the centerline on all these points.

    X, Y, Z = ((data < 1) * (data > 0)).nonzero()  # X is empty if binary image
    if (len(X) > 0):  # Scenario 1
        for iz in range(min_z_index, max_z_index + 1, 1):
            x_centerline[iz - min_z_index], y_centerline[iz - min_z_index] = numpy.unravel_index(data[:, :, iz].argmax(), data[:, :, iz].shape)
    else:  # Scenario 2
        for iz in range(min_z_index, max_z_index + 1, 1):
            x_seg, y_seg = (data[:, :, iz] > 0).nonzero()
            if len(x_seg) > 0:
                x_centerline[iz - min_z_index] = numpy.mean(x_seg)
                y_centerline[iz - min_z_index] = numpy.mean(y_seg)

    # TODO: find a way to do the previous loop with this, which is more neat:
    # [numpy.unravel_index(data[:,:,iz].argmax(), data[:,:,iz].shape) for iz in range(0,nz,1)]

    # clear variable
    del data

    # Fit the centerline points with the kind of curve given as argument of the script and return the new smoothed coordinates
    if centerline_fitting == 'nurbs':
        try:
            x_centerline_fit, y_centerline_fit = b_spline_centerline(x_centerline, y_centerline, z_centerline)
        except ValueError:
            sct.printv("splines fitting doesn't work, trying with polynomial fitting...\n")
            x_centerline_fit, y_centerline_fit = polynome_centerline(x_centerline, y_centerline, z_centerline)
    elif centerline_fitting == 'polynome':
        x_centerline_fit, y_centerline_fit = polynome_centerline(x_centerline, y_centerline, z_centerline)

    #==========================================================================================
    # Split input volume
    sct.printv('\nSplit input volume...')
    im_anat_orient_split_list = split_data(im_anat_orient, 2)
    file_anat_split = []
    for im in im_anat_orient_split_list:
        file_anat_split.append(im.absolutepath)
        im.save()

    # initialize variables
    file_mat_inv_cumul = ['tmp.mat_inv_cumul_Z' + str(z).zfill(4) for z in range(0, nz, 1)]
    z_init = min_z_index
    displacement_max_z_index = x_centerline_fit[z_init - min_z_index] - x_centerline_fit[max_z_index - min_z_index]

    # write centerline as text file
    sct.printv('\nGenerate fitted transformation matrices...')
    file_mat_inv_cumul_fit = ['tmp.mat_inv_cumul_fit_Z' + str(z).zfill(4) for z in range(0, nz, 1)]
    for iz in range(min_z_index, max_z_index + 1, 1):
        # compute inverse cumulative fitted transformation matrix
        fid = open(file_mat_inv_cumul_fit[iz], 'w')
        if (x_centerline[iz - min_z_index] == 0 and y_centerline[iz - min_z_index] == 0):
            displacement = 0
        else:
            displacement = x_centerline_fit[z_init - min_z_index] - x_centerline_fit[iz - min_z_index]
        fid.write('%i %i %i %f\n' % (1, 0, 0, displacement))
        fid.write('%i %i %i %f\n' % (0, 1, 0, 0))
        fid.write('%i %i %i %i\n' % (0, 0, 1, 0))
        fid.write('%i %i %i %i\n' % (0, 0, 0, 1))
        fid.close()

    # we complete the displacement matrix in z direction
    for iz in range(0, min_z_index, 1):
        fid = open(file_mat_inv_cumul_fit[iz], 'w')
        fid.write('%i %i %i %f\n' % (1, 0, 0, 0))
        fid.write('%i %i %i %f\n' % (0, 1, 0, 0))
        fid.write('%i %i %i %i\n' % (0, 0, 1, 0))
        fid.write('%i %i %i %i\n' % (0, 0, 0, 1))
        fid.close()
    for iz in range(max_z_index + 1, nz, 1):
        fid = open(file_mat_inv_cumul_fit[iz], 'w')
        fid.write('%i %i %i %f\n' % (1, 0, 0, displacement_max_z_index))
        fid.write('%i %i %i %f\n' % (0, 1, 0, 0))
        fid.write('%i %i %i %i\n' % (0, 0, 1, 0))
        fid.write('%i %i %i %i\n' % (0, 0, 0, 1))
        fid.close()

    # apply transformations to data
    sct.printv('\nApply fitted transformation matrices...')
    file_anat_split_fit = ['tmp.anat_orient_fit_Z' + str(z).zfill(4) for z in range(0, nz, 1)]
    for iz in range(0, nz, 1):
        # forward cumulative transformation to data
        sct.run(fsloutput + 'flirt -in ' + file_anat_split[iz] + ' -ref ' + file_anat_split[iz] + ' -applyxfm -init ' + file_mat_inv_cumul_fit[iz] + ' -out ' + file_anat_split_fit[iz] + ' -interp ' + interp)

    # Merge into 4D volume
    sct.printv('\nMerge into 4D volume...')
    from glob import glob
    im_to_concat_list = [Image(fname) for fname in glob('tmp.anat_orient_fit_Z*.nii')]
    im_concat_out = concat_data(im_to_concat_list, 2)
    im_concat_out.setFileName('tmp.anat_orient_fit.nii')
    im_concat_out.save()
    # sct.run(fsloutput+'fslmerge -z tmp.anat_orient_fit tmp.anat_orient_fit_z*')

    # Reorient data as it was before
    sct.printv('\nReorient data back into native orientation...')
    fname_anat_fit_orient = set_orientation(im_concat_out.absolutepath, input_image_orientation, filename=True)
    move(fname_anat_fit_orient, 'tmp.anat_orient_fit_reorient.nii')

    # Generate output file (in current folder)
    sct.printv('\nGenerate output file (in current folder)...')
    sct.generate_output_file('tmp.anat_orient_fit_reorient.nii', file_anat + '_flatten' + ext_anat)

    # Delete temporary files
    if remove_temp_files == 1:
        sct.printv('\nDelete temporary files...')
        sct.run('rm -rf tmp.*')

    # to view results
    sct.printv('\nDone! To view results, type:')
    sct.printv('fslview ' + file_anat + ext_anat + ' ' + file_anat + '_flatten' + ext_anat + ' &\n')


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
                      description='Centerline.',
                      mandatory=True,
                      example='centerline.nii.gz')
    parser.add_option(name='-c',
                      type_value=None,
                      description='Centerline.',
                      mandatory=False,
                      deprecated_by='-s')
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
    sct.start_stream_logger()
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
