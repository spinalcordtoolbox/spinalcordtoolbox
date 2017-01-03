#!/usr/bin/env python
#
# This program straightens the spinal cord of an anatomic image, apply a smoothing in the z dimension and apply
# the inverse warping field to get back the curved spinal cord but smoothed.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Simon Levy
# Modified: 2014-09-01
#
# About the license: see the file LICENSE.TXT
#########################################################################################
# TODO: maybe no need to convert RPI at the beginning because strainghten spinal cord already does it!

import os
import shutil
import sys
import time
from shutil import move

import numpy as np

import sct_utils as sct
from msct_parser import Parser
from sct_convert import convert
from sct_image import set_orientation


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Smooth the spinal cord along its centerline. Steps are:\n'
                                 '1) Spinal cord is straightened (using centerline),\n'
                                 '2) a Gaussian kernel is applied in the superior-inferior direction,\n'
                                 '3) then cord is de-straightened as originally.\n')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image to smooth",
                      mandatory=True,
                      example='data.nii.gz')
    parser.add_option(name="-s",
                      type_value="file",
                      description="Spinal cord centerline or segmentation",
                      mandatory=True,
                      example='data_centerline.nii.gz')
    parser.add_option(name="-c",
                      type_value=None,
                      description="Spinal cord centerline or segmentation",
                      mandatory=False,
                      deprecated_by='-s')
    parser.add_option(name="-smooth",
                      type_value="int",
                      description="Sigma of the smoothing Gaussian kernel (in mm).",
                      mandatory=False,
                      default_value=3,
                      example='2')
    parser.usage.addSection('MISC')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    return parser


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    # Initialization
    # fname_anat = ''
    # fname_centerline = ''
    sigma = 3 # default value of the standard deviation for the Gaussian smoothing (in terms of number of voxels)
    # remove_temp_files = param.remove_temp_files
    # verbose = param.verbose
    start_time = time.time()

    parser = get_parser()
    arguments = parser.parse(args)

    fname_anat = arguments['-i']
    fname_centerline = arguments['-s']
    if '-smooth' in arguments:
        sigma = arguments['-smooth']
    if '-r' in arguments:
        remove_temp_files = int(arguments['-r'])
    if '-v' in arguments:
        verbose = int(arguments['-v'])

    # Display arguments
    print '\nCheck input arguments...'
    print '  Volume to smooth .................. ' + fname_anat
    print '  Centerline ........................ ' + fname_centerline
    print '  Sigma (mm) ........................ '+str(sigma)
    print '  Verbose ........................... '+str(verbose)

    # Check that input is 3D:
    from msct_image import Image
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_anat).dim
    dim = 4  # by default, will be adjusted later
    if nt == 1:
        dim = 3
    if nz == 1:
        dim = 2
    if dim == 4:
        sct.printv('WARNING: the input image is 4D, please split your image to 3D before smoothing spinalcord using :\n'
                   'sct_image -i '+fname_anat+' -split t -o '+fname_anat, verbose, 'warning')
        sct.printv('4D images not supported, aborting ...', verbose, 'error')

    # Extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    path_centerline, file_centerline, ext_centerline = sct.extract_fname(fname_centerline)

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.tmp_create(verbose)

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    shutil.copy(fname_anat, path_tmp + 'anat' + ext_anat)
    shutil.copy(fname_centerline, path_tmp + 'centerline' + ext_centerline)

    # go to tmp folder
    os.chdir(path_tmp)

    # convert to nii format
    convert('anat'+ext_anat, 'anat.nii')
    convert('centerline'+ext_centerline, 'centerline.nii')

    # Change orientation of the input image into RPI
    print '\nOrient input volume to RPI orientation...'
    fname_anat_rpi = set_orientation('anat.nii', 'RPI', filename=True)
    move(fname_anat_rpi, 'anat_rpi.nii')
    # Change orientation of the input image into RPI
    print '\nOrient centerline to RPI orientation...'
    fname_centerline_rpi = set_orientation('centerline.nii', 'RPI', filename=True)
    move(fname_centerline_rpi, 'centerline_rpi.nii')

    # Straighten the spinal cord
    # straighten segmentation
    sct.printv('\nStraighten the spinal cord using centerline/segmentation...', verbose)
    # check if warp_curve2straight and warp_straight2curve already exist (i.e. no need to do it another time)
    if os.path.isfile('../warp_curve2straight.nii.gz') and os.path.isfile('../warp_straight2curve.nii.gz') and os.path.isfile('../straight_ref.nii.gz'):
        # if they exist, copy them into current folder
        sct.printv('WARNING: Straightening was already run previously. Copying warping fields...', verbose, 'warning')
        shutil.copy('../warp_curve2straight.nii.gz', 'warp_curve2straight.nii.gz')
        shutil.copy('../warp_straight2curve.nii.gz', 'warp_straight2curve.nii.gz')
        shutil.copy('../straight_ref.nii.gz', 'straight_ref.nii.gz')
        # apply straightening
        sct.run('sct_apply_transfo -i anat_rpi.nii -w warp_curve2straight.nii.gz -d straight_ref.nii.gz -o anat_rpi_straight.nii -x spline', verbose)
    else:
        sct.run('sct_straighten_spinalcord -i anat_rpi.nii -s centerline_rpi.nii -qc 0 -x spline', verbose)

    # Smooth the straightened image along z
    print '\nSmooth the straightened image along z...'
    sct.run('sct_maths -i anat_rpi_straight.nii -smooth 0,0,'+str(sigma)+' -o anat_rpi_straight_smooth.nii', verbose)

    # Apply the reversed warping field to get back the curved spinal cord
    print '\nApply the reversed warping field to get back the curved spinal cord...'
    sct.run('sct_apply_transfo -i anat_rpi_straight_smooth.nii -o anat_rpi_straight_smooth_curved.nii -d anat.nii -w warp_straight2curve.nii.gz -x spline', verbose)

    # replace zeroed voxels by original image (issue #937)
    sct.printv('\nReplace zeroed voxels by original image...', verbose)
    nii_smooth = Image('anat_rpi_straight_smooth_curved.nii')
    data_smooth = nii_smooth.data
    data_input = Image('anat.nii').data
    indzero = np.where(data_smooth == 0)
    data_smooth[indzero] = data_input[indzero]
    nii_smooth.data = data_smooth
    nii_smooth.setFileName('anat_rpi_straight_smooth_curved_nonzero.nii')
    nii_smooth.save()

    # come back to parent folder
    os.chdir('..')

    # Generate output file
    print '\nGenerate output file...'
    sct.generate_output_file(path_tmp+'/anat_rpi_straight_smooth_curved_nonzero.nii', file_anat+'_smooth'+ext_anat)

    # Remove temporary files
    if remove_temp_files == 1:
        print('\nRemove temporary files...')
        shutil.rmtree(path_tmp, ignore_errors=True)

    # Display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s\n'

    # to view results
    sct.printv('Done! To view results, type:', verbose)
    sct.printv('fslview '+file_anat+' '+file_anat+'_smooth &\n', verbose, 'info')


if __name__ == "__main__":
    main()
