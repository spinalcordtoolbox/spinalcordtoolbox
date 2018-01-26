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


import sys, io, os, getopt, shutil, time

import numpy as np

import sct_utils as sct
from sct_image import set_orientation
from sct_convert import convert
from msct_parser import Parser


# PARSER
# ==========================================================================================
def get_parser():
    # Initialize the parser
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


# MAIN
# ==========================================================================================
def main(args=None):

    # Initialization
    # fname_anat = ''
    # fname_centerline = ''
    sigma = 3  # default value of the standard deviation for the Gaussian smoothing (in terms of number of voxels)
    # remove_temp_files = param.remove_temp_files
    # verbose = param.verbose
    start_time = time.time()

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    fname_anat = arguments['-i']
    fname_centerline = arguments['-s']
    if '-smooth' in arguments:
        sigma = arguments['-smooth']
    if '-r' in arguments:
        remove_temp_files = int(arguments['-r'])
    if '-v' in arguments:
        verbose = int(arguments['-v'])

    # Display arguments
    sct.printv('\nCheck input arguments...')
    sct.printv('  Volume to smooth .................. ' + fname_anat)
    sct.printv('  Centerline ........................ ' + fname_centerline)
    sct.printv('  Sigma (mm) ........................ ' + str(sigma))
    sct.printv('  Verbose ........................... ' + str(verbose))

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
                   'sct_image -i ' + fname_anat + ' -split t -o ' + fname_anat, verbose, 'warning')
        sct.printv('4D images not supported, aborting ...', verbose, 'error')

    # Extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    path_centerline, file_centerline, ext_centerline = sct.extract_fname(fname_centerline)

    path_tmp = sct.tmp_create(basename="smooth_spinalcord", verbose=verbose)

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    sct.copy(fname_anat, os.path.join(path_tmp, "anat" + ext_anat))
    sct.copy(fname_centerline, os.path.join(path_tmp, "centerline" + ext_centerline))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # convert to nii format
    convert('anat' + ext_anat, 'anat.nii')
    convert('centerline' + ext_centerline, 'centerline.nii')

    # Change orientation of the input image into RPI
    sct.printv('\nOrient input volume to RPI orientation...')
    fname_anat_rpi = set_orientation('anat.nii', 'RPI', filename=True)
    shutil.move(fname_anat_rpi, 'anat_rpi.nii')
    # Change orientation of the input image into RPI
    sct.printv('\nOrient centerline to RPI orientation...')
    fname_centerline_rpi = set_orientation('centerline.nii', 'RPI', filename=True)
    shutil.move(fname_centerline_rpi, 'centerline_rpi.nii')

    # Straighten the spinal cord
    # straighten segmentation
    sct.printv('\nStraighten the spinal cord using centerline/segmentation...', verbose)
    # check if warp_curve2straight and warp_straight2curve already exist (i.e. no need to do it another time)
    if os.path.isfile(os.path.join(curdir, 'warp_curve2straight.nii.gz')) and os.path.isfile(os.path.join(curdir, 'warp_straight2curve.nii.gz')) and os.path.isfile(os.path.join(curdir, 'straight_ref.nii.gz')):
        # if they exist, copy them into current folder
        sct.printv('WARNING: Straightening was already run previously. Copying warping fields...', verbose, 'warning')
        sct.copy(os.path.join(curdir, 'warp_curve2straight.nii.gz'), 'warp_curve2straight.nii.gz')
        sct.copy(os.path.join(curdir, 'warp_straight2curve.nii.gz'), 'warp_straight2curve.nii.gz')
        sct.copy(os.path.join(curdir, 'straight_ref.nii.gz'), 'straight_ref.nii.gz')
        # apply straightening
        sct.run('sct_apply_transfo -i anat_rpi.nii -w warp_curve2straight.nii.gz -d straight_ref.nii.gz -o anat_rpi_straight.nii -x spline', verbose)
    else:
        sct.run('sct_straighten_spinalcord -i anat_rpi.nii -s centerline_rpi.nii -qc 0 -x spline', verbose)

    # Smooth the straightened image along z
    sct.printv('\nSmooth the straightened image along z...')
    sct.run('sct_maths -i anat_rpi_straight.nii -smooth 0,0,' + str(sigma) + ' -o anat_rpi_straight_smooth.nii', verbose)

    # Apply the reversed warping field to get back the curved spinal cord
    sct.printv('\nApply the reversed warping field to get back the curved spinal cord...')
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

    # come back
    os.chdir(curdir)

    # Generate output file
    sct.printv('\nGenerate output file...')
    sct.generate_output_file(os.path.join(path_tmp, "anat_rpi_straight_smooth_curved_nonzero.nii"),
                             file_anat + '_smooth' + ext_anat)

    # Remove temporary files
    if remove_temp_files == 1:
        sct.printv('\nRemove temporary files...')
        shutil.rmtree(path_tmp)

    # Display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(round(elapsed_time))) + 's\n')

    sct.display_viewer_syntax([file_anat, file_anat + '_smooth'], verbose=verbose)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # initialize parameters
    # param = Param()
    # call main function
    main()


# OLD CODE

# ## new
#
# ### Make sure that centerline file does not have halls
# file_c = load('centerline_rpi.nii')
# data_c = file_c.get_data()
# hdr_c = file_c.get_header()
#
# data_temp = copy(data_c)
# data_temp *= 0
# data_output = copy(data_c)
# data_output *= 0
# nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('centerline_rpi.nii')
#
# ## Change seg to centerline if it is a segmentation
# sct.printv('\nChange segmentation to centerline if it is a centerline...\n')
# z_centerline = [iz for iz in range(0, nz, 1) if data_c[:,:,iz].any() ]
# nz_nonz = len(z_centerline)
# if nz_nonz==0 :
#     sct.printv('\nERROR: Centerline is empty')
#     sys.exit()
# x_centerline = [0 for iz in range(0, nz_nonz, 1)]
# y_centerline = [0 for iz in range(0, nz_nonz, 1)]
# #sct.printv("z_centerline", z_centerline,nz_nonz,len(x_centerline)))
# sct.printv('\nGet center of mass of the centerline ...')
# for iz in range(len(z_centerline)):
#     x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(array(data_c[:,:,z_centerline[iz]]))
#     data_temp[x_centerline[iz], y_centerline[iz], z_centerline[iz]] = 1
#
# ## Complete centerline
# sct.printv('\nComplete the halls of the centerline if there are any...\n')
# X,Y,Z = data_temp.nonzero()
#
# x_centerline_extended = [0 for i in range(0, nz, 1)]
# y_centerline_extended = [0 for i in range(0, nz, 1)]
# for iz in range(len(Z)):
#     x_centerline_extended[Z[iz]] = X[iz]
#     y_centerline_extended[Z[iz]] = Y[iz]
#
# X_centerline_extended = nonzero(x_centerline_extended)
# X_centerline_extended = transpose(X_centerline_extended)
# Y_centerline_extended = nonzero(y_centerline_extended)
# Y_centerline_extended = transpose(Y_centerline_extended)
#
# # initialization: we set the extrem values to avoid edge effects
# x_centerline_extended[0] = x_centerline_extended[X_centerline_extended[0]]
# x_centerline_extended[-1] = x_centerline_extended[X_centerline_extended[-1]]
# y_centerline_extended[0] = y_centerline_extended[Y_centerline_extended[0]]
# y_centerline_extended[-1] = y_centerline_extended[Y_centerline_extended[-1]]
#
# # Add two rows to the vector X_means_smooth_extended:
# # one before as means_smooth_extended[0] is now diff from 0
# # one after as means_smooth_extended[-1] is now diff from 0
# X_centerline_extended = append(X_centerline_extended, len(x_centerline_extended)-1)
# X_centerline_extended = insert(X_centerline_extended, 0, 0)
# Y_centerline_extended = append(Y_centerline_extended, len(y_centerline_extended)-1)
# Y_centerline_extended = insert(Y_centerline_extended, 0, 0)
#
# #recurrence
# count_zeros_x=0
# count_zeros_y=0
# for i in range(1,nz-1):
#     if x_centerline_extended[i]==0:
#        x_centerline_extended[i] = 0.5*(x_centerline_extended[X_centerline_extended[i-1-count_zeros_x]] + x_centerline_extended[X_centerline_extended[i-count_zeros_x]])
#        count_zeros_x += 1
#     if y_centerline_extended[i]==0:
#        y_centerline_extended[i] = 0.5*(y_centerline_extended[Y_centerline_extended[i-1-count_zeros_y]] + y_centerline_extended[Y_centerline_extended[i-count_zeros_y]])
#        count_zeros_y += 1
#
# # Save image centerline completed to be used after
# sct.printv('\nSave image completed: centerline_rpi_completed.nii...\n')
# for i in range(nz):
#     data_output[x_centerline_extended[i],y_centerline_extended[i],i] = 1
# img = Nifti1Image(data_output, None, hdr_c)
# save(img, 'centerline_rpi_completed.nii')
#
# #end new
