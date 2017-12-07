#!/usr/bin/env python

## @package sct_get_centerline
#
# - run a gaussian-weighted slice-by-slice registration to roughtly estimate the spinal cord centerline.
# - fit the rought centerline to 3D spline function
#
#
# Description about how the function works:
#
# 1. slice-wise realignement
# ------------------------------------------------
# inputs:
# - spinal cord image
# - binary image with a point center of spinal cord.
#
# the algorithm iterates in the upwards and then downward direction (along Z). For each direction, it registers the slice i+1 on the slice i. It does that using a gaussian mask, centers on the spinal cord, and is applied on both the i and i+1 slices (inweight and refweight from flirt).
# NB: the code works on APRLIS
# output:
# - rough centerline
# - gaussian mask
# - transformation matrices Tx,Ty
#
# 2. smoothing of the centerline
# ------------------------------------------------
# input: centerline, transfo matrices
# it fits in 3d (using an independent decomposition of the planes XZ and YZ) a spline function of order 1.
# apply the smoothed transformation matrices to the image --> gives a "smoothly" straighted spinal cord (but stil with errors due to deformation of strucutre related to curvatyre)
# output:
# - centerline fitted
# - transformation matrices Tx,Ty smoothed
# - straighted spinal cord
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Geoffrey Leveque
#
# License: see the LICENSE.TXT
# ==========================================================================================

# TODO: remove FSL dependency
# TODO: apply previous transfo to mask (assuming spatial autocorrelation --> transformation is expected to be close to previous one)
# TODO: output data in short instead of float
# TODO: try to fit in 3D instead of 2D
# TODO: add option verbose for sct.run()
# TODO: include the schedule file in the python script to remove the dependence
# TODO: try applying N4 before
# TODO: 3d smooth of the input image (e.g., 3mm gaussian kernel) -> later
# TODO: subsample input data for faster processing (but might create problems of coordinate)-- alternatively, find appropriate schedule file without the 1mm at the end


import sys, io, os, time, getopt, glob

# import nibabel
from numpy import mgrid, zeros, exp, unravel_index, argmax, poly1d, polyval, linalg, max, polyfit, sqrt, abs, savetxt

import sct_utils as sct
from sct_utils import fsloutput
from sct_orientation import get_orientation, set_orientation
from sct_convert import convert
from msct_image import Image
from sct_split_data import split_data
from sct_concat_data import concat_data
from sct_copy_header import copy_header

# get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))

## Default parameters
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.schedule_file = 'flirtsch/schedule_TxTy.sch'
        self.gap = 4  # default gap between co-registered slices.
        self.gaussian_kernel = 4 # gaussian kernel for creating gaussian mask from center point.
        self.deg_poly = 10 # maximum degree of polynomial function for fitting centerline.
        self.remove_tmp_files = 1 # remove temporary files


#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization
    fname_anat = ''
    fname_point = ''
    slice_gap = param.gap
    remove_tmp_files = param.remove_tmp_files
    gaussian_kernel = param.gaussian_kernel
    start_time = time.time()
    verbose = 1

    # Parameters for debug mode
    if param.debug == 1:
        sct.printv('\n*** WARNING: DEBUG MODE ON ***\n\t\t\tCurrent working directory: '+os.getcwd(), 'warning')
        path_sct_data = os.environ.get("SCT_TESTING_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__))), "testing_data")
        fname_anat = os.path.join(path_sct_testing_data, "t2", "t2.nii.gz")
        fname_point = os.path.join(path_sct_testing_data, "t2", "t2_centerline_init.nii.gz")
        slice_gap = 5

    else:
        # Check input param
        try:
            opts, args = getopt.getopt(sys.argv[1:],'hi:p:g:r:k:')
        except getopt.GetoptError as err:
            print str(err)
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-i'):
                fname_anat = arg
            elif opt in ('-p'):
                fname_point = arg
            elif opt in ('-g'):
                slice_gap = int(arg)
            elif opt in ('-r'):
                remove_tmp_files = int(arg)
            elif opt in ('-k'):
                gaussian_kernel = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_anat == '' or fname_point == '':
        usage()

    # check existence of input files
    sct.check_file_exist(fname_anat)
    sct.check_file_exist(fname_point)

    # extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    path_point, file_point, ext_point = sct.extract_fname(fname_point)

    # extract path of schedule file
    # TODO: include schedule file in sct
    # TODO: check existence of schedule file
    file_schedule = os.path.join(path_sct, param.schedule_file)

    # Get input image orientation
    input_image_orientation = get_orientation(fname_anat)

    # Display arguments
    print '\nCheck input arguments...'
    print '  Anatomical image:     '+fname_anat
    print '  Orientation:          '+input_image_orientation
    print '  Point in spinal cord: '+fname_point
    print '  Slice gap:            '+str(slice_gap)
    print '  Gaussian kernel:      '+str(gaussian_kernel)
    print '  Degree of polynomial: '+str(param.deg_poly)

    path_tmp = sct.tmp_create(basename="get_centerline_automatic", verbose=verbose)

    print '\nCopy input data...'
    sct.run('cp '+fname_anat+ ' '+os.path.join(path_tmp, "tmp.anat"+ext_anat))
    sct.run('cp '+fname_point+ ' '+os.path.join(path_tmp, "tmp.point"+ext_point))

    # go to temporary folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # convert to nii
    convert('tmp.anat'+ext_anat, 'tmp.anat.nii')
    convert('tmp.point'+ext_point, 'tmp.point.nii')

    # Reorient input anatomical volume into RL PA IS orientation
    print '\nReorient input volume to RL PA IS orientation...'
    #sct.run(sct.fsloutput + 'fslswapdim tmp.anat RL PA IS tmp.anat_orient')
    set_orientation('tmp.anat.nii', 'RPI', 'tmp.anat_orient.nii')
    # Reorient binary point into RL PA IS orientation
    print '\nReorient binary point into RL PA IS orientation...'
    # sct.run(sct.fsloutput + 'fslswapdim tmp.point RL PA IS tmp.point_orient')
    set_orientation('tmp.point.nii', 'RPI', 'tmp.point_orient.nii')

    # Get image dimensions
    print '\nGet image dimensions...'
    nx, ny, nz, nt, px, py, pz, pt = Image('tmp.anat_orient.nii').dim
    print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
    print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'

    # Split input volume
    print '\nSplit input volume...'
    split_data('tmp.anat_orient.nii', 2, '_z')
    file_anat_split = ['tmp.anat_orient_z'+str(z).zfill(4) for z in range(0, nz, 1)]
    split_data('tmp.point_orient.nii', 2, '_z')
    file_point_split = ['tmp.point_orient_z'+str(z).zfill(4) for z in range(0, nz, 1)]

    # Extract coordinates of input point
    # sct.printv('\nExtract the slice corresponding to z='+str(z_init)+'...', verbose)
    #
    data_point = Image('tmp.point_orient.nii').data
    x_init, y_init, z_init = unravel_index(data_point.argmax(), data_point.shape)
    sct.printv('Coordinates of input point: ('+str(x_init)+', '+str(y_init)+', '+str(z_init)+')', verbose)

    # Create 2D gaussian mask
    sct.printv('\nCreate gaussian mask from point...', verbose)
    xx, yy = mgrid[:nx, :ny]
    mask2d = zeros((nx, ny))
    radius = round(float(gaussian_kernel+1)/2)  # add 1 because the radius includes the center.
    sigma = float(radius)
    mask2d = exp(-(((xx-x_init)**2)/(2*(sigma**2)) + ((yy-y_init)**2)/(2*(sigma**2))))

    # Save mask to 2d file
    file_mask_split = ['tmp.mask_orient_z'+str(z).zfill(4) for z in range(0,nz,1)]
    nii_mask2d = Image('tmp.anat_orient_z0000.nii')
    nii_mask2d.data = mask2d
    nii_mask2d.setFileName(file_mask_split[z_init]+'.nii')
    nii_mask2d.save()
    #
    # # Get the coordinates of the input point
    # print '\nGet the coordinates of the input point...'
    # data_point = Image('tmp.point_orient.nii').data
    # x_init, y_init, z_init = unravel_index(data_point.argmax(), data_point.shape)
    # print '('+str(x_init)+', '+str(y_init)+', '+str(z_init)+')'

    # x_init, y_init, z_init = (data > 0).nonzero()
    # x_init = x_init[0]
    # y_init = y_init[0]
    # z_init = z_init[0]
    # print '('+str(x_init)+', '+str(y_init)+', '+str(z_init)+')'
    #
    # numpy.unravel_index(a.argmax(), a.shape)
    #
    # file = nibabel.load('tmp.point_orient.nii')
    # data = file.get_data()
    # x_init, y_init, z_init = (data > 0).nonzero()
    # x_init = x_init[0]
    # y_init = y_init[0]
    # z_init = z_init[0]
    # print '('+str(x_init)+', '+str(y_init)+', '+str(z_init)+')'
    #
    # # Extract the slice corresponding to z=z_init
    # print '\nExtract the slice corresponding to z='+str(z_init)+'...'
    # file_point_split = ['tmp.point_orient_z'+str(z).zfill(4) for z in range(0,nz,1)]
    # nii = Image('tmp.point_orient.nii')
    # data_crop = nii.data[:, :, z_init:z_init+1]
    # nii.data = data_crop
    # nii.setFileName(file_point_split[z_init]+'.nii')
    # nii.save()
    #
    # # Create gaussian mask from point
    # print '\nCreate gaussian mask from point...'
    # file_mask_split = ['tmp.mask_orient_z'+str(z).zfill(4) for z in range(0,nz,1)]
    # sct.run(sct.fsloutput+'fslmaths '+file_point_split[z_init]+' -s '+str(gaussian_kernel)+' '+file_mask_split[z_init])
    #
    # # Obtain max value from mask
    # print '\nFind maximum value from mask...'
    # file = nibabel.load(file_mask_split[z_init]+'.nii')
    # data = file.get_data()
    # max_value_mask = numpy.max(data)
    # print '..'+str(max_value_mask)
    #
    # # Normalize mask beween 0 and 1
    # print '\nNormalize mask beween 0 and 1...'
    # sct.run(sct.fsloutput+'fslmaths '+file_mask_split[z_init]+' -div '+str(max_value_mask)+' '+file_mask_split[z_init])

    ## Take the square of the mask
    #print '\nCalculate the square of the mask...'
    #sct.run(sct.fsloutput+'fslmaths '+file_mask_split[z_init]+' -mul '+file_mask_split[z_init]+' '+file_mask_split[z_init])

    # initialize variables
    file_mat = ['tmp.mat_z'+str(z).zfill(4) for z in range(0,nz,1)]
    file_mat_inv = ['tmp.mat_inv_z'+str(z).zfill(4) for z in range(0,nz,1)]
    file_mat_inv_cumul = ['tmp.mat_inv_cumul_z'+str(z).zfill(4) for z in range(0,nz,1)]

    # create identity matrix for initial transformation matrix
    fid = open(file_mat_inv_cumul[z_init], 'w')
    fid.write('%i %i %i %i\n' %(1, 0, 0, 0) )
    fid.write('%i %i %i %i\n' %(0, 1, 0, 0) )
    fid.write('%i %i %i %i\n' %(0, 0, 1, 0) )
    fid.write('%i %i %i %i\n' %(0, 0, 0, 1) )
    fid.close()

    # initialize centerline: give value corresponding to initial point
    x_centerline = [x_init]
    y_centerline = [y_init]
    z_centerline = [z_init]
    warning_count = 0

    # go up (1), then down (2) in reference to the binary point
    for iUpDown in range(1, 3):

        if iUpDown == 1:
            # z increases
            slice_gap_signed = slice_gap
        elif iUpDown == 2:
            # z decreases
            slice_gap_signed = -slice_gap
            # reverse centerline (because values will be appended at the end)
            x_centerline.reverse()
            y_centerline.reverse()
            z_centerline.reverse()

        # initialization before looping
        z_dest = z_init # point given by user
        z_src = z_dest + slice_gap_signed

        # continue looping if 0 < z < nz
        while 0 <= z_src and z_src <= nz-1:

            # print current z:
            print 'z='+str(z_src)+':'

            # estimate transformation
            sct.run(fsloutput+'flirt -in '+file_anat_split[z_src]+' -ref '+file_anat_split[z_dest]+' -schedule '+file_schedule+ ' -verbose 0 -omat '+file_mat[z_src]+' -cost normcorr -forcescaling -inweight '+file_mask_split[z_dest]+' -refweight '+file_mask_split[z_dest])

            # display transfo
            status, output = sct.run('cat '+file_mat[z_src])
            print output

            # check if transformation is bigger than 1.5x slice_gap
            tx = float(output.split()[3])
            ty = float(output.split()[7])
            norm_txy = linalg.norm([tx, ty],ord=2)
            if norm_txy > 1.5*slice_gap:
                print 'WARNING: Transformation is too large --> using previous one.'
                warning_count = warning_count + 1
                # if previous transformation exists, replace current one with previous one
                if os.path.isfile(file_mat[z_dest]):
                    sct.run('cp '+file_mat[z_dest]+' '+file_mat[z_src])

            # estimate inverse transformation matrix
            sct.run('convert_xfm -omat '+file_mat_inv[z_src]+' -inverse '+file_mat[z_src])

            # compute cumulative transformation
            sct.run('convert_xfm -omat '+file_mat_inv_cumul[z_src]+' -concat '+file_mat_inv[z_src]+' '+file_mat_inv_cumul[z_dest])

            # apply inverse cumulative transformation to initial gaussian mask (to put it in src space)
            sct.run(fsloutput+'flirt -in '+file_mask_split[z_init]+' -ref '+file_mask_split[z_init]+' -applyxfm -init '+file_mat_inv_cumul[z_src]+' -out '+file_mask_split[z_src])

            # open inverse cumulative transformation file and generate centerline
            fid = open(file_mat_inv_cumul[z_src])
            mat = fid.read().split()
            x_centerline.append(x_init + float(mat[3]))
            y_centerline.append(y_init + float(mat[7]))
            z_centerline.append(z_src)
            #z_index = z_index+1

            # define new z_dest (target slice) and new z_src (moving slice)
            z_dest = z_dest + slice_gap_signed
            z_src = z_src + slice_gap_signed


    # Reconstruct centerline
    # ====================================================================================================

    # reverse back centerline (because it's been reversed once, so now all values are in the right order)
    x_centerline.reverse()
    y_centerline.reverse()
    z_centerline.reverse()

    # fit centerline in the Z-X plane using polynomial function
    print '\nFit centerline in the Z-X plane using polynomial function...'
    coeffsx = polyfit(z_centerline, x_centerline, deg=param.deg_poly)
    polyx = poly1d(coeffsx)
    x_centerline_fit = polyval(polyx, z_centerline)
    # calculate RMSE
    rmse = linalg.norm(x_centerline_fit-x_centerline)/sqrt( len(x_centerline) )
    # calculate max absolute error
    max_abs = max( abs(x_centerline_fit-x_centerline) )
    print '.. RMSE (in mm): '+str(rmse*px)
    print '.. Maximum absolute error (in mm): '+str(max_abs*px)

    # fit centerline in the Z-Y plane using polynomial function
    print '\nFit centerline in the Z-Y plane using polynomial function...'
    coeffsy = polyfit(z_centerline, y_centerline, deg=param.deg_poly)
    polyy = poly1d(coeffsy)
    y_centerline_fit = polyval(polyy, z_centerline)
    # calculate RMSE
    rmse = linalg.norm(y_centerline_fit-y_centerline)/sqrt( len(y_centerline) )
    # calculate max absolute error
    max_abs = max( abs(y_centerline_fit-y_centerline) )
    print '.. RMSE (in mm): '+str(rmse*py)
    print '.. Maximum absolute error (in mm): '+str(max_abs*py)

    # display
    if param.debug == 1:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(z_centerline,x_centerline,'.',z_centerline,x_centerline_fit,'r')
        plt.legend(['Data','Polynomial Fit'])
        plt.title('Z-X plane polynomial interpolation')
        plt.show()

        plt.figure()
        plt.plot(z_centerline,y_centerline,'.',z_centerline,y_centerline_fit,'r')
        plt.legend(['Data','Polynomial Fit'])
        plt.title('Z-Y plane polynomial interpolation')
        plt.show()

    # generate full range z-values for centerline
    z_centerline_full = [iz for iz in range(0, nz, 1)]

    # calculate X and Y values for the full centerline
    x_centerline_fit_full = polyval(polyx, z_centerline_full)
    y_centerline_fit_full = polyval(polyy, z_centerline_full)

    # Generate fitted transformation matrices and write centerline coordinates in text file
    print '\nGenerate fitted transformation matrices and write centerline coordinates in text file...'
    file_mat_inv_cumul_fit = ['tmp.mat_inv_cumul_fit_z'+str(z).zfill(4) for z in range(0,nz,1)]
    file_mat_cumul_fit = ['tmp.mat_cumul_fit_z'+str(z).zfill(4) for z in range(0,nz,1)]
    fid_centerline = open('tmp.centerline_coordinates.txt', 'w')
    for iz in range(0, nz, 1):
        # compute inverse cumulative fitted transformation matrix
        fid = open(file_mat_inv_cumul_fit[iz], 'w')
        fid.write('%i %i %i %f\n' %(1, 0, 0, x_centerline_fit_full[iz]-x_init) )
        fid.write('%i %i %i %f\n' %(0, 1, 0, y_centerline_fit_full[iz]-y_init) )
        fid.write('%i %i %i %i\n' %(0, 0, 1, 0) )
        fid.write('%i %i %i %i\n' %(0, 0, 0, 1) )
        fid.close()
        # compute forward cumulative fitted transformation matrix
        sct.run('convert_xfm -omat '+file_mat_cumul_fit[iz]+' -inverse '+file_mat_inv_cumul_fit[iz])
        # write centerline coordinates in x, y, z format
        fid_centerline.write('%f %f %f\n' %(x_centerline_fit_full[iz], y_centerline_fit_full[iz], z_centerline_full[iz]) )
    fid_centerline.close()


    # Prepare output data
    # ====================================================================================================

    # write centerline as text file
    for iz in range(0, nz, 1):
        # compute inverse cumulative fitted transformation matrix
        fid = open(file_mat_inv_cumul_fit[iz], 'w')
        fid.write('%i %i %i %f\n' %(1, 0, 0, x_centerline_fit_full[iz]-x_init) )
        fid.write('%i %i %i %f\n' %(0, 1, 0, y_centerline_fit_full[iz]-y_init) )
        fid.write('%i %i %i %i\n' %(0, 0, 1, 0) )
        fid.write('%i %i %i %i\n' %(0, 0, 0, 1) )
        fid.close()

    # write polynomial coefficients
    savetxt('tmp.centerline_polycoeffs_x.txt',coeffsx)
    savetxt('tmp.centerline_polycoeffs_y.txt',coeffsy)

    # apply transformations to data
    print '\nApply fitted transformation matrices...'
    file_anat_split_fit = ['tmp.anat_orient_fit_z'+str(z).zfill(4) for z in range(0,nz,1)]
    file_mask_split_fit = ['tmp.mask_orient_fit_z'+str(z).zfill(4) for z in range(0,nz,1)]
    file_point_split_fit = ['tmp.point_orient_fit_z'+str(z).zfill(4) for z in range(0,nz,1)]
    for iz in range(0, nz, 1):
        # forward cumulative transformation to data
        sct.run(fsloutput+'flirt -in '+file_anat_split[iz]+' -ref '+file_anat_split[iz]+' -applyxfm -init '+file_mat_cumul_fit[iz]+' -out '+file_anat_split_fit[iz])
        # inverse cumulative transformation to mask
        sct.run(fsloutput+'flirt -in '+file_mask_split[z_init]+' -ref '+file_mask_split[z_init]+' -applyxfm -init '+file_mat_inv_cumul_fit[iz]+' -out '+file_mask_split_fit[iz])
        # inverse cumulative transformation to point
        sct.run(fsloutput+'flirt -in '+file_point_split[z_init]+' -ref '+file_point_split[z_init]+' -applyxfm -init '+file_mat_inv_cumul_fit[iz]+' -out '+file_point_split_fit[iz]+' -interp nearestneighbour')

    # Merge into 4D volume
    print '\nMerge into 4D volume...'
    # sct.run(fsloutput+'fslmerge -z tmp.anat_orient_fit tmp.anat_orient_fit_z*')
    # sct.run(fsloutput+'fslmerge -z tmp.mask_orient_fit tmp.mask_orient_fit_z*')
    # sct.run(fsloutput+'fslmerge -z tmp.point_orient_fit tmp.point_orient_fit_z*')
    concat_data(glob.glob('tmp.anat_orient_fit_z*.nii'), 'tmp.anat_orient_fit.nii', dim=2)
    concat_data(glob.glob('tmp.mask_orient_fit_z*.nii'), 'tmp.mask_orient_fit.nii', dim=2)
    concat_data(glob.glob('tmp.point_orient_fit_z*.nii'), 'tmp.point_orient_fit.nii', dim=2)

    # Copy header geometry from input data
    print '\nCopy header geometry from input data...'
    copy_header('tmp.anat_orient.nii', 'tmp.anat_orient_fit.nii')
    copy_header('tmp.anat_orient.nii', 'tmp.mask_orient_fit.nii')
    copy_header('tmp.anat_orient.nii', 'tmp.point_orient_fit.nii')

    # Reorient outputs into the initial orientation of the input image
    print '\nReorient the centerline into the initial orientation of the input image...'
    set_orientation('tmp.point_orient_fit.nii', input_image_orientation, 'tmp.point_orient_fit.nii')
    set_orientation('tmp.mask_orient_fit.nii', input_image_orientation, 'tmp.mask_orient_fit.nii')

    # Generate output file (in current folder)
    print '\nGenerate output file (in current folder)...'

    os.chdir(curdir)  # come back

    #sct.generate_output_file('tmp.centerline_polycoeffs_x.txt','./','centerline_polycoeffs_x','.txt')
    #sct.generate_output_file('tmp.centerline_polycoeffs_y.txt','./','centerline_polycoeffs_y','.txt')
    #sct.generate_output_file('tmp.centerline_coordinates.txt','./','centerline_coordinates','.txt')
    #sct.generate_output_file('tmp.anat_orient.nii','./',file_anat+'_rpi',ext_anat)
    #sct.generate_output_file('tmp.anat_orient_fit.nii', file_anat+'_rpi_align'+ext_anat)
    #sct.generate_output_file('tmp.mask_orient_fit.nii', file_anat+'_mask'+ext_anat)
    fname_output_centerline = sct.generate_output_file(os.path.join(path_tmp, 'tmp.point_orient_fit.nii'), file_anat+'_centerline'+ext_anat)

    # Delete temporary files
    if remove_tmp_files == 1:
        print '\nRemove temporary files...'
        sct.run('rm -rf '+path_tmp)

    # print number of warnings
    print '\nNumber of warnings: '+str(warning_count)+' (if >10, you should probably reduce the gap and/or increase the kernel size'

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! \n\tGenerated file: '+fname_output_centerline+'\n\tElapsed time: '+str(int(round(elapsed_time)))+'s\n'


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  This program estimates the spinal cord centerline from an anatomical image and from another image
  pointing anywhere in the spinal cord. It works by registering adjacent slices together (2DOF
  transformation). The transformation is constrained by a gaussian mask. After registering all
  slices, the centerline is approximated by a polynomial function to smooth it.
  N.B. The centerline is usually not well estimated with this method. For better results, use
  sct_propseg.

USAGE
  """+os.path.basename(__file__)+""" -i <anat> -p <point>

MANDATORY ARGUMENTS
  -i <anat>        anatomic nifti file. Image to extract centerline from.
  -p <point>       binary nifti file. Point in the spinal cord (e.g., created using fslview)

OPTIONAL ARGUMENTS
  -g <gap>         gap between slices for registration. Higher is faster but less robust. Default="""+str(param_default.gap)+"""
  -k <kernel>      kernel size for gaussian mask. Higher is more robust but less accurate. Default="""+str(param_default.gaussian_kernel)+"""
  -r {0,1}         remove temporary files. Default="""+str(param_default.remove_tmp_files)+"""
  -h               help. Show this message.

EXAMPLE
  """+os.path.basename(__file__)+""" -i t2.nii.gz -p point_in_cord.nii.gz\n"""
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()

