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


import getopt
import os
import sys
import time

import sct_utils as sct
from sct_orientation import set_orientation
from numpy import append, insert, nonzero, transpose, array
from nibabel import load, Nifti1Image, save
from scipy import ndimage
from copy import copy

class Param:
    ## The constructor
    def __init__(self):
        self.remove_temp_files = 1 # remove temporary files
        self.verbose = 1


#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization
    fname_anat = ''
    fname_centerline = ''
    sigma = 3 # default value of the standard deviation for the Gaussian smoothing (in terms of number of voxels)
    remove_temp_files = param.remove_temp_files
    verbose = param.verbose
    start_time = time.time()


    # Check input param
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:c:r:s:v:')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    if not opts:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-c'):
            fname_centerline = arg
        elif opt in ('-i'):
            fname_anat = arg
        elif opt in ('-r'):
            remove_temp_files = arg
        elif opt in ('-s'):
            sigma = arg
        elif opt in ('-v'):
            verbose = int(arg)

    # Display usage if a mandatory argument is not provided
    if fname_anat == '' or fname_centerline == '':
        usage()

    # Display arguments
    print '\nCheck input arguments...'
    print '  Volume to smooth .................. ' + fname_anat
    print '  Centerline ........................ ' + fname_centerline
    print '  FWHM .............................. '+str(sigma)
    print '  Verbose ........................... '+str(verbose)

    # Check existence of input files
    print('\nCheck existence of input files...')
    sct.check_file_exist(fname_anat)
    sct.check_file_exist(fname_centerline)

    # Extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    path_centerline, file_centerline, ext_centerline = sct.extract_fname(fname_centerline)

    # create temporary folder
    print('\nCreate temporary folder...')
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)

    # copy files to temporary folder
    print('\nCopy files...')
    sct.run('isct_c3d '+fname_anat+' -o '+path_tmp+'/anat.nii')
    sct.run('isct_c3d '+fname_centerline+' -o '+path_tmp+'/centerline.nii')

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of the input image into RPI
    print '\nOrient input volume to RPI orientation...'
    set_orientation('anat.nii', 'RPI', 'anat_rpi.nii')
    # Change orientation of the input image into RPI
    print '\nOrient centerline to RPI orientation...'
    set_orientation('centerline.nii', 'RPI', 'centerline_rpi.nii')


    ## new

    ### Make sure that centerline file does not have halls
    file_c = load('centerline_rpi.nii')
    data_c = file_c.get_data()
    hdr_c = file_c.get_header()

    data_temp = copy(data_c)
    data_temp *= 0
    data_output = copy(data_c)
    data_output *= 0
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('centerline_rpi.nii')

    ## Change seg to centerline if it is a segmentation
    sct.printv('\nChange segmentation to centerline if it is a centerline...\n')
    z_centerline = [iz for iz in range(0, nz, 1) if data_c[:,:,iz].any() ]
    nz_nonz = len(z_centerline)
    if nz_nonz==0 :
        print '\nERROR: Centerline is empty'
        sys.exit()
    x_centerline = [0 for iz in range(0, nz_nonz, 1)]
    y_centerline = [0 for iz in range(0, nz_nonz, 1)]
    #print("z_centerline", z_centerline,nz_nonz,len(x_centerline))
    print '\nGet center of mass of the centerline ...'
    for iz in xrange(len(z_centerline)):
        x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(array(data_c[:,:,z_centerline[iz]]))
        data_temp[x_centerline[iz], y_centerline[iz], z_centerline[iz]] = 1

    ## Complete centerline
    sct.printv('\nComplete the halls of the centerline if there are any...\n')
    X,Y,Z = data_temp.nonzero()

    x_centerline_extended = [0 for i in range(0, nz, 1)]
    y_centerline_extended = [0 for i in range(0, nz, 1)]
    for iz in range(len(Z)):
        x_centerline_extended[Z[iz]] = X[iz]
        y_centerline_extended[Z[iz]] = Y[iz]

    X_centerline_extended = nonzero(x_centerline_extended)
    X_centerline_extended = transpose(X_centerline_extended)
    Y_centerline_extended = nonzero(y_centerline_extended)
    Y_centerline_extended = transpose(Y_centerline_extended)

    # initialization: we set the extrem values to avoid edge effects
    x_centerline_extended[0] = x_centerline_extended[X_centerline_extended[0]]
    x_centerline_extended[-1] = x_centerline_extended[X_centerline_extended[-1]]
    y_centerline_extended[0] = y_centerline_extended[Y_centerline_extended[0]]
    y_centerline_extended[-1] = y_centerline_extended[Y_centerline_extended[-1]]

    # Add two rows to the vector X_means_smooth_extended:
    # one before as means_smooth_extended[0] is now diff from 0
    # one after as means_smooth_extended[-1] is now diff from 0
    X_centerline_extended = append(X_centerline_extended, len(x_centerline_extended)-1)
    X_centerline_extended = insert(X_centerline_extended, 0, 0)
    Y_centerline_extended = append(Y_centerline_extended, len(y_centerline_extended)-1)
    Y_centerline_extended = insert(Y_centerline_extended, 0, 0)

    #recurrence
    count_zeros_x=0
    count_zeros_y=0
    for i in range(1,nz-1):
        if x_centerline_extended[i]==0:
           x_centerline_extended[i] = 0.5*(x_centerline_extended[X_centerline_extended[i-1-count_zeros_x]] + x_centerline_extended[X_centerline_extended[i-count_zeros_x]])
           count_zeros_x += 1
        if y_centerline_extended[i]==0:
           y_centerline_extended[i] = 0.5*(y_centerline_extended[Y_centerline_extended[i-1-count_zeros_y]] + y_centerline_extended[Y_centerline_extended[i-count_zeros_y]])
           count_zeros_y += 1

    # Save image centerline completed to be used after
    sct.printv('\nSave image completed: centerline_rpi_completed.nii...\n')
    for i in range(nz):
        data_output[x_centerline_extended[i],y_centerline_extended[i],i] = 1
    img = Nifti1Image(data_output, None, hdr_c)
    save(img, 'centerline_rpi_completed.nii')

    #end new


   # Straighten the spinal cord
    print '\nStraighten the spinal cord...'
    sct.run('sct_straighten_spinalcord -i anat_rpi.nii -c centerline_rpi_completed.nii -x spline -v '+str(verbose))

    # Smooth the straightened image along z
    print '\nSmooth the straightened image along z...'
    sct.run('isct_c3d anat_rpi_straight.nii -smooth 0x0x'+str(sigma)+'vox -o anat_rpi_straight_smooth.nii', verbose)

    # Apply the reversed warping field to get back the curved spinal cord
    print '\nApply the reversed warping field to get back the curved spinal cord...'
    sct.run('sct_apply_transfo -i anat_rpi_straight_smooth.nii -o anat_rpi_straight_smooth_curved.nii -d anat.nii -w warp_straight2curve.nii.gz -x spline', verbose)

    # come back to parent folder
    os.chdir('..')

    # Generate output file
    print '\nGenerate output file...'
    sct.generate_output_file(path_tmp+'/anat_rpi_straight_smooth_curved.nii', file_anat+'_smooth'+ext_anat)

    # Remove temporary files
    if remove_temp_files == 1:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)

    # Display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s\n'

    # to view results
    sct.printv('Done! To view results, type:', verbose)
    sct.printv('fslview '+file_anat+' '+file_anat+'_smooth &\n', verbose, 'info')



#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Smooth the spinal cord along its centerline. Steps are: 1) Spinal cord is straightened (using\n' \
        '  centerline), 2) a Gaussian kernel is applied in the superior-inferior direction, 3) then cord is\n' \
        '  de-straightened as originally.\n' \
        '\n' \
        'USAGE\n' \
        '  sct_smooth_spinalcord -i <image> -c <centerline/segmentation>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <image>        input image to smooth.\n' \
        '  -c <centerline>   spinal cord centerline or segmentation.\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -s                sigma of the smoothing Gaussian kernel (in voxel). Default=3.' \
        '  -r {0,1}          remove temporary files. Default='+str(param_default.remove_temp_files)+'\n' \
        '  -v {0,1,2}        verbose. 0: nothing, 1: small, 2: extended, 3: fig. Default='+str(param_default.verbose)+'\n' \
        '  -h                help. Show this message.\n' \
        '\n'

    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = Param()
    param_default = Param()
    main()