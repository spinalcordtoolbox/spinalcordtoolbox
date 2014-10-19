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
from sct_orientation import get_orientation, set_orientation


class param:
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
    sct.run('sct_c3d '+fname_anat+' -o '+path_tmp+'/anat.nii')
    sct.run('sct_c3d '+fname_centerline+' -o '+path_tmp+'/centerline.nii')

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of the input image into RPI
    print '\nOrient input volume to RPI orientation...'
    set_orientation('anat.nii', 'RPI', 'anat_rpi.nii')
    # Change orientation of the input image into RPI
    print '\nOrient centerline to RPI orientation...'
    set_orientation('centerline.nii', 'RPI', 'centerline_rpi.nii')

    # Straighten the spinal cord
    print '\nStraighten the spinal cord...'
    sct.run('sct_straighten_spinalcord -i anat_rpi.nii -c centerline_rpi.nii -w spline -v '+str(verbose))

    # Smooth the straightened image along z
    print '\nSmooth the straightened image along z...'
    sct.run('sct_c3d anat_rpi_straight.nii -smooth 0x0x'+str(sigma)+'vox -o anat_rpi_straight_smooth.nii')

    # Apply the reversed warping field to get back the curved spinal cord
    print '\nApply the reversed warping field to get back the curved spinal cord...'
    sct.run('sct_apply_transfo -i anat_rpi_straight_smooth.nii -o anat_rpi_straight_smooth_curved.nii -d anat.nii -w warp_straight2curve.nii.gz -p spline')

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
    print 'To view results, type:'
    print 'fslview '+file_anat+' '+file_anat+'_smooth &\n'

    # End of Main
    

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
        '  -c <centerline>   spinal cord centerline (given by the function sct_get_centerline) or segmentation.\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -s                sigma of the smoothing Gaussian kernel (in voxel). Default=3.' \
        '  -r {0,1}          remove temporary files. Default='+str(param.remove_temp_files)+'\n' \
        '  -v {0,1,2}        verbose. 0: nothing, 1: txt, 2: txt+fig. Default='+str(param.verbose)+'\n' \
        '  -h                help. Show this message.\n' \
        '\n'

    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = param()
    main()