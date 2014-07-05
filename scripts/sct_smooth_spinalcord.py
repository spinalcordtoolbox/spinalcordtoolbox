#!/usr/bin/env python

## @package sct_smooth_spinal_cord_straightening.py
#
# - spinal cord MRI volume 3D (as nifti format)
# - centerline of the spinal cord (given by sct_get_centerline.py)
# - sct_straighten_spinalcord.py from the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
#
#
# Description about how the function works:
# This program straightens the spinal cord of an anatomic image, apply a smoothing in the z dimension and apply
# the inverse warping field to get back the curved spinal cord but smoothed.
#
#
# USAGE
# ---------------------------------------------------------------------------------------
# sct_smooth_spinal_cord_straightening.py -i <input_image> -c <centerline>
#
# MANDATORY ARGUMENTS
# ---------------------------------------------------------------------------------------
#   -i       input 3D image to smooth.
#   -c       centerline (generated with sct_get_centerline).
#
#
# OPTIONAL ARGUMENTS
# ---------------------------------------------------------------------------------------
#   -r       if 1, remove temporary files (straightening output, smoothing output,
#            warping fields, outputs from orientation changes), default=1.
#
#
# EXAMPLES
# ---------------------------------------------------------------------------------------
#   sct_smooth_spinal_cord_straightening.py -i t2.nii.gz -c centerline.nii.gz
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
#
#
# EXTERNAL SOFTWARE
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymt.ca>
# Author: Simon LEVY
# Modified: 2014-07-05
#
# License: see the LICENSE.TXT
#=======================================================================================================================


import getopt
import os
import sys
import time
import sct_utils as sct



#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization
    fname_anat = ''
    fname_centerline = ''
    sigma = 3 # default value of the standard deviation for the Gaussian smoothing (in terms of number of voxels)
    remove_temp_files = 1
    start_time = time.time()


    # Check input param
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:c:r:s:')
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

    # Display usage if a mandatory argument is not provided
    if fname_anat == '' or fname_centerline == '':
        usage()

    # Check existence of input files
    sct.check_file_exist(fname_anat)
    sct.check_file_exist(fname_centerline)

    # Display arguments
    print '\nCheck input arguments...'
    print '.. Anatomical image:           ' + fname_anat
    print '.. Centerline:                 ' + fname_centerline

    # Extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    path_centerline, file_centerline, ext_centerline = sct.extract_fname(fname_centerline)

    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)

    # copy files into tmp folder
    sct.run('cp '+fname_anat+' '+path_tmp)
    sct.run('cp '+fname_centerline+' '+path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)

    # Change orientation of the input image into RPI
    print '\nOrient input volume to RPI orientation...'
    fname_anat_orient = path_anat+ file_anat+'_rpi'+ ext_anat
    sct.run('sct_orientation -i ' + file_anat + ext_anat + ' -o ' + fname_anat_orient + ' -orientation RPI')
    # Change orientation of the input image into RPI
    print '\nOrient centerline to RPI orientation...'
    fname_centerline_orient = path_centerline+file_centerline+'_rpi'+ ext_centerline
    sct.run('sct_orientation -i ' + file_centerline + ext_centerline + ' -o ' + fname_centerline_orient + ' -orientation RPI')

    # Straighten the spinal cord
    print '\nStraighten the spinal cord...'
    sct.run('sct_straighten_spinalcord.py -i '+fname_anat_orient+' -c '+fname_centerline_orient+' -w --use-BSpline')
    fname_straightening_output = file_anat+'_rpi_straight'+ext_anat

    # Smooth the straightened image along z
    print '\nSmooth the straightened image along z...'
    fname_smoothing_output = file_anat+'_z_smoothed'+ext_anat
    sct.run('c3d '+fname_straightening_output+' -smooth 0x0x'+str(sigma)+'vox -o '+fname_smoothing_output)

    # Apply the reversed warping field to get back the curved spinal cord
    print '\nApply the reversed warping field to get back the curved spinal cord (assuming a 3D image)...'
    fname_destraightened_output = file_anat+'_destraight_smoothed'+ext_anat
    sct.run('WarpImageMultiTransform 3 '+fname_smoothing_output+' '+fname_destraightened_output+' -R '+fname_anat+
            ' --use-BSpline warp_straight2curve.nii.gz')

    # come back to parent folder
    os.chdir('..')

    # Generate output file
    print '\nGenerate output file...'
    sct.generate_output_file(path_tmp+'/'+fname_destraightened_output,'',file_anat+'_smooth',ext_anat)

    # Remove temporary files
    if remove_temp_files == 1:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)

    #Display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s\n'

    # to view results
    print 'To view results, type:'
    print 'fslview '+file_anat+'_smooth &\n'

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
        '  sct_smooth_spinalcord.py -i <input_image> -c <centerline/segmentation>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <anat>         input image to smooth.\n' \
        '  -c <centerline>   spinal cord centerline (given by the function sct_get_centerline.py) or segmentation.\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -h                help. Show this message.\n' \
        '  -r {0,1}          remove temporary files. Default=1\n' \
        '  -s                sigma of the smoothing Gaussian kernel (expressed in voxels). Default=3.' \
        '\n'

    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()