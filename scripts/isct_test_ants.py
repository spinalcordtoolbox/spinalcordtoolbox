#!/usr/bin/env python
#########################################################################################
#
# This function test the integrity of ANTs output, given that some versions of ANTs give a wrong BSpline transform,
# notably when using sct_ANTSUseLandmarkImagesToGetBSplineDisplacementField.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-07-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys

import numpy as np

import nibabel as nib

import getopt
import os
import sct_utils as sct


# main
#=======================================================================================================================
def main():

    # Initialization
    size_data = 61
    size_label = 1  # put zero for labels that are single points.
    dice_acceptable = 0.39  # computed DICE should be 0.931034
    test_passed = 0
    remove_temp_files = 1
    verbose = 1

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hvr:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-v'):
            verbose = int(arg)
        elif opt in ('-r'):
            remove_temp_files = int(arg)

    # create temporary folder
    path_tmp = sct.tmp_create(verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # Initialise numpy volumes
    data_src = np.zeros((size_data, size_data, size_data), dtype=np.int16)
    data_dest = np.zeros((size_data, size_data, size_data), dtype=np.int16)

    # add labels for src image (curved).
    # Labels can be big (more than single point), because when applying NN interpolation, single points might disappear
    data_src[20 - size_label:20 + size_label + 1, 20 - size_label:20 + size_label + 1, 10 - size_label:10 + size_label + 1] = 1
    data_src[30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1] = 2
    data_src[20 - size_label:20 + size_label + 1, 20 - size_label:20 + size_label + 1, 50 - size_label:50 + size_label + 1] = 3

    # add labels for dest image (straight).
    # Here, no need for big labels (bigger than single point) because these labels will not be re-interpolated.
    data_dest[30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1, 10 - size_label:10 + size_label + 1] = 1
    data_dest[30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1] = 2
    data_dest[30 - size_label:30 + size_label + 1, 30 - size_label:30 + size_label + 1, 50 - size_label:50 + size_label + 1] = 3

    # save as nifti
    img_src = nib.Nifti1Image(data_src, np.eye(4))
    nib.save(img_src, 'data_src.nii.gz')
    img_dest = nib.Nifti1Image(data_dest, np.eye(4))
    nib.save(img_dest, 'data_dest.nii.gz')

    # Estimate rigid transformation
    printv('\nEstimate rigid transformation between paired landmarks...', verbose)
    # sct.run('isct_ANTSLandmarksBSplineTransform data_dest.nii.gz data_src.nii.gz rigid curve2straight_rigid.txt', verbose)
    sct.run('isct_antsRegistration -d 3 -t syn[1,3,1] -m MeanSquares[data_dest.nii.gz,data_src.nii.gz,1,3] -f 2 -s 0 -o [src2reg, data_src_reg.nii.gz] -c 5 -v 1 -n NearestNeighbor', verbose)

    # # Apply rigid transformation
    # printv('\nApply rigid transformation to curved landmarks...', verbose)
    # sct.run('sct_apply_transfo -i data_src.nii.gz -o data_src_rigid.nii.gz -d data_dest.nii.gz -w curve2straight_rigid.txt -p nn', verbose)
    #
    # # Estimate b-spline transformation curve --> straight
    # printv('\nEstimate b-spline transformation: curve --> straight...', verbose)
    # sct.run('isct_ANTSLandmarksBSplineTransform data_dest.nii.gz data_src_rigid.nii.gz warp_curve2straight_intermediate.nii.gz 5x5x5 3 2 0', verbose)
    #
    # # Concatenate rigid and non-linear transformations...
    # printv('\nConcatenate rigid and non-linear transformations...', verbose)
    # cmd = 'isct_ComposeMultiTransform 3 warp_curve2straight.nii.gz -R data_dest.nii.gz warp_curve2straight_intermediate.nii.gz curve2straight_rigid.txt'
    # printv('>> '+cmd, verbose)
    # commands.getstatusoutput(cmd)
    #
    # # Apply deformation to input image
    # printv('\nApply transformation to input image...', verbose)
    # sct.run('sct_apply_transfo -i data_src.nii.gz -o data_src_warp.nii.gz -d data_dest.nii.gz -w warp_curve2straight.nii.gz -p nn', verbose)
    #
    # Compute DICE coefficient between src and dest
    printv('\nCompute DICE coefficient...', verbose)
    sct.run('sct_dice_coefficient -i data_dest.nii.gz -d data_src_reg.nii.gz -o dice.txt', verbose)
    with open("dice.txt", "r") as file_dice:
        dice = float(file_dice.read().replace('3D Dice coefficient = ', ''))
    printv('Dice coeff = ' + str(dice) + ' (should be above ' + str(dice_acceptable) + ')', verbose)

    # Check if DICE coefficient is above acceptable value
    if dice > dice_acceptable:
        test_passed = 1

    # come back to parent folder
    os.chdir('..')

    # Delete temporary files
    if remove_temp_files == 1:
        printv('\nDelete temporary files...', verbose)
        sct.run('rm -rf ' + path_tmp, verbose)

    # output result for parent function
    if test_passed:
        printv('\nTest passed!\n', verbose)
        sys.exit(0)
    else:
        printv('\nTest failed!\n', verbose)
        sys.exit(1)


# printv: enables to sct.printv(or not, depending on verbose status)
#=======================================================================================================================
def printv(string, verbose):
    if verbose:
        sct.log.info(string)


# sct.printv(usage)
# ==========================================================================================
def usage():
    print('\n' \
        '' + os.path.basename(__file__) + '\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  This function test the integrity of ANTs output, given that some versions of ANTs give a wrong BSpline ' \
        '  transform notably when using sct_ANTSUseLandmarkImagesToGetBSplineDisplacementField..\n' \
        '\n' \
        'USAGE\n' \
        '  ' + os.path.basename(__file__) + '\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -h                         show this help\n' \
        '  -r {0, 1}                  remove temp files. Default=1\n' \
        '  -v {0, 1}                  verbose. Default=1\n' \
        '\n')

    # exit program
    sys.exit(2)


# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # call main function
    main()
