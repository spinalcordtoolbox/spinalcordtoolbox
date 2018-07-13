import sys

import os
import sct_utils as sct

# printv: enables to sct.printv(or not, depending on verbose status)
#=======================================================================================================================
def printv(string, verbose):
    if verbose:
        sct.log.info(string)

def suf(verbose=1, remove_temp_files=1):
    dice_acceptable = 0.39  # computed DICE should be 0.931034
    test_passed = 0
    size_data = 61
    size_label = 1  # put zero for labels that are single points.

    path_tmp = sct.tmp_create(basename="test_ants", verbose=verbose)

    # go to tmp folder
    
    curdir = os.getcwd()
    
    os.chdir(path_tmp)

    import numpy as np

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
    import nibabel as nib

    img_src = nib.Nifti1Image(data_src, np.eye(4))
    nib.save(img_src, 'data_src.nii.gz')
    img_dest = nib.Nifti1Image(data_dest, np.eye(4))
    nib.save(img_dest, 'data_dest.nii.gz')


    # Estimate rigid transformation
    printv('\nEstimate rigid transformation between paired landmarks...', verbose)
    # TODO fixup isct_ants* parsers
    sct.run(['isct_antsRegistration',
     '-d', '3',
     '-t', 'syn[1,3,1]',
     '-m', 'MeanSquares[data_dest.nii.gz,data_src.nii.gz,1,3]',
     '-f', '2',
     '-s', '0',
     '-o', '[src2reg,data_src_reg.nii.gz]',
     '-c', '5',
     '-v', '1',
     '-n', 'NearestNeighbor'], verbose)

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
    # sct.run(cmd)
    #
    # # Apply deformation to input image
    # printv('\nApply transformation to input image...', verbose)
    # sct.run('sct_apply_transfo -i data_src.nii.gz -o data_src_warp.nii.gz -d data_dest.nii.gz -w warp_curve2straight.nii.gz -p nn', verbose)
    #
    # Compute DICE coefficient between src and dest
    printv('\nCompute DICE coefficient...', verbose)
    sct.run(["sct_dice_coefficient",
     "-i", "data_dest.nii.gz",
     "-d", "data_src_reg.nii.gz",
     "-o", "dice.txt"], verbose)
    
    with open("dice.txt", "r") as file_dice:
        dice = float(file_dice.read().replace('3D Dice coefficient = ', ''))
    printv('Dice coeff = ' + str(dice) + ' (should be above ' + str(dice_acceptable) + ')', verbose)

    # Check if DICE coefficient is above acceptable value
    if dice > dice_acceptable:
        test_passed = 1

    # come back
    os.chdir(curdir)

    # Delete temporary files
    if remove_temp_files == 1:
        printv('\nDelete temporary files...', verbose)
        sct.rmtree(path_tmp)

    # output result for parent function
    if test_passed:
        printv('\nTest passed!\n', verbose)
        sys.exit(0)
    else:
        printv('\nTest failed!\n', verbose)
        sys.exit(1)