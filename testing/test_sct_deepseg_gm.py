#!/usr/bin/env python
#########################################################################################
#
# Test function sct_deepseg_gm
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sct_utils as sct
from msct_image import Image, compute_dice


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2s/t2s_uncropped.nii.gz -o t2s/t2s_uncropped_gmseg.nii.gz']
    param_test.dice_threshold = 0.9

    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    dice_segmentation = float('nan')
    # extract name of output segmentation: data_seg.nii.gz
    file_seg = os.path.join(param_test.path_output, sct.add_suffix(param_test.file_input, '_gmseg'))
    # open output segmentation
    im_seg = Image(file_seg)
    # open ground truth
    im_seg_manual = Image(param_test.fname_gt)
    # compute dice coefficient between generated image and image from database
    dice_segmentation = compute_dice(im_seg, im_seg_manual, mode='3d', zboundaries=False)
    # display
    param_test.output += 'Computed dice: '+str(dice_segmentation)
    param_test.output += '\nDice threshold (if computed Dice smaller: fail): '+str(param_test.dice_threshold)

    if dice_segmentation < param_test.dice_threshold:
        param_test.status = 99
        param_test.output += '\n--> FAILED'
    else:
        param_test.output += '\n--> PASSED'

    # update Panda structure
    param_test.results['dice_segmentation'] = dice_segmentation

    return param_test



    # initialization of results: must be NaN if test fails
    result_dice_gm, result_hausdorff, result_median_dist = float('nan'), float('nan'), float('nan')

    target_name = sct.extract_fname(param_test.file_input)[1]

    dice_fname = os.path.join(param_test.path_output, 'dice_coefficient_' + target_name + '.txt')
    hausdorff_fname = os.path.join(param_test.path_output, 'hausdorff_dist_' + target_name + '.txt')

    # Extracting dice results:
    dice = open(dice_fname, 'r')
    dice_lines = dice.readlines()
    dice.close()
    gm_start = dice_lines.index('Dice coefficient on the Gray Matter segmentation:\n')

    # extracting dice on GM
    gm_dice_lines = dice_lines[gm_start:wm_start - 1]
    gm_dice_lines = gm_dice_lines[gm_dice_lines.index('2D Dice coefficient by slice:\n') + 1:-1]

    null_slices = []
    gm_dice = []
    for line in gm_dice_lines:
        n_slice, dc = line.split(' ')
        # remove \n from dice result
        dc = dc[:-1]
        dc = dc[:-4] if '[0m' in dc else dc

        if dc == '0' or dc == 'nan':
            null_slices.append(n_slice)
        else:
            try:
                gm_dice.append(float(dc))
            except ValueError:
                gm_dice.append(float(dc[:-4]))
    result_dice_gm = mean(gm_dice)

    # extracting dice on WM
    wm_dice_lines = dice_lines[wm_start:]
    wm_dice_lines = wm_dice_lines[wm_dice_lines.index('2D Dice coefficient by slice:\n') + 1:]
    wm_dice = []
    for line in wm_dice_lines:
        n_slice, dc = line.split(' ')
        # remove \n from dice result
        if line is not wm_dice_lines[-1]:
            dc = dc[:-1]
        if n_slice not in null_slices:
            try:
                wm_dice.append(float(dc))
            except ValueError:
                wm_dice.append(float(dc[:-4]))
    result_dice_wm = mean(wm_dice)

    # Extracting hausdorff distance results
    hd = open(hausdorff_fname, 'r')
    hd_lines = hd.readlines()
    hd.close()

    # remove title of columns and last empty/non important lines
    hd_lines = hd_lines[1:-4]

    hausdorff = []
    max_med = []
    for line in hd_lines:
        slice_id, res = line.split(':')
        slice, n_slice = slice_id.split(' ')
        if n_slice not in null_slices:
            hd, med1, med2 = res[:-1].split(' - ')
            hd, med1, med2 = float(hd), float(med1), float(med2)
            hausdorff.append(hd)
            max_med.append(max(med1, med2))

    result_hausdorff = mean(hausdorff)
    result_median_dist = mean(max_med)

    # Integrity check
    if result_hausdorff > param_test.hd_threshold or result_dice_wm < param_test.wm_dice_threshold:
        param_test.status = 99
        param_test.output += '\nResulting segmentation is too different from manual segmentation:\n' \
                             'WM dice: ' + str(result_dice_wm) + '\n' \
                             'Hausdorff distance: ' + str(result_hausdorff) + '\n'

    # transform results into Pandas structure
    results = DataFrame(data={'status': param_test.status, 'output': param_test.output, 'dice_gm': result_dice_gm, 'dice_wm': result_dice_wm,
                              'hausdorff': result_hausdorff, 'med_dist': result_median_dist, 'duration_[s]': param_test.duration},
                        index=[param_test.path_data])

    return param_test
