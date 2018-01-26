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
    default_args = ['-i t2s/t2s_uncropped.nii.gz -o t2s_uncropped_gmseg.nii.gz']
    param_test.dice_threshold = 0.85
    param_test.f_ground_truth = 't2s/t2s_uncropped_gmseg_manual.nii.gz'

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
    fname_gt_path = os.path.join(param_test.path_data, param_test.f_ground_truth)
    im_seg_manual = Image(fname_gt_path)
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
