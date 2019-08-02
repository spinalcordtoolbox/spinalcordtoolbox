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

from __future__ import absolute_import

from spinalcordtoolbox.image import Image, compute_dice


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2s/t2s_uncropped.nii.gz -o output.nii.gz -qc testing-qc']
    param_test.file_seg = 'output.nii.gz'
    param_test.fname_gt = 't2s/t2s_uncropped_gmseg_manual.nii.gz'
    param_test.dice_threshold = 0.85

    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # open output segmentation
    im_seg = Image(param_test.file_seg)
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
    param_test.results['dice'] = dice_segmentation
    return param_test
