#!/usr/bin/env python
#########################################################################################
#
# Test function sct_propseg
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys, io, os, commands

import sct_utils as sct
from msct_image import Image, compute_dice
from pandas import DataFrame


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2.nii.gz -c t2 -igt t2/t2_seg_manual.nii.gz']  # default parameters
    # param_test.list_fname_gt = [os.path.join(param_test.path_data, 't2', 't2_seg_manual.nii.gz')]  # file name suffix for ground truth (used for integrity testing)
    param_test.dice_threshold = 0.9

    # check if isct_propseg compatibility
    # TODO: MAKE SURE THIS CASE WORKS AFTER MAJOR REFACTORING
    status_isct_propseg, output_isct_propseg = commands.getstatusoutput('isct_propseg')
    isct_propseg_version = output_isct_propseg.split('\n')[0]
    if isct_propseg_version != 'sct_propseg - Version 1.1 (2015-03-24)':
        status = 99
        param_test.output += '\nERROR: isct_propseg does not seem to be compatible with your system or is no up-to-date... Please contact SCT administrators.'
        return param_test

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
    file_seg = os.path.join(param_test.path_output, sct.add_suffix(param_test.file_input, '_seg'))
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
