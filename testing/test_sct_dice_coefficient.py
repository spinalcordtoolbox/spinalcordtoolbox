#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_dice_coefficient
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys, io, os

import sct_utils as sct
from msct_image import Image, compute_dice


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2_seg_manual.nii.gz -d t2/t2_seg_manual.nii.gz']  # default parameters
    param_test.contrast = 't2'

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    path_data = os.path.join(param_test.path_data, param_test.contrast, param_test.file_input)
    integrity_value = 1.0

    # open output segmentation
    try:
        im_seg_manual = Image(path_data)
    except:
        param_test.output += 'ERROR: Cannot open ground truth segmentation: ' + path_data
        param_test.status = 99
        return param_test

    # compute dice coefficient between generated image and image from database
    dice_segmentation = compute_dice(im_seg_manual, im_seg_manual, mode='3d', zboundaries=False)

    param_test.output += 'Computed dice: '+str(dice_segmentation)

    if dice_segmentation != integrity_value:
        param_test.output += '\nERROR: Dice coefficient should be : ' + str(integrity_value)
        param_test.status = 99

    return param_test
