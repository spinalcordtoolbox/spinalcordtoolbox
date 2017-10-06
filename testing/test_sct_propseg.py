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

import sct_utils as sct
import commands
from msct_image import Image, compute_dice
from pandas import DataFrame


def init(param_test):
    """
    Initialize testing.
    Parameters
    ----------
    param_test: Class defined in sct_testing.py

    Returns
    -------
    param_test
    """
    # initialization
    default_args = ['-i t2/t2.nii.gz -c t2']  # default parameters
    param_test.dice_threshold = 0.9
    param_test.suffix_groundtruth = '_seg_manual'  # file name suffix for ground truth (used for integrity testing)

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
    Parameters
    ----------
    param_test: Class defined in sct_testing.py

    Returns
    -------
    param_test
    """
    # initializations
    dice_segmentation = float('nan')

    # extract name of output segmentation: data_seg.nii.gz
    file_seg = param_test.path_output + sct.add_suffix(param_test.file_input, '_seg')

    # open output segmentation
    try:
        im_seg = Image(file_seg)
    except:
        param_test.output += 'ERROR: Cannot open output segmentation: ' + segmentation_filename
        return param_test

    # open ground truth
    try:
        im_seg_manual = Image(param_test.fname_groundtruth)
    except:
        param_test.output += 'ERROR: Cannot open ground truth segmentation: ' + param_test.fname_groundtruth
        return param_test

    # compute dice coefficient between generated image and image from database
    dice_segmentation = compute_dice(im_seg, im_seg_manual, mode='3d', zboundaries=False)

    param_test.output += 'Computed dice: '+str(dice_segmentation)
    param_test.output += 'Dice threshold (if computed dice smaller: fail): '+str(param_test.dice_threshold)

    if dice_segmentation < param_test.dice_threshold:
        param_test.status = 99

    # transform results into Pandas structure
    param_test.results = DataFrame(index=[param_test.path_data], data={'status': param_test.status, 'output': param_test.output, 'dice_segmentation': dice_segmentation, 'duration [s]': param_test.duration})

    return param_test
