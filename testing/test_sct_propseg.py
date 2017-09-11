#!/usr/bin/env python
#########################################################################################
#
# Test function sct_propseg
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/09
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
import commands
from sct_testing import write_to_log_file, init_testing
from msct_image import Image, compute_dice
from pandas import DataFrame


def init(param_test):

    # initialization
    default_args = '-i t2/t2.nii.gz -c t2'  # default parameters
    param_test.dice_threshold = 0.9
    # param.dict_result = {'status': param.status, 'output': param.status, 'dice_segmentation': float('nan'), 'duration [s]': 0}

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

    # initialize testing and run function
    param_test = init_testing(param_test)

def test_integrity(param_test):
    """

    Parameters
    ----------
    param_test

    Returns
    -------

    """
    # extract name of manual segmentation
    # by convention, manual segmentation are called inputname_seg_manual.nii.gz where inputname is the filename
    # of the input image
    segmentation_filename = param_test.path_output + sct.add_suffix(param_test.fname_input, '_seg')
    manual_segmentation_filename = param_test.path_data + param_test.contrast + '/' + sct.add_suffix(param_test.fname_input, '_seg_manual')
    dice_segmentation = float('nan')

    # if command ran without error, test integrity
    if param_test.status == 0:
        # compute dice coefficient between generated image and image from database
        dice_segmentation = compute_dice(Image(segmentation_filename), Image(manual_segmentation_filename), mode='3d', zboundaries=False)

        if dice_segmentation < param_test.dice_threshold:
            param_test.status = 99

    # transform results into Pandas structure
    param_test.results = DataFrame(index=[param_test.path_data], data={'status': param_test.status, 'output': param_test.output, 'dice_segmentation': dice_segmentation, 'duration [s]': param_test.duration})

    return param_test


if __name__ == "__main__":
    # call main function
    test()