#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_merge_images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: charley
#
# About the license: see the file LICENSE.TXT
#########################################################################################


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i template/template/PAM50_small_cord.nii.gz,t2/t2_seg-manual.nii.gz -w mt/warp_template2mt.nii.gz,t2/warp_template2anat.nii.gz -d mt/mt1.nii.gz']

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    param_test.output += '\nNot implemented.'
    return param_test
