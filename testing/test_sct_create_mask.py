#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_create_mask
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i mt/mt1.nii.gz -p coord,15x17 -size 10 -r 0',
                    '-i mt/mt1.nii.gz -p point,mt/mt1_point.nii.gz -size 10 -r 0',
                    '-i mt/mt1.nii.gz -p center -size 10 -r 0',
                    '-i mt/mt1.nii.gz -p centerline,mt/mt1_seg.nii.gz -size 5 -r 0',
                    '-i dmri/dmri.nii.gz -p center -size 10 -r 0']

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
