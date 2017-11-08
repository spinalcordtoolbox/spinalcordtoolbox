#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_dmri_create_noisemask
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley Gros
#
# About the license: see the file LICENSE.TXT
#########################################################################################


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i dmri/dmri.nii.gz -dof 1 -o noise_mask.nii.gz']  # default parameters
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
