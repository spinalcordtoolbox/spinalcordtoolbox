#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_pipeline
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
    default_args = ['-f sct_maths -d ../ -p \"-i mt/mt0.nii.gz -percent 95 -o mt0_95.nii.gz\" -p \"-i mt/mt1.nii.gz -add 5 -o mt1_add5.nii.gz\"']
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
