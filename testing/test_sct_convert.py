0#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_convert
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: charley
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import os


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2.nii.gz -o t2.nii']
    param_test.file_out = 't2.nii'

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # Simply check if output file exists
    if os.path.exists(param_test.file_out):
        param_test.output += '--> PASSED'
    else:
        param_test.status = 99
        param_test.output += '\nOutput file does not exist.'
    return param_test
