#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_warp_template script
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
    default_args = ['-d mt/mt1.nii.gz -w mt/warp_template2mt.nii.gz -a 0 -t template -qc 0']  # default parameters

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
