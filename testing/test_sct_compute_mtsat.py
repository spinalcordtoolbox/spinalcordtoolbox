#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_compute_mtsat
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Joshue Newton
#
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = [
        "-mt mt/mt1.nii.gz -pd mt/mt0.nii.gz -t1 mt/t1w.nii.gz",
        "-mt mt/mt1.nii.gz -pd mt/mt0.nii.gz -t1 mt/t1w.nii.gz -trmt 51 -trpd 52 -trt1 10 -famt 4 -fapd 5 -fat1 14"
    ]

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    if not os.path.isfile('mt/mtsat.nii.gz'):
        param_test.status = 99
        param_test.output += "\n--> FAILED"
    else:
        param_test.output += "\n--> PASSED"

    return param_test
