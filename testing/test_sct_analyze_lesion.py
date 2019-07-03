#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_analyze_lesion script
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
    default_args = ['-m t2/t2_seg_manual.nii.gz -s t2/t2_seg_manual.nii.gz']

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # Simply check if output pkl file exists
    if not os.path.exists('t2_seg_manual_analyzis.pkl'):
        param_test.output += '\nOutput file does not exist.'
    return param_test
