#!/usr/bin/env python
#########################################################################################
#
# Test function sct_maths
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
    default_args = ['-i mt/mtr.nii.gz -percent 95 -o test.nii.gz',
                    '-i mt/mtr.nii.gz -add 1 -o test.nii.gz',
                    '-i mt/mtr.nii.gz -add mt/mtr.nii.gz mt/mtr.nii.gz -o test.nii.gz']

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
