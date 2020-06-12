#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_fmri_compute_tsnr
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
    default_args = ['-i fmri/fmri_r.nii.gz -g 4 -x nn -r 0']
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
