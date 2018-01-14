#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_dmri_separate_b0_and_dwi
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys, io, os

import numpy as np

from msct_image import Image


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i dmri/dmri.nii.gz -bvec dmri/bvecs.txt -a 1 -ofolder . -r 0']
    param_test.threshold = 0.001
    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # check DWI
    param_test.output += 'Checking DWI\n'
    ref_dwi = Image(os.path.join(param_test.path_data, 'dmri', 'dwi.nii.gz'))
    new_dwi = Image(os.path.join(param_test.path_output, 'dwi.nii.gz'))
    diff_dwi = ref_dwi.data - new_dwi.data
    if np.sum(diff_dwi) > param_test.threshold:
        param_test.status = 99
        param_test.output += '--> FAILED'
    else:
        param_test.output += '--> PASSED'
    # check b=0
    param_test.output += '\n\nChecking b=0\n'
    ref_b0 = Image(os.path.join(param_test.path_data, 'dmri', 'dmri_T0000.nii.gz'))
    new_b0 = Image(os.path.join(param_test.path_output, 'b0.nii.gz'))
    diff_b0 = ref_b0.data - new_b0.data
    if np.sum(diff_b0) > param_test.threshold:
        param_test.status = 99
        param_test.output += '--> FAILED'
    else:
        param_test.output += '--> PASSED'
    return param_test
