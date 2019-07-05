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

from __future__ import absolute_import
import os
import numpy as np
from spinalcordtoolbox.image import Image


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i dmri/dmri.nii.gz -bvec dmri/bvecs.txt -a 1 -r 0']
    param_test.file_dwi_in = 'dmri/dwi.nii.gz'
    param_test.file_dwi_out = 'dmri_dwi.nii.gz'
    param_test.file_b0_in = 'dmri/dmri_T0000.nii.gz'
    param_test.file_b0_out = 'dmri_b0.nii.gz'
    param_test.norm_threshold = 0.001
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
    ref_dwi = Image(param_test.file_dwi_in)
    new_dwi = Image(param_test.file_dwi_out)
    norm_img = np.linalg.norm(ref_dwi.data - new_dwi.data)
    if norm_img > param_test.norm_threshold:
        param_test.status = 99
        param_test.output += '--> FAILED'
    else:
        param_test.output += '--> PASSED'
    # check b=0
    param_test.output += '\n\nChecking b=0\n'
    ref_dwi = Image(param_test.file_b0_in)
    new_dwi = Image(param_test.file_b0_out)
    norm_img = np.linalg.norm(ref_dwi.data - new_dwi.data)
    if norm_img > param_test.norm_threshold:
        param_test.status = 99
        param_test.output += '--> FAILED'
    else:
        param_test.output += '--> PASSED'
    return param_test
