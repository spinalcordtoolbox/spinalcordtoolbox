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

def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i dmri/dmri.nii.gz -bvec dmri/bvecs.txt -a 1 -ofolder ./ -r 0']
    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    from msct_image import Image
    from numpy import sum
    threshold = 1e-3
    ref_dwi = Image(param_test.path_data + 'dmri/dwi.nii.gz')
    new_dwi = Image(param_test.path_output + 'dwi.nii.gz')
    diff_dwi = ref_dwi.data - new_dwi.data
    if sum(diff_dwi) > threshold:
        param_test.status = 99
        param_test.output += '\nResulting DWI image differs from gold-standard.\n'
    else:
        param_test.output += '\nResulting DWI is the same as gold-standard.\n'

    ref_b0 = Image(param_test.path_data + 'dmri/dmri_T0000.nii.gz')
    new_b0 = Image(param_test.path_output + 'b0.nii.gz')
    diff_b0 = ref_b0.data - new_b0.data
    if sum(diff_b0) > threshold:
        param_test.status = 99
        param_test.output = '\nResulting b0 image differs from gold-standard.\n'
    return param_test
