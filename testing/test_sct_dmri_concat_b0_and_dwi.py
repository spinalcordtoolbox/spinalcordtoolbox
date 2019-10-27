#!/usr/bin/env python
#
# Test function for sct_dmri_concat_b0_and_dwi
#
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i dmri/dmri_T0000.nii.gz dmri/dmri.nii.gz -bvec dmri/bvecs.txt -bval dmri/bvals.txt '
                    '-order b0 dwi -o b0_dwi_concat.nii -obval bvals_concat.txt -obvec bvecs_concat.txt']
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