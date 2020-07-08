#!/usr/bin/env python
#########################################################################################
#
# Test function sct_qc
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2.nii.gz -s t2/t2_seg-manual.nii.gz -p sct_deepseg_sc -qc-dataset sct_testing_data -qc-subject dummy']  # default parameters

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

