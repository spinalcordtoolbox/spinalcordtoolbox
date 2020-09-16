#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_process_segmentation script
#
#   replace the shell test script in sct 1.0
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/09/28
#
# About the license: see the file LICENSE.TXT
#########################################################################################
# TODO: make it compatible with isct_test_function
# TODO: add log file

from __future__ import absolute_import


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2_seg-manual.nii.gz']

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
