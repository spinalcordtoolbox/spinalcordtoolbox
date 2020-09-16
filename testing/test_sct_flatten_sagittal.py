#!/usr/bin/env python
#########################################################################################
#
# Test function sct_flatten_sagital
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/19
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

import sct_utils as sct


def init(param_test):
    """
    Initialize testing.
    Parameters
    ----------
    param_test: Class defined in sct_testing.py

    Returns
    -------
    param_test
    """
    # initialization
    default_args = ['-i t2/t2.nii.gz -s t2/t2_seg-manual.nii.gz']  # default parameters

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
