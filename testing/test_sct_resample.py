#!/usr/bin/env python
#########################################################################################
#
# Test function sct_resample
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# modified: 2014-10-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
import commands


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i fmri/fmri.nii.gz -f 0.5x0.5x1 -v 1']

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
