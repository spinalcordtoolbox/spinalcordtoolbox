#!/usr/bin/env python
#########################################################################################
#
# Test function sct_compute_mscc
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# modified: 2017/10/16
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

import numpy as np

import sct_compute_mscc

def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    param_test.default_args_values = {'di': 6.85, 'da': 7.65, 'db': 7.02}
    default_args = ['-di 6.85 -da 7.65 -db 7.02']  # default parameters
    param_test.default_result = 6.612133606

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    mscc = sct_compute_mscc.mscc(di=param_test.default_args_values['di'],
                                 da=param_test.default_args_values['da'],
                                 db=param_test.default_args_values['db'])

    error = np.abs(np.round(mscc, 4) - np.round(param_test.default_result, 4))

    if error != 0.0:
        param_test.status = 99

    return param_test
