#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_extract_metric
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

import os, csv
import numpy as np


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i mt/mtr.nii.gz -f mt/label/atlas -method wa -l 51 -z 1:2 -o quantif_mtr.csv']
    param_test.file_out = 'quantif_mtr.csv'
    param_test.mtr_groundtruth = 32.6404  # ground truth value
    param_test.threshold_diff = 0.001  # threshold for computing difference between result and ground truth

    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    with open(param_test.file_out, 'r') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=',')
        mtr_result = np.float([row['WA()'] for row in spamreader][0])

    param_test.output += 'Computed MTR:     ' + str(mtr_result)
    param_test.output += '\nGround truth MTR: ' + str(param_test.mtr_groundtruth) + '\n'
    if abs(mtr_result - param_test.mtr_groundtruth) < param_test.threshold_diff:
        param_test.output += '--> PASSED'
    else:
        param_test.status = 99
        param_test.output += '--> FAILED'
    return param_test
