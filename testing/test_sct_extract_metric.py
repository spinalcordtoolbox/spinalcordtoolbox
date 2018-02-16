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

import sys, io, os, pickle


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i mt/mtr.nii.gz -f mt/label/atlas -method wa -vert 4:5 -l 51 -o quantif_mtr.pickle']
    param_test.mtr_groundtruth = 32.7919  # ground truth value
    param_test.threshold_diff = 0.001  # threshold for computing difference between result and ground truth

    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    mtr_result = pickle.load(io.open(os.path.join(param_test.path_output, "quantif_mtr.pickle"), "rb"))['Metric value'][0]
    param_test.output += 'Computed MTR:     ' + str(mtr_result)
    param_test.output += '\nGround truth MTR: ' + str(param_test.mtr_groundtruth) + '\n'
    if abs(mtr_result - param_test.mtr_groundtruth) < param_test.threshold_diff:
        param_test.output += '--> PASSED'
    else:
        param_test.status = 99
        param_test.output += '--> FAILED'
    return param_test
