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

def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i mt/mtr.nii.gz -f mt/label/atlas -method wa -vert 4:5 -l 51 -o quantif_mtr.pickle']
    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    import pickle
    mtr_groundtruth = 32.7919  # ground truth value
    threshold_diff = 0.001  # threshold for computing difference between result and ground truth
    mtr_result = pickle.load(open(param_test.path_output +"quantif_mtr.pickle", "rb"))['Metric value'][0]
    param_test.output += 'Computed MTR:     ' + str(mtr_result)
    param_test.output += '\nGround truth MTR: ' + str(mtr_groundtruth)
    if abs(mtr_result - mtr_groundtruth) < threshold_diff:
        param_test.output += '\n--> PASSED'
    else:
        param_test.status = 99
        param_test.output += '\n--> FAILED'
    return param_test
