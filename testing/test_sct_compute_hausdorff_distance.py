#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_compute_hausdorff_distance
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: charley
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

import os


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2s/t2s_gmseg_manual.nii.gz -d t2s/t2s_gmseg_manual.nii.gz']
    param_test.max_hausdorff_distance = 1.0
    param_test.file_out = 'hausdorff_distance.txt'

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # open output segmentation
    hausdorff_txt = open(param_test.file_out, 'r')
    hausdorff_distance_lst = []
    for i, line in enumerate(hausdorff_txt):
        if line.startswith('Slice'):
            hausdorff_distance_lst.append(float(line.split(': ')[1].split(' -')[0]))
    hausdorff_txt.close()
    max_hausdorff_distance = max(hausdorff_distance_lst)

    param_test.output += 'Max Computed Hausdorff distance: ' + str(max_hausdorff_distance)
    param_test.output += 'Hausdorff distance threshold (if Max Computed Hausdorff distance higher: fail): ' + str(param_test.max_hausdorff_distance)

    if max_hausdorff_distance > param_test.max_hausdorff_distance:
        param_test.status = 99
        param_test.output += '--> FAILED'
    else:
        param_test.output += '--> PASSED'

    # update Panda structure
    param_test.results['max_hausdorff_distance'] = max_hausdorff_distance

    return param_test
