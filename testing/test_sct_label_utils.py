#!/usr/bin/env python
#########################################################################################
#
# Test function sct_label_utils
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add test to other processes.

import os
import sct_utils as sct
import sct_label_utils

from pandas import DataFrame


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    folder_data = ['t2']
    file_data = ['t2_seg.nii.gz', 't2_seg_labeled.nii.gz']

    default_args = ['-i ' + os.path.join(folder_data[0], file_data[0]) + ' -create 1,1,1,1:2,2,2,2',
                    '-i ' + os.path.join(folder_data[0], file_data[0]) + ' -cubic-to-point -o test_centerofmass.nii.gz']
    param_test.centers_of_mass = '31,28,25,1'

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """

    # find the test that is performed and check the integrity of the output
    index_args = param_test.default_args.index(param_test.args)

    if index_args == 1:
        # compute center of mass of labeled segmentation
        centers_of_mass_image = sct_label_utils.main(['-i', 'test_centerofmass.nii.gz', '-display', '-v', '0'])
        # compare with ground truch value
        if centers_of_mass_image != param_test.centers_of_mass:
            param_test.output += 'WARNING: Center of mass different from gold-standard. \n--> Results:   ' + centers_of_mass_image + '\n--> Should be: ' + param_test.centers_of_mass + '\n'
            param_test.status = 99

    # transform results into Pandas structure
    param_test.results = DataFrame(data={'status': param_test.status, 'output': param_test.output, 'duration [s]': param_test.duration}, index=[param_test.path_data])

    # end test
    return param_test
