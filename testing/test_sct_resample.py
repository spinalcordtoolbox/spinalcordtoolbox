#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_resample
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# modified: 2014-10-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys, io, os

from msct_image import Image


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i dmri/dmri.nii.gz -f 0.5x0.5x1 -v 1 -o resampled.nii.gz',  # 4D, factor
                    '-i t2/t2.nii.gz -mm 0.97x1.14x1.2 -v 1 -o resampled.nii.gz',  # 3D, mm
                    '-i t2/t2.nii.gz -vox 120x110x26 -v 1 -o resampled.nii.gz'  # 3D, vox
                    ]

    param_test.results_dims = [(20, 21, 5, 7, 1.6826923, 1.6826923, 17.5, 2.2),  # 4D, factor
                               (62, 48, 43, 1, 0.96774191, 1.1458334, 1.2093023, 1),  # 3D, mm
                               (120, 110, 26, 1, 0.5, 0.5, 2.0, 1)  # 3D, vox
                               ]

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
    param_test.output += '\nTesting ' + param_test.args + '\n'

    # Open resulting image and check dimensions and spacing
    image_result = Image(os.path.join(param_test.path_output, 'resampled.nii.gz'))
    dims = image_result.dim

    if not all(round(i, 4) == round(j, 4) for i, j in zip(dims, param_test.results_dims[index_args])):
        param_test.output += 'WARNING: dimensions and spacing different from expected.' \
                             '\n--> Results: ' + str(dims) + \
                             '\n--> Expected: ' + str(param_test.results_dims[index_args])
        param_test.status = 99

    return param_test
