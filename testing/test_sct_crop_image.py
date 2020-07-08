#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_crop_image
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

from spinalcordtoolbox.image import Image


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    param_test.fname_out = [
        't2_crop_xyz.nii',
        't2_crop_mask.nii',
        't2_crop_ref.nii',
    ]
    default_args = [
        '-i t2/t2.nii.gz -o {} -xmin 1 -xmax -3 -ymin 2 -ymax 10'.format(param_test.fname_out[0]),
        '-i t2/t2.nii.gz -o {} -m t2/t2_seg-manual.nii.gz'.format(param_test.fname_out[1]),
        '-i t2/t2.nii.gz -o {} -ref mt/mt0.nii.gz'.format(param_test.fname_out[2]),
    ]
    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # find which test is performed
    index_args = param_test.default_args.index(param_test.args)

    # check if cropping was correct depending on the scenario
    if index_args == 0:
        xyz = (57, 9, 52)
    elif index_args == 1:
        xyz = (11, 55, 13)
    elif index_args == 2:
        xyz = (37, 55, 34)

    nx, ny, nz, nt, px, py, pz, pt = Image(param_test.fname_out[index_args]).dim
    if (nx, ny, nz) == xyz:
        param_test.output += '--> PASSED'
    else:
        param_test.status = 99
        param_test.output += "Output dimensions: {}, {}, {}\n".format(nx, ny, nz)
        param_test.output += '--> FAILED'
    return param_test
