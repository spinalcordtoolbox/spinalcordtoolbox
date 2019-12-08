#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_multimodal
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import, division

import os

import numpy as np

from spinalcordtoolbox.image import Image


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i mt/mt0.nii.gz -d mt/mt1.nii.gz -o mt0_reg.nii.gz -param step=1,algo=syn,type=im,iter=1,smooth=1,shrink=2,metric=MI -x linear -r 0',
                    '-i mt/mt0.nii.gz -d mt/mt1.nii.gz -o mt0_reg.nii.gz -param step=1,algo=slicereg,type=im,iter=5,smooth=0,metric=MeanSquares -x linear -r 0',
                    '-i mt/mt0.nii.gz -iseg mt/mt0_seg.nii.gz -d mt/mt1.nii.gz -dseg mt/mt1_seg.nii.gz -o mt0_reg.nii.gz -param step=1,algo=centermassrot,type=seg,rot_method=pca -x linear -r 0',
                    '-i mt/mt0.nii.gz -iseg mt/mt0_seg.nii.gz -d mt/mt1.nii.gz -dseg mt/mt1_seg.nii.gz -o mt0_reg.nii.gz -param step=1,algo=centermassrot,type=imseg,rot_method=hog -x linear -r 0',
                    '-i mt/mt0.nii.gz -iseg mt/mt0_seg.nii.gz -d mt/mt1.nii.gz -dseg mt/mt1_seg.nii.gz -o mt0_reg.nii.gz -param step=1,algo=centermassrot,type=imseg,rot_method=pcahog -x linear -r 0',
                    '-i mt/mt0.nii.gz -iseg mt/mt0_seg.nii.gz -d mt/mt1.nii.gz -dseg mt/mt1_seg.nii.gz -o mt0_reg.nii.gz -param step=1,algo=columnwise,type=seg,smooth=1 -x linear -r 0']
    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # check if ground truth exists.
    if hasattr(param_test, 'fname_gt'):
        # compare result and groundtruth images
        param_test = compare_two_images(os.path.join(param_test.path_output, 'mt0_reg.nii.gz'), param_test.fname_gt, param_test)
    else:
        param_test.output += '\n--> N/A'
    return param_test


def compare_two_images(fname_result, fname_gt, param_test):
    """
    Compare two images and return status=99 if difference is above threshold
    """
    param_test.output += '\nComparing: ' + fname_result + ' and ' + fname_gt

    im_gt = Image(fname_gt)
    data_gt = im_gt.data
    data_result = Image(fname_result).data
    # get dimensions
    nx, ny, nz, nt, px, py, pz, pt = im_gt.dim
    # set the difference threshold to 1e-3 pe voxel
    threshold = 1e-3 * nx * ny * nz * nt
    # check if non-zero elements are present when computing the difference of the two images
    diff = data_gt - data_result
    # compare images
    if abs(np.sum(diff)) > threshold:
        param_test.status = 99
        param_test.output += '\n--> FAILED (%.3f)' % (np.sum(np.abs(diff)) / threshold)
    else:
        param_test.output += '\n--> PASSED'
    return param_test
