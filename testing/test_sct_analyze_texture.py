#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_analyze_texture script
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: charley
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import, division

import numpy as np

from spinalcordtoolbox.image import Image


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2.nii.gz -m t2/t2_seg-manual.nii.gz -feature contrast -distance 1 -ofolder .']
    param_test.norm_threshold = 0.001
    param_test.file_texture = 't2_contrast_1_mean.nii.gz'
    param_test.fname_gt = 't2/t2_contrast_1_mean_ref.nii.gz'

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # open output
    im_texture = Image(param_test.file_texture)
    # open ground truth
    im_texture_ref = Image(param_test.fname_gt)
    # Compute norm
    norm_img = np.linalg.norm(im_texture.data - im_texture_ref.data)
    if norm_img > param_test.norm_threshold:
        param_test.output += '--> FAILED'
        param_test.status = 99
    else:
        param_test.output += '--> PASSED'

    # update Panda structure
    param_test.results['norm'] = norm_img

    return param_test
