#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_get_centerline script
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: charley
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import, division

import math

import numpy as np
from scipy.ndimage.measurements import center_of_mass

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2s/t2s.nii.gz -c t2s']  # default parameters
    param_test.file_ctr = 't2s/t2s_centerline.nii.gz'
    param_test.fname_gt = 't2s/t2s_seg.nii.gz'
    param_test.mse_threshold = 3.0

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Integrity test has moved to unit_testing/ and is run with pytest
    """
    return param_test
