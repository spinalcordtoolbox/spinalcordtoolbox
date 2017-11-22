#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_detect_pmj script
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: charley
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import math, shutil, os

import numpy as np

import sct_utils as sct
from msct_image import Image

def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i template/template/PAM50_small_t2.nii.gz -c t2 -igt template/template/PAM50_small_t2_pmj_manual.nii.gz']
    param_test.dist_threshold = 10.0

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # initializations
    distance_detection = float('nan')

    # extract name of output centerline: data_centerline_optic.nii.gz
    file_pmj = os.path.join(param_test.path_output, sct.add_suffix(param_test.file_input, '_pmj'))

    # open output segmentation
    im_pmj = Image(file_pmj)

    # open ground truth
    im_pmj_manual = Image(param_test.fname_gt)

    # compute Euclidean distance between predicted and GT PMJ label
    x_true, y_true, z_true = np.where(im_pmj_manual.data == 50)
    x_pred, y_pred, z_pred = np.where(im_pmj.data == 50)

    x_true, y_true, z_true = im_pmj_manual.transfo_pix2phys([[x_true[0], y_true[0], z_true[0]]])[0]
    x_pred, y_pred, z_pred = im_pmj.transfo_pix2phys([[x_pred[0], y_pred[0], z_pred[0]]])[0]

    distance_detection = math.sqrt(((x_true - x_pred))**2 + ((y_true - y_pred))**2 + ((z_true - z_pred))**2)

    param_test.output += 'Computed distance: ' + str(distance_detection)
    param_test.output += 'Distance threshold (if computed Distance higher: fail): ' + str(param_test.dist_threshold)

    if distance_detection > param_test.dist_threshold:
        param_test.status = 99
        param_test.output += '--> FAILED'
    else:
        param_test.output += '--> PASSED'

    # update Panda structure
    param_test.results['distance_detection'] = distance_detection

    return param_test
