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

import numpy as np
import sct_utils as sct
from msct_image import Image
from pandas import DataFrame
from math import sqrt


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i template/template/PAM50_small_t2.nii.gz -c t2']
    param_test.dist_threshold = 10.0
    # param_test.suffix_groundtruth = '_pmj_manual'  # file name suffix for ground truth (used for integrity testing)
    param_test.fname_groundtruth = param_test.path_data + 'template/template/PAM50_small_t2_pmj_manual.nii.gz'

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    Parameters
    ----------
    param_test: Class defined in sct_testing.py

    Returns
    -------
    param_test
    """
    # initializations
    distance_detection = float('nan')

    # extract name of output centerline: data_centerline_optic.nii.gz
    file_pmj = param_test.path_output + sct.add_suffix(param_test.file_input, '_pmj')

    # open output segmentation
    try:
        im_pmj = Image(file_pmj)
    except:
        param_test.output += 'ERROR: Cannot open output pmj mask: ' + file_pmj
        param_test.status = 99
        return param_test

    # open ground truth
    try:
        im_pmj_manual = Image(param_test.fname_groundtruth)
    except:
        param_test.output += 'ERROR: Cannot open ground truth pmj mask: ' + param_test.fname_groundtruth
        param_test.status = 99
        return param_test

    # compute Euclidean distance between predicted and GT PMJ label
    x_true, y_true, z_true = np.where(im_pmj_manual.data == 50)
    x_pred, y_pred, z_pred = np.where(im_pmj.data == 50)

    x_true, y_true, z_true = im_pmj_manual.transfo_pix2phys([[x_true[0], y_true[0], z_true[0]]])[0]
    x_pred, y_pred, z_pred = im_pmj.transfo_pix2phys([[x_pred[0], y_pred[0], z_pred[0]]])[0]

    distance_detection = sqrt(((x_true - x_pred))**2 + ((y_true - y_pred))**2 + ((z_true - z_pred))**2)

    param_test.output += 'Computed distance: ' + str(distance_detection)
    param_test.output += 'Distance threshold (if computed Distance higher: fail): ' + str(param_test.dist_threshold)

    if distance_detection > param_test.dist_threshold:
        param_test.status = 99

    # transform results into Pandas structure
    param_test.results = DataFrame(index=[param_test.path_data], data={'status': param_test.status, 'output': param_test.output, 'distance_detection': distance_detection, 'duration [s]': param_test.duration})

    return param_test
