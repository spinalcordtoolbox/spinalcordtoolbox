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

import os, math

import numpy as np
from scipy.ndimage.measurements import center_of_mass

import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image


def compute_mse(im_true, im_pred):
    mse_dist = []
    count_slice = 0
    for z in range(im_true.dim[2]):

        if np.sum(im_true.data[:, :, z]):
            x_true, y_true = [np.where(im_true.data[:, :, z] > 0)[i][0] for i in range(len(np.where(im_true.data[:, :, z] > 0)))]
            x_pred, y_pred = [np.where(im_pred.data[:, :, z] > 0)[i][0] for i in range(len(np.where(im_pred.data[:, :, z] > 0)))]

            x_true, y_true = im_true.transfo_pix2phys([[x_true, y_true, z]])[0][0], im_true.transfo_pix2phys([[x_true, y_true, z]])[0][1]
            x_pred, y_pred = im_pred.transfo_pix2phys([[x_pred, y_pred, z]])[0][0], im_pred.transfo_pix2phys([[x_pred, y_pred, z]])[0][1]

            dist = ((x_true - x_pred))**2 + ((y_true - y_pred))**2
            mse_dist.append(dist)

            count_slice += 1

    return math.sqrt(sum(mse_dist) / float(count_slice))


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2s/t2s.nii.gz -c t2s -igt t2s/t2s_seg.nii.gz']  # default parameters
    param_test.mse_threshold = 3.0

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """

    # open ground truth
    im_seg_manual = Image(param_test.fname_gt).change_orientation("RPI")

    # Compute center of mass of the SC seg on each axial slice.
    center_of_mass_x_y_z_lst = [[int(center_of_mass(im_seg_manual.data[:, :, zz])[0]),
                                 int(center_of_mass(im_seg_manual.data[:, :, zz])[1]),
                                 zz] for zz in range(im_seg_manual.dim[2])]

    im_ctr_manual = msct_image.zeros_like(im_seg_manual)
    for x_y_z in center_of_mass_x_y_z_lst:
        im_ctr_manual.data[x_y_z[0], x_y_z[1], x_y_z[2]] = 1

    # open output segmentation
    path_in, file_in, _ = sct.extract_fname(param_test.file_input)
    file_ctr = os.path.join(param_test.path_data, 't2s', sct.add_suffix(param_test.file_input, '_centerline'))
    im_ctr = Image(file_ctr).change_orientation("RPI")

    # compute MSE between generated ctr and ctr from database
    mse_detection = compute_mse(im_ctr, im_ctr_manual)

    param_test.output += 'Computed MSE: ' + str(mse_detection)
    param_test.output += 'MSE threshold (if computed MSE higher: fail): ' + str(param_test.mse_threshold)

    if mse_detection > param_test.mse_threshold:
        param_test.status = 99
        param_test.output += '--> FAILED'
    else:
        param_test.output += '--> PASSED'

    # update Panda structure
    param_test.results['mse_detection'] = mse_detection

    return param_test
