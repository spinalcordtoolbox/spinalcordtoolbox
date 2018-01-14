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

import sys, io, os, math

import numpy as np
from scipy.ndimage.measurements import center_of_mass

import sct_utils as sct
from msct_image import Image
from pandas import DataFrame


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
    # initializations
    mse_detection = float('nan')

    # extract name of output centerline: data_centerline_optic.nii.gz
    file_ctr = os.path.join(param_test.path_output, sct.add_suffix(param_test.file_input, '_centerline_optic'))

    # open output segmentation
    im_ctr = Image(file_ctr)

    # open ground truth
    im_seg_manual = Image(param_test.fname_gt)
    im_ctr_manual = im_seg_manual.copy() # Create Ctr GT from SC seg GT

    if im_ctr_manual.orientation != 'RPI':
        im_ctr_manual.change_orientation('RPI')

    im_ctr_manua_data = im_ctr_manual.data

    # Compute center of mass of the SC seg on each axial slice.
    center_of_mass_x_y_z_lst = [[int(center_of_mass(im_ctr_manua_data[:,:,zz])[0]), int(center_of_mass(im_ctr_manua_data[:,:,zz])[1]), zz] for zz in range(im_ctr_manual.dim[2])]

    im_ctr_manual.data *= 0
    for x_y_z in center_of_mass_x_y_z_lst:
        im_ctr_manual.data[x_y_z[0], x_y_z[1], x_y_z[2]] = 1

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
