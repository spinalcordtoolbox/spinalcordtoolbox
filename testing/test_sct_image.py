#!/usr/bin/env python
#########################################################################################
#
# Test function sct_image
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# modified: 2015-10-06
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys, io, os

import sct_utils as sct
from msct_image import Image
import numpy as np


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    param_test.folder_data = ['mt/', 't2/', 'dmri/']
    param_test.file_data = ['mtr.nii.gz', 't2.nii.gz', 'dmri.nii.gz']

    # test padding
    param_test.pad = 2

    # test concatenation of data
    path_fname, file_fname, ext_fname = sct.extract_fname(param_test.file_data[2])
    param_test.dmri_t_slices = [os.path.join(param_test.folder_data[2], file_fname + '_T' + str(i).zfill(4) + ext_fname) for i in range(7)]
    input_concat = ','.join(param_test.dmri_t_slices)

    default_args = ['-i ' + os.path.join(param_test.folder_data[0], param_test.file_data[0]) + ' -o test.nii.gz' + ' -pad 0,0,'+str(param_test.pad),
                    '-i ' + os.path.join(param_test.folder_data[1], param_test.file_data[1]) + ' -getorient',  # 3D
                    '-i ' + os.path.join(param_test.folder_data[2], param_test.file_data[2]) + ' -getorient',  # 4D
                    '-i ' + os.path.join(param_test.folder_data[2], param_test.file_data[2]) + ' -split t',
                    '-i ' + input_concat + ' -concat t -o dmri_concat.nii.gz']

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

    # checking the integrity of padding an image
    if index_args == 0:
        nx, ny, nz, nt, px, py, pz, pt = Image(os.path.join(param_test.path_data, param_test.folder_data[0], param_test.file_data[0])).dim
        nx2, ny2, nz2, nt2, px2, py2, pz2, pt2 = Image(os.path.join(param_test.path_output, 'test.nii.gz')).dim

        if nz2 != nz + 2 * param_test.pad:
            param_test.status = 99
            param_test.output += '\nResulting pad image\'s dimension differs from expected:\n'
            param_test.output += 'dim : ' + str(nx2) + 'x' + str(ny2) + 'x' + str(nz2) + '\n'
            param_test.output += 'expected : ' + str(nx) + 'x' + str(ny) + 'x' + str(nz + 2 * param_test.pad) + '\n'

    elif index_args == 3:
        threshold = 1e-3
        try:
            path_fname, file_fname, ext_fname = sct.extract_fname(os.path.join(param_test.path_data, param_test.folder_data[2], param_test.file_data[2]))
            ref = Image(os.path.join(param_test.path_data, param_test.dmri_t_slices[0]))
            new = Image(os.path.join(param_test.path_data, param_test.folder_data[2], file_fname + '_T0000' + ext_fname))
            diff = ref.data - new.data
            if np.sum(diff) > threshold:
                param_test.status = 99
                param_test.output += '\nResulting split image differs from gold-standard.\n'
        except Exception as e:
            param_test.status = 99
            param_test.output += 'ERROR: ' + str(e.message) + str(e.args)

    elif index_args == 4:
        try:
            threshold = 1e-3
            ref = Image(os.path.join(param_test.path_data, param_test.folder_data[2], param_test.file_data[2]))
            new = Image(os.path.join(param_test.path_output, 'dmri_concat.nii.gz'))
            diff = ref.data - new.data
            if np.sum(diff) > threshold:
                param_test.status = 99
                param_test.output += '\nResulting concatenated image differs from gold-standard (original dmri image).\n'
        except Exception as e:
            param_test.status = 99
            param_test.output += 'ERROR: ' + str(e.message) + str(e.args)

    return param_test
