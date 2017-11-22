#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_process_segmentation script
#
#   replace the shell test script in sct 1.0
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/09/28
#
# About the license: see the file LICENSE.TXT
#########################################################################################
# TODO: make it compatible with isct_test_function
# TODO: add log file

import sct_utils as sct
from pandas import DataFrame
import time
import numpy as np
import nibabel as nib
import pickle
import sys


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2_seg.nii.gz -p centerline -v 1',
                    '-i t2/t2_seg.nii.gz -p length']

    # try:
    #     # Creation of files for multiple testing. These files are created in param_test.full_path_tmp
    #     # CSA with integrity testing on various angles of segmentation
    #     size_data = 31
    #     size_seg = 2  # segmentation size in the X and Y dimension is size_seg*2+1
    #
    #     # create folder with data
    #     param_test.full_path_tmp_data = os.path.join(param_test.full_path_tmp, 'data')
    #     sct.create_folder(os.path.join(param_test.full_path_tmp, 'data_sct_process_segmentation/'))
    #
    #     # Initialise numpy volumes
    #     data_seg = np.zeros((size_data, size_data, size_data), dtype=np.int16)
    #     # create labels=1 in a shape of a parallelipiped: 5x5xnz
    #     data_seg[np.round(size_data / 2) - size_seg:np.round(size_data / 2) + size_seg + 1,
    #              np.round(size_data / 2) - size_seg:np.round(size_data / 2) + size_seg + 1,
    #              0: size_data] = 1
    #     # save as nifti
    #     img = nib.Nifti1Image(data_seg, np.eye(4))
    #     nib.save(img, os.path.join(param_test.full_path_tmp_data, 'data_seg.nii.gz'))
    #
    #     # create nifti with rotated header
    #     # build rotation matrix
    #     alpha = 0.175  # corresponds to 10 deg angle
    #     beta = 0.175  # corresponds to 10 deg angle
    #     gamma = 0.175  # corresponds to 10 deg angle
    #     rotation_matrix = np.matrix([[np.cos(alpha)*np.cos(beta), np.cos(alpha)*np.sin(beta)*np.sin(gamma)-np.sin(alpha)*np.cos(gamma), np.cos(alpha)*np.sin(beta)*np.cos(gamma)+np.sin(alpha)*np.sin(gamma)],
    #                                  [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta)*np.sin(gamma)+np.cos(alpha)*np.cos(gamma), np.sin(alpha)*np.sin(beta)*np.cos(gamma)-np.cos(alpha)*np.sin(gamma)],
    #                                  [-np.sin(beta), np.cos(beta)*np.sin(gamma), np.cos(beta)*np.cos(gamma)]])
    #     affine_matrix = np.eye(4)
    #     affine_matrix[0:3, 0:3] = rotation_matrix
    #     img_rothd = nib.Nifti1Image(data_seg, affine_matrix)
    #     nib.save(img_rothd, os.path.join(param_test.full_path_tmp_data, 'data_seg_rothd.nii.gz'))
    #
    #     # rotate src image
    #     sct.run('sct_label_utils -i ' + os.path.join(param_test.full_path_tmp_data, 'data_seg.nii.gz') + ' -create 13,13,0,3:13,13,15,2:13,16,15,4:13,13,30,1 -o ' + os.path.join(param_test.full_path_tmp_data, 'data_rot_src.nii.gz'), 0)
    #     sct.run('sct_label_utils -i ' + os.path.join(param_test.full_path_tmp_data, 'data_seg.nii.gz') + ' -create 4,13,6,3:13,13,15,2:13,16,15,4:22,13,26,1 -o ' + os.path.join(param_test.full_path_tmp_data, 'data_rot_dest.nii.gz'), 0)
    #     sct.run('sct_register_multimodal -i ' + os.path.join(param_test.full_path_tmp_data, 'data_seg.nii.gz') + ' -d ' + os.path.join(param_test.full_path_tmp_data, 'data_seg.nii.gz') + ' -ilabel ' + os.path.join(param_test.full_path_tmp_data, 'data_rot_src.nii.gz') + ' -dlabel ' + os.path.join(param_test.full_path_tmp_data, 'data_rot_dest.nii.gz') + ' -param step=0,type=label,dof=Rx_Ry:step=1,type=im,algo=syn,iter=0 -x linear -ofolder ' + param_test.full_path_tmp_data, 0)
    #
    #     param_test.csa_truevalue = 25.0
    #     param_test.th_csa_error = 1  # maximum threshold of CSA error allowed
    #
    # except Exception as e:
    #     param_test.status = 99
    #     param_test.output += 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
    #     param_test.output += str(e)

    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    #
    # # find the test that is performed and check the integrity of the output
    # index_args = param_test.default_args.index(param_test.args)
    #
    # if index_args == 0:  # only do this integrity test once
    #     try:
    #         # Integrity testing on fake data
    #         # compute CSA
    #         cmd = 'sct_process_segmentation -i ' + os.path.join(param_test.full_path_tmp_data, 'data_seg.nii.gz') \
    #               + ' -p csa' \
    #               + ' -size 0' \
    #               + ' -r 0' \
    #               + ' -v 1' \
    #               + ' -z 5:23' \
    #               + ' -ofolder csa'
    #         param_test.output += '\n====================================================================================================\n' + cmd + '\n====================================================================================================\n\n'  # copy command
    #         status, o = sct.run(cmd, 0)
    #         param_test.output += o
    #         # check integrity
    #         csa = pickle.load(open('csa/csa_mean.pickle', 'rb'))
    #         csa_error = np.abs(csa['MEAN across slices'] - param_test.csa_truevalue)
    #         if csa_error > param_test.th_csa_error:
    #             param_test.status = 99
    #             param_test.output += '\nWARNING: CSA_ERROR = ' + str(csa_error) + ' < ' + str(param_test.th_csa_error)
    #
    #         # compute CSA on rotated image
    #         cmd = 'sct_process_segmentation -i ' + os.path.join(param_test.full_path_tmp_data, 'data_seg_src_reg.nii.gz') \
    #               + ' -p csa' \
    #               + ' -size 0' \
    #               + ' -r 0' \
    #               + ' -v 1' \
    #               + ' -z 5:23' \
    #               + ' -ofolder csa_rot'
    #         param_test.output += '\n====================================================================================================\n' + cmd + '\n====================================================================================================\n\n'  # copy command
    #         status, o = sct.run(cmd, 0)
    #         param_test.output += o
    #         # check integrity
    #         csa = pickle.load(open('csa_rot/csa_mean.pickle', 'rb'))
    #         csa_error = np.abs(csa['MEAN across slices'] - param_test.csa_truevalue)
    #         if csa_error > param_test.th_csa_error:
    #             param_test.status = 99
    #             param_test.output += '\nWARNING: CSA_ERROR = ' + str(csa_error) + ' < ' + str(param_test.th_csa_error)
    #
    #         # compute CSA on rotated header
    #         cmd = 'sct_process_segmentation -i ' + os.path.join(param_test.full_path_tmp_data, 'data_seg_rothd.nii.gz') \
    #               + ' -p csa' \
    #               + ' -size 0' \
    #               + ' -r 0' \
    #               + ' -v 1' \
    #               + ' -z 5:23' \
    #               + ' -ofolder csa_rothd'
    #         param_test.output += '\n====================================================================================================\n' + cmd + '\n====================================================================================================\n\n'  # copy command
    #         status, o = sct.run(cmd, 0)
    #         param_test.output += o
    #         # check integrity
    #         csa = pickle.load(open('csa_rothd/csa_mean.pickle', 'rb'))
    #         csa_error = np.abs(csa['MEAN across slices'] - param_test.csa_truevalue)
    #         if csa_error > param_test.th_csa_error:
    #             param_test.status = 99
    #             param_test.output += '\nWARNING: CSA_ERROR = ' + str(csa_error) + ' < ' + str(param_test.th_csa_error)
    #
    #     except Exception as e:
    #         param_test.status = 99
    #         param_test.output += 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
    #         param_test.output += str(e)

    param_test.output += '\nNot implemented.'
    return param_test
