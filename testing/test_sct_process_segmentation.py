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

import sct_utils as sct
from pandas import DataFrame
import time


def test(path_data='', parameters=''):

    # initializations
    result_rmse = float('NaN')
    result_dist_max = float('NaN')
    time_start = time.time()

    if not parameters:
        parameters = '-i t2/t2_seg.nii.gz'

    # parameters
    folder_data = 't2/'
    file_data = 't2_seg.nii.gz'

    # define command
    cmd = 'sct_process_segmentation -i ' + path_data + folder_data + file_data \
          + ' -p centerline' \
          + ' -v 1'
    output = '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    status, o = sct.run(cmd, 0)
    output += o

    # define command
    cmd = 'sct_process_segmentation -i ' + path_data + folder_data + file_data \
          + ' -p length'
    output = '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    status, o = sct.run(cmd, 0)
    output += o

    # define command
    cmd = 'sct_process_segmentation -i ' + path_data + folder_data + file_data \
          + ' -p csa' \
          + ' -size 1'\
          + ' -r 0'\
          + ' -v 1'
    output = '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    status, o = sct.run(cmd, 0)
    output += o

    # CSA with integrity testing on various angles of segmentation
    import numpy as np
    import nibabel as nib
    import pickle
    size_data = 31
    size_seg = 2  # segmentation size in the X and Y dimension is size_seg*2+1
    csa_truevalue = 25.0
    th_csa_error = 1  # maximum threshold of CSA error allowed
    # Initialise numpy volumes
    data_seg = np.zeros((size_data, size_data, size_data), dtype=np.int16)
    # create labels=1 in a shape of a parallelipiped: 5x5xnz
    data_seg[np.round(size_data / 2) - size_seg:np.round(size_data / 2) + size_seg + 1,
             np.round(size_data / 2) - size_seg:np.round(size_data / 2) + size_seg + 1,
             0: size_data] = 1
    # save as nifti
    img = nib.Nifti1Image(data_seg, np.eye(4))
    nib.save(img, 'data_seg.nii.gz')
    # rotate src image
    sct.run('sct_label_utils -i data_seg.nii.gz -create 13,13,0,3:13,13,15,2:13,16,15,4:13,13,30,1 -o data_rot_src.nii.gz', 0)
    sct.run('sct_label_utils -i data_seg.nii.gz -create 4,13,6,3:13,13,15,2:13,16,15,4:22,13,26,1 -o data_rot_dest.nii.gz', 0)
    sct.run('sct_register_multimodal -i data_seg.nii.gz -d data_seg.nii.gz -ilabel data_rot_src.nii.gz -dlabel data_rot_dest.nii.gz -param step=0,type=label,dof=Rx_Ry:step=1,type=im,algo=syn,iter=0 -x linear', 0)
    # compute CSA
    cmd = 'sct_process_segmentation -i data_seg.nii.gz' \
          + ' -p csa' \
          + ' -size 0' \
          + ' -r 0' \
          + ' -v 1' \
          + ' -z 5:23' \
          + ' -ofolder csa'
    output = '\n====================================================================================================\n' + cmd + '\n====================================================================================================\n\n'  # copy command
    status, o = sct.run(cmd, 0)
    output += o
    # check integrity
    csa = pickle.load(open('csa/csa_mean.pickle', 'rb'))
    csa_error = np.abs(csa['MEAN across slices'] - csa_truevalue)
    if csa_error > th_csa_error:
        status = 99
        output += '\nWARNING: CSA_ERROR = ' + str(csa_error) + ' < ' + str(th_csa_error)

    # compute CSA on rotated image
    cmd = 'sct_process_segmentation -i data_seg_reg.nii.gz' \
          + ' -p csa' \
          + ' -size 0' \
          + ' -r 0' \
          + ' -v 1' \
          + ' -z 5:23' \
          + ' -ofolder csa_rot'
    output = '\n====================================================================================================\n' + cmd + '\n====================================================================================================\n\n'  # copy command
    status, o = sct.run(cmd, 0)
    output += o
    # check integrity
    csa = pickle.load(open('csa_rot/csa_mean.pickle', 'rb'))
    csa_error = np.abs(csa['MEAN across slices'] - csa_truevalue)
    if csa_error > th_csa_error:
        status = 99
        output += '\nWARNING: CSA_ERROR = ' + str(csa_error) + ' < ' + str(th_csa_error)

    # transform results into Pandas structure
    duration = time.time() - time_start
    results = DataFrame(data={'status': int(status), 'output': output, 'csa_error': csa_error, 'duration': duration}, index=[path_data])
    return status, output, results


if __name__ == "__main__":
    # call main function
    test()
