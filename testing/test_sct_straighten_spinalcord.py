#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_sctraighten_spinalcord script
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

import sys, io, os

from pandas import DataFrame

import sct_utils as sct


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2.nii.gz -s t2/t2_seg.nii.gz -param accuracy_results=1']

    param_test.fname_segmentation = 't2/t2_seg.nii.gz'
    param_test.th_result_dist_max = 2.0
    param_test.th_result_rmse = 1.0
    param_test.th_dice = 0.9


    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """

    try:
        # extraction of results
        output_split = param_test.output.split('Maximum x-y error = ')[1].split(' mm')
        result_dist_max = float(output_split[0])
        result_rmse = float(output_split[1].split('Accuracy of straightening (MSE) = ')[1])
        duration_accuracy_results = float(param_test.output.split('\nincluding ')[1].split(' s')[0])

        # integrity testing
        if result_dist_max > param_test.th_result_dist_max:
            param_test.status = 99
            param_test.output += '\nWARNING: Maximum x-y error = ' + str(result_dist_max) + ' < ' + str(param_test.th_result_dist_max)
        if result_rmse > param_test.th_result_rmse:
            param_test.status = 99
            param_test.output += '\nWARNING: RMSE = ' + str(result_rmse) + ' < ' + str(param_test.th_result_rmse)

        # apply curved2straight, then straight2curve, then compared results
        path_input, file_input, ext_input = sct.extract_fname(param_test.file_input)
        sct.run('sct_apply_transfo -i ' + os.path.join(param_test.path_data, param_test.fname_segmentation) + ' -d ' + os.path.join(param_test.path_output, file_input) + '_straight' + ext_input + ' -w ' + os.path.join(param_test.path_output, 'warp_curve2straight.nii.gz') + ' -o ' + os.path.join(param_test.path_output, 'tmp_seg_straight.nii.gz') + ' -x linear', 0)
        sct.run('sct_apply_transfo -i ' + os.path.join(param_test.path_output, 'tmp_seg_straight.nii.gz') + ' -d ' + os.path.join(param_test.path_data, param_test.fname_segmentation) + ' -w ' + os.path.join(param_test.path_output, 'warp_straight2curve.nii.gz') + ' -o ' + os.path.join(param_test.path_output, 'tmp_seg_straight_curved.nii.gz') + ' -x nn',0)

        # threshold and binarize
        sct.run('sct_maths -i ' + os.path.join(param_test.path_output, 'tmp_seg_straight_curved.nii.gz') + ' -bin 0.5 -o ' + os.path.join(param_test.path_output, 'tmp_seg_straight_curved.nii.gz'), 0)

        # compute DICE
        cmd = 'sct_dice_coefficient -i ' + os.path.join(param_test.path_output, 'tmp_seg_straight_curved.nii.gz') + ' -d ' + os.path.join(param_test.path_data, param_test.fname_segmentation)
        status2, output2 = sct.run(cmd, 0)
        # parse output and compare to acceptable threshold
        result_dice = float(output2.split('3D Dice coefficient = ')[1].split('\n')[0])

        if result_dice < param_test.th_dice:
            param_test.status = 99
            param_test.output += '\nWARNING: DICE = ' + str(result_dice) + ' < ' + str(param_test.th_dice)

        # transform results into Pandas structure
        param_test.results = DataFrame(data={'status': param_test.status, 'output': param_test.output, 'rmse': result_rmse, 'dist_max': result_dist_max,
                                             'dice': result_dice, 'duration': param_test.duration, 'duration_accuracy_results': duration_accuracy_results},
                                       index=[param_test.path_data])

    except Exception as e:
        param_test.status = 99
        param_test.output += 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        param_test.output += str(e)

    return param_test

