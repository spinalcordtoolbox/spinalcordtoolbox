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

import sct_utils as sct
import sct_straighten_spinalcord
from pandas import DataFrame
import os.path


def test(path_data='', parameters=''):

    # initializations
    result_rmse = float('NaN')
    result_dist_max = float('NaN')
    result_dice = float('NaN')

    if not parameters:
        parameters = '-i t2/t2.nii.gz -s t2/t2_seg.nii.gz -qc 0'

    parser = sct_straighten_spinalcord.get_parser()
    dict_param = parser.parse(parameters.split(), check_file_exist=False)
    dict_param_with_path = parser.add_path_to_file(dict_param, path_data, input_file=True)
    param_with_path = parser.dictionary_to_string(dict_param_with_path)

    # Check if input files exist
    if not (os.path.isfile(dict_param_with_path['-i']) and os.path.isfile(dict_param_with_path['-s'])):
        status = 200
        output = 'ERROR: the file(s) provided to test function do not exist in folder: ' + path_data
        return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
        # return status, output, DataFrame(data={'status': status, 'output': output, 'mse': float('nan'), 'dist_max': float('nan')}, index=[path_data])

    # create output folder to deal with multithreading (i.e., we don't want to have outputs from several subjects in the current directory)
    import time, random
    subject_folder = path_data.split('/')
    if subject_folder[-1] == '' and len(subject_folder) > 1:
        subject_folder = subject_folder[-2]
    else:
        subject_folder = subject_folder[-1]
    path_output = sct.slash_at_the_end('sct_straighten_spinalcord_' + subject_folder + '_' + time.strftime("%y%m%d%H%M%S") + '_'+str(random.randint(1, 1000000)), slash=1)
    param_with_path += ' -ofolder ' + path_output

    # run command
    cmd = 'sct_straighten_spinalcord ' + param_with_path
    output = '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    time_start = time.time()
    try:
        status, o = sct.run(cmd, 0)
    except:
        status, o = 1, 'ERROR: Function crashed!'
    output += o
    duration = time.time() - time_start

    # initialization of results: must be NaN if test fails
    result_rmse, result_dist_max = float('nan'), float('nan')
    if status == 0:
        # extraction of results
        output_split = output.split('Maximum x-y error = ')[1].split(' mm')
        result_dist_max = float(output_split[0])
        result_rmse = float(output_split[1].split('Accuracy of straightening (MSE) = ')[1])
        # integrity testing
        th_result_dist_max = 2.0
        if result_dist_max > th_result_dist_max:
            status = 99
            output += '\nWARNING: Maximum x-y error = '+str(result_dist_max)+' < '+str(th_result_dist_max)
        th_result_rmse = 1.0
        if result_rmse > th_result_rmse:
            status = 99
            output += '\nWARNING: RMSE = '+str(result_rmse)+' < '+str(th_result_rmse)
        # apply curved2straight, then straight2curve, then compared results
        path_input, file_input, ext_input = sct.extract_fname(dict_param_with_path['-i'])
        sct.run('sct_apply_transfo -i '+dict_param_with_path['-s']+' -d '+path_output+file_input+'_straight'+ext_input+' -w '+path_output+'warp_curve2straight.nii.gz -o '+path_output+'tmp_seg_straight.nii.gz -x linear', 0)
        sct.run('sct_apply_transfo -i '+path_output+'tmp_seg_straight.nii.gz -d '+dict_param_with_path['-s']+' -w '+path_output+'warp_straight2curve.nii.gz -o '+path_output+'tmp_seg_straight_curved.nii.gz -x nn', 0)
        # threshold and binarize
        sct.run('sct_maths -i '+path_output+'tmp_seg_straight_curved.nii.gz -thr 0.5 -o '+path_output+'tmp_seg_straight_curved.nii.gz', 0)
        sct.run('sct_maths -i '+path_output+'tmp_seg_straight_curved.nii.gz -bin -o '+path_output+'tmp_seg_straight_curved.nii.gz', 0)
        # compute DICE
        cmd = 'sct_dice_coefficient -i '+path_output+'tmp_seg_straight_curved.nii.gz -d ' + dict_param_with_path['-s']
        status2, output2 = sct.run(cmd, 0)
        # parse output and compare to acceptable threshold
        result_dice = float(output2.split('3D Dice coefficient = ')[1].split('\n')[0])
        th_dice = 0.9
        if result_dice < th_dice:
            status = 99
            output += '\nWARNING: DICE = '+str(result_dice)+' < '+str(th_dice)

    # transform results into Pandas structure
    results = DataFrame(data={'status': int(status), 'output': output, 'rmse': result_rmse, 'dist_max': result_dist_max, 'dice': result_dice, 'duration': duration}, index=[path_data])

    return status, output, results

if __name__ == "__main__":
    # call main function
    test()