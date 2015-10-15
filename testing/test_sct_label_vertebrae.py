#!/usr/bin/env python
#########################################################################################
#
# Test function sct_detect_vertebral_levels
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
# from msct_parser import Parser
import sct_label_vertebrae
from pandas import DataFrame
import os.path
# import commands
from msct_image import Image
from numpy import where
# # append path that contains scripts, to be able to load modules
# status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# sys.path.append(path_sct + '/scripts')

def test(path_data='', parameters=''):

    file_init_label_vertebrae = 'init_label_vertebrae.txt'
    if not parameters:
        parameters = '-i t2/t2.nii.gz -seg t2/t2_seg.nii.gz -o t2_seg_labeled.nii.gz'

    parser = sct_label_vertebrae.get_parser()
    dict_param = parser.parse(parameters.split(), check_file_exist=False)
    dict_param_with_path = parser.add_path_to_file(dict_param, path_data, input_file=True)
    param_with_path = parser.dictionary_to_string(dict_param_with_path)

    # Check if input files exist
    if not (os.path.isfile(dict_param_with_path['-i']) and os.path.isfile(dict_param_with_path['-seg'])):
        status = 200
        output = 'ERROR: the file(s) provided to test function do not exist in folder: ' + path_data
        return status, output, DataFrame(data={'status': status, 'output': output, 'mse': float('nan')}, index=[path_data])

    # add initialization parameter contained in file: init_label_vertebrae.txt
    if not os.path.isfile(path_data+'t2/'+file_init_label_vertebrae):
        status = 200
        output = 'ERROR: the file init_label_vertebrae.txt does not exist in folder: ' + path_data
        return status, output, DataFrame(data={'status': status, 'output': output, 'mse': float('nan')}, index=[path_data])
    else:
        file = open(path_data+'t2/'+file_init_label_vertebrae, 'r')
        param_with_path += ' '+file.read().replace('\n', '')

    cmd = 'sct_label_vertebrae ' + param_with_path
    output = '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    status, o = sct.run(cmd, 0)
    output += o

    # initialization of results: must be NaN if test fails
    result_mse = float('nan'), float('nan')

    if status == 0:
        # open output data
        data_result = Image('t2_seg_labeled.nii.gz').data
        data_goldstandard = Image(path_data+'t2/t2_seg_labeled.nii.gz').data
        # compute MSE
        ind_nonzero = where(data_result>0)  # only get non-zero values to obtain meaningful MSE calculations
        result_mse = ((data_result[ind_nonzero] - data_goldstandard[ind_nonzero]) ** 2).mean()
        # check if MSE is superior to threshold
        if result_mse > 0.0:
            status = 99
            output += '\nResulting image differs from gold-standard.'

    # transform results into Pandas structure
    results = DataFrame(data={'status': status, 'output': output, 'mse': result_mse}, index=[path_data])

    return status, output, results

if __name__ == "__main__":
    # call main function
    test()
