#!/usr/bin/env python
#########################################################################################
#
# Test function sct_propseg
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/09
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
import commands
import sct_propseg
from msct_parser import Parser
from pandas import DataFrame
import os.path
import time, random
from copy import deepcopy


def test(path_data='', parameters=''):
    verbose = 0

    # parameters
    if not parameters:
        parameters = '-i t2/t2.nii.gz -c t2'

    dice_threshold = 0.95

    parser = sct_propseg.get_parser()
    dict_param = parser.parse(parameters.split(), check_file_exist=False)
    dict_param_with_path = parser.add_path_to_file(deepcopy(dict_param), path_data, input_file=True)
    param_with_path = parser.dictionary_to_string(dict_param_with_path)

    # Check if input files exist
    if not (os.path.isfile(dict_param_with_path['-i'])):
        status = 200
        output = 'ERROR: the file(s) provided to test function do not exist in folder: ' + path_data
        return status, output, DataFrame(
            data={'status': status, 'output': output, 'dice_segmentation': float('nan')}, index=[path_data])

    contrast_folder = ''
    input_filename = ''
    if dict_param['-i'][0] == '/':
        dict_param['-i'] = dict_param['-i'][1:]
    input_split = dict_param['-i'].split('/')
    if len(input_split) == 2:
        contrast_folder = input_split[0] + '/'
        input_filename = input_split[1]
    else:
        input_filename = input_split[0]
    if not contrast_folder:  # if no contrast folder, send error.
        status = 201
        output = 'ERROR: when extracting the contrast folder from input file in command line: ' + dict_param[
            '-i'] + ' for ' + path_data
        return status, output, DataFrame(
            data={'status': status, 'output': output, 'dice_segmentation': float('nan')}, index=[path_data])

    import time, random
    subject_folder = path_data.split('/')
    if subject_folder[-1] == '' and len(subject_folder) > 1:
        subject_folder = subject_folder[-2]
    else:
        subject_folder = subject_folder[-1]
    path_output = sct.slash_at_the_end('sct_propseg_' + subject_folder + '_' + time.strftime("%y%m%d%H%M%S") + '_' + str(random.randint(1, 1000000)), slash=1)
    param_with_path += ' -ofolder ' + path_output

    # run command
    cmd = 'sct_propseg ' + param_with_path
    output = '\n====================================================================================================\n'\
             + cmd + \
             '\n====================================================================================================\n\n'  # copy command
    time_start = time.time()
    status, o = sct.run(cmd, verbose)
    output += o
    duration = time.time() - time_start

    # extract name of manual segmentation
    # by convention, manual segmentation are called inputname_seg_manual.nii.gz where inputname is the filename
    # of the input image
    segmentation_filename = path_output + sct.add_suffix(input_filename, '_seg')
    manual_segmentation_filename = path_data + contrast_folder + sct.add_suffix(input_filename, '_seg_manual')

    dice_segmentation = float('nan')

    # if command ran without error, test integrity
    if status == 0:
        # compute dice coefficient between generated image and image from database
        cmd = 'sct_dice_coefficient -i ' + segmentation_filename + ' -d ' + manual_segmentation_filename
        status, output = sct.run(cmd, verbose)
        # parse output and compare to acceptable threshold
        dice_segmentation = float(output.split('3D Dice coefficient = ')[1].split('\n')[0])
        if dice_segmentation < dice_threshold:
            status = 99

    # transform results into Pandas structure
    results = DataFrame(data={'status': status, 'output': output, 'dice_segmentation': dice_segmentation, 'duration [s]': duration}, index=[path_data])

    return status, output, results


if __name__ == "__main__":
    # call main function
    test()