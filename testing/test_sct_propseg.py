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
from sct_testing import write_to_log_file, init_testing
from msct_parser import Parser
from pandas import DataFrame
import os.path
import time, random
from msct_image import Image, compute_dice


def test(param):

    # initialization
    default_args = '-i t2/t2.nii.gz -c t2'  # default parameters
    dice_threshold = 0.9
    verbose = 0
    output = ''

    # check if isct_propseg compatibility
    status_isct_propseg, output_isct_propseg = commands.getstatusoutput('isct_propseg')
    isct_propseg_version = output_isct_propseg.split('\n')[0]
    if isct_propseg_version != 'sct_propseg - Version 1.1 (2015-03-24)':
        status = 99
        output += '\nERROR: isct_propseg does not seem to be compatible with your system or is no up-to-date... Please contact SCT administrators.'
        return status, output, DataFrame(data={'status': status, 'output': output}, index=[path_data])

    # assign default params
    if not param.args:
        param.args = default_args

    # initialize testing
    param = init_testing(param)

    # Extract contrast
    contrast = ''
    input_filename = ''
    if dict_param['-i'][0] == '/':
        dict_param['-i'] = dict_param['-i'][1:]
    input_split = dict_param['-i'].split('/')
    if len(input_split) == 2:
        contrast = input_split[0]
        input_filename = input_split[1]
    else:
        input_filename = input_split[0]
    if not contrast:  # if no contrast folder, send error.
        status = 1
        output += '\nERROR: when extracting the contrast folder from input file in command line: ' + dict_param['-i'] + ' for ' + path_data
        return status, output, DataFrame(data={'status': status, 'output': output, 'dice_segmentation': float('nan')}, index=[path_data])

    # Check if input files exist
    if not (os.path.isfile(dict_param_with_path['-i'])):
        status = 200
        output += '\nERROR: the file(s) provided to test function do not exist in folder: ' + path_data
        write_to_log_file(fname_log, output, 'w')
        return status, output, DataFrame(
            data={'status': status, 'output': output, 'dice_segmentation': float('nan')}, index=[path_data])

    # Check if ground truth files exist
    if not os.path.isfile(path_data + contrast + '/' + contrast + '_seg_manual.nii.gz'):
        status = 201
        output += '\nERROR: the file *_labeled_center_manual.nii.gz does not exist in folder: ' + path_data
        write_to_log_file(fname_log, output, 'w')
        return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])

    # run command
    cmd = 'sct_propseg ' + param_with_path
    output += '\n====================================================================================================\n'\
             + cmd + \
             '\n====================================================================================================\n\n'  # copy command
    time_start = time.time()
    try:
        status, o = sct.run(cmd, 0)
    except:
        status, o = 1, 'ERROR: Function crashed!'
    output += o
    duration = time.time() - time_start

    # extract name of manual segmentation
    # by convention, manual segmentation are called inputname_seg_manual.nii.gz where inputname is the filename
    # of the input image
    segmentation_filename = path_output + sct.add_suffix(input_filename, '_seg')
    manual_segmentation_filename = path_data + contrast + '/' + sct.add_suffix(input_filename, '_seg_manual')

    dice_segmentation = float('nan')

    # if command ran without error, test integrity
    if status == 0:
        # compute dice coefficient between generated image and image from database
        dice_segmentation = compute_dice(Image(segmentation_filename), Image(manual_segmentation_filename), mode='3d', zboundaries=False)

        if dice_segmentation < dice_threshold:
            status = 99

    # transform results into Pandas structure
    results = DataFrame(data={'status': status, 'output': output, 'dice_segmentation': dice_segmentation, 'duration [s]': duration}, index=[path_data])

    sys.stdout.close()
    sys.stdout = stdout_orig

    # write log file
    write_to_log_file(fname_log, output, mode='r+', prepend=True)

    return status, output, results


if __name__ == "__main__":
    # call main function
    test()