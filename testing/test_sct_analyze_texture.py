#!/usr/bin/env python
#########################################################################################
#
# Test function sct_analyze_texture
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley
# modified: 2017/06/18
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
import commands
import sct_analyze_texture
from sct_testing import write_to_log_file
from msct_parser import Parser
from pandas import DataFrame
import os.path
import time, random
from copy import deepcopy
from msct_image import Image, compute_dice


def test(path_data='', parameters=''):

    # initialization
    verbose = 0
    output = ''

    # # check if isct_propseg compatibility
    # status_isct_propseg, output_isct_propseg = commands.getstatusoutput('isct_propseg')
    # isct_propseg_version = output_isct_propseg.split('\n')[0]
    # if isct_propseg_version != 'sct_propseg - Version 1.1 (2015-03-24)':
    #     status = 99
    #     output += '\nERROR: isct_propseg does not seem to be compatible with your system or is no up-to-date... Please contact SCT administrators.'
    #     return status, output, DataFrame(data={'status': status, 'output': output}, index=[path_data])

    # parameters
    if not parameters:
        parameters = '-i t2/t2.nii.gz -s t2/t2_seg.nii.gz'

    # retrieve flags
    try:
        parser = sct_analyze_texture.get_parser()
        dict_param = parser.parse(parameters.split(), check_file_exist=False)
        dict_param_with_path = parser.add_path_to_file(deepcopy(dict_param), path_data, input_file=True)
        param_with_path = parser.dictionary_to_string(dict_param_with_path)
    # in case not all mandatory flags are filled
    except SyntaxError as err:
        print err
        status = 1
        output = err
        return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])

    import time, random
    subject_folder = path_data.split('/')
    if subject_folder[-1] == '' and len(subject_folder) > 1:
        subject_folder = subject_folder[-2]
    else:
        subject_folder = subject_folder[-1]
    path_output = sct.slash_at_the_end('sct_analyze_texture_' + subject_folder + '_' + time.strftime("%y%m%d%H%M%S") + '_' + str(random.randint(1, 1000000)), slash=1)
    param_with_path += ' -ofolder ' + path_output
    sct.create_folder(path_output)

    # log file
    import sys
    fname_log = path_output + 'output.log'
    stdout_log = file(fname_log, 'w')
    # redirect to log file
    stdout_orig = sys.stdout
    sys.stdout = stdout_log

    # Check if input files exist
    if not (os.path.isfile(dict_param_with_path['-i']) and
            os.path.isfile(dict_param_with_path['-s'])):
        status = 200
        output += '\nERROR: the file(s) provided to test function do not exist in folder: ' + path_data
        write_to_log_file(fname_log, output, 'w')
        return status, output, DataFrame(
            data={'status': status, 'output': output, 'dice_segmentation': float('nan')}, index=[path_data])

    # run command
    cmd = 'sct_analyze_texture ' + param_with_path
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

    # transform results into Pandas structure
    results = DataFrame(data={'status': status, 'output': output, 'duration [s]': duration}, index=[path_data])

    sys.stdout.close()
    sys.stdout = stdout_orig

    # write log file
    write_to_log_file(fname_log, output, mode='r+', prepend=True)

    return status, output, results


if __name__ == "__main__":
    # call main function
    test()