#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_to_template script
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

import commands
import sct_utils as sct
import sct_register_to_template
from sct_testing import write_to_log_file
from pandas import DataFrame
import os.path
from copy import deepcopy
from sct_warp_template import get_file_label


def test(path_data='', parameters=''):
    verbose = 0
    dice_threshold = 0.9
    add_path_for_template = False  # if absolute path or no path to template is provided, then path to data should not be added.

    # initializations
    dice_template2anat = float('NaN')
    dice_anat2template = float('NaN')
    output = ''

    if not parameters:
        parameters = '-i t2/t2.nii.gz -l t2/labels.nii.gz -s t2/t2_seg.nii.gz ' \
                     '-param step=1,type=seg,algo=centermassrot,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=5,metric=MeanSquares ' \
                     '-t template/ -r 0'
        add_path_for_template = True  # in this case, path to data should be added

    parser = sct_register_to_template.get_parser()
    dict_param = parser.parse(parameters.split(), check_file_exist=False)
    if add_path_for_template:
        dict_param_with_path = parser.add_path_to_file(deepcopy(dict_param), path_data, input_file=True)
    else:
        dict_param_with_path = parser.add_path_to_file(deepcopy(dict_param), path_data, input_file=True, do_not_add_path=['-t'])
    param_with_path = parser.dictionary_to_string(dict_param_with_path)

    # Check if input files exist
    if not (os.path.isfile(dict_param_with_path['-i']) and
            os.path.isfile(dict_param_with_path['-l']) and
            os.path.isfile(dict_param_with_path['-s'])):
        status = 200
        output = 'ERROR: the file(s) provided to test function do not exist in folder: ' + path_data
        return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
        # return status, output, DataFrame(
        #     data={'status': status, 'output': output,
        #           'dice_template2anat': float('nan'), 'dice_anat2template': float('nan')},
        #     index=[path_data])

    # if template is not specified, use default
    # if not os.path.isdir(dict_param_with_path['-t']):
    #     status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    #     dict_param_with_path['-t'] = path_sct + default_template
    #     param_with_path = parser.dictionary_to_string(dict_param_with_path)

    # get contrast folder from -i option.
    # We suppose we can extract it as the first object when spliting with '/' delimiter.
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
        output = 'ERROR: when extracting the contrast folder from input file in command line: ' + dict_param['-i'] + ' for ' + path_data
        return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
        # return status, output, DataFrame(
        #     data={'status': status, 'output': output, 'dice_template2anat': float('nan'), 'dice_anat2template': float('nan')}, index=[path_data])

    # create output path
    # TODO: create function for that
    import time, random
    subject_folder = path_data.split('/')
    if subject_folder[-1] == '' and len(subject_folder) > 1:
        subject_folder = subject_folder[-2]
    else:
        subject_folder = subject_folder[-1]
    path_output = sct.slash_at_the_end('sct_register_to_template_' + subject_folder + '_' + time.strftime("%y%m%d%H%M%S") + '_'+str(random.randint(1, 1000000)), slash=1)
    param_with_path += ' -ofolder ' + path_output
    sct.create_folder(path_output)

    # log file
    # TODO: create function for that
    import sys
    fname_log = path_output + 'output.log'

    sct.pause_stream_logger()
    file_handler = sct.add_file_handler_to_logger(filename=fname_log, mode='w', log_format="%(message)s")
    #
    # stdout_log = file(fname_log, 'w')
    # redirect to log file
    # stdout_orig = sys.stdout
    # sys.stdout = stdout_log

    cmd = 'sct_register_to_template ' + param_with_path
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    time_start = time.time()
    try:
        status, o = sct.run(cmd, verbose)
    except:
        status, o = 1, 'ERROR: Function crashed!'
    output += o
    duration = time.time() - time_start

    # if command ran without error, test integrity
    if status == 0:
        # get filename_template_seg
        fname_template_seg = get_file_label(sct.slash_at_the_end(dict_param_with_path['-t'], 1) + 'template/', 'spinal cord', output='filewithpath')
        # apply transformation to binary mask: template --> anat
        sct.run(
            'sct_apply_transfo -i ' + fname_template_seg +
            ' -d ' + dict_param_with_path['-s'] +
            ' -w ' + path_output + 'warp_template2anat.nii.gz' +
            ' -o ' + path_output + 'test_template2anat.nii.gz -x nn', verbose)
        # apply transformation to binary mask: anat --> template
        sct.run(
            'sct_apply_transfo -i ' + dict_param_with_path['-s'] +
            ' -d ' + fname_template_seg +
            ' -w ' + path_output + 'warp_anat2template.nii.gz' +
            ' -o ' + path_output + 'test_anat2template.nii.gz -x nn', verbose)
        # compute dice coefficient between template segmentation warped into anat and segmentation from anat
        cmd = 'sct_dice_coefficient -i ' + dict_param_with_path['-s'] + ' -d ' + path_output + 'test_template2anat.nii.gz'
        status1, output1 = sct.run(cmd, verbose)
        # parse output and compare to acceptable threshold
        dice_template2anat = float(output1.split('3D Dice coefficient = ')[1].split('\n')[0])
        if dice_template2anat < dice_threshold:
            status1 = 99
        # compute dice coefficient between segmentation from anat warped into template and template segmentation
        # N.B. here we use -bmax because the FOV of the anat is smaller than the template
        cmd = 'sct_dice_coefficient -i ' + fname_template_seg + ' -d ' + path_output + 'test_anat2template.nii.gz -bmax 1'
        status2, output2 = sct.run(cmd, verbose)
        # parse output and compare to acceptable threshold
        dice_anat2template = float(output2.split('3D Dice coefficient = ')[1].split('\n')[0])
        if dice_anat2template < dice_threshold:
            status2 = 99
        # check if at least one integrity status was equal to 99
        if status1 == 99 or status2 == 99:
            status = 99

        # concatenate outputs
        output = output + output1 + output2

    # transform results into Pandas structure
    results = DataFrame(data={'status': int(status), 'output': output, 'dice_template2anat': dice_template2anat, 'dice_anat2template': dice_anat2template, 'duration [s]': duration}, index=[path_data])

    sct.log.info(output)
    sct.remove_handler(file_handler)
    sct.start_stream_logger()

    return status, output, results


if __name__ == "__main__":
    # call main function
    test()