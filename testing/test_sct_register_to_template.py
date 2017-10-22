#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_to_template
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from pandas import DataFrame
from msct_image import Image, compute_dice
import sct_apply_transfo


def init(param_test):
    """
    Initialize class: param_test
    """

    # initialization
    default_args = ['-i t2/t2.nii.gz -l t2/labels.nii.gz -s t2/t2_seg.nii.gz -param step=1,type=seg,algo=centermassrot,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=5,metric=MeanSquares -t template -r 0 -igt template/template/PAM50_small_cord.nii.gz']
    param_test.dice_threshold = 0.9

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """

    # apply transformation to binary mask: template --> anat
    sct_apply_transfo.main(args=[
        '-i', param_test.fname_gt,
        '-d', param_test.dict_args_with_path['-s'],
        '-w', 'warp_template2anat.nii.gz',
        '-o', 'test_template2anat.nii.gz',
        '-x', 'nn',
        '-v', '0'])

    # apply transformation to binary mask: anat --> template
    sct_apply_transfo.main(args=[
        '-i', param_test.dict_args_with_path['-s'],
        '-d', param_test.fname_gt,
        '-w', 'warp_anat2template.nii.gz',
        '-o', 'test_anat2template.nii.gz',
        '-x', 'nn',
        '-v', '0'])

    # compute dice coefficient between template segmentation warped to anat and segmentation from anat
    im_seg = Image(param_test.dict_args_with_path['-s'])
    im_template_seg_reg = Image('test_template2anat.nii.gz')
    dice_template2anat = compute_dice(im_seg, im_template_seg_reg, mode='3d', zboundaries=True)
    # check
    param_test.output += 'Dice[seg,template_seg_reg]: '+str(dice_template2anat)
    if dice_template2anat > param_test.dice_threshold:
        param_test.output += '\n--> PASSED'
    else:
        param_test.status = 99
        param_test.output += '\n--> FAILED'

    # compute dice coefficient between anat segmentation warped to template and segmentation from template
    im_seg_reg = Image('test_anat2template.nii.gz')
    im_template_seg = Image(param_test.fname_gt)
    dice_anat2template = compute_dice(im_seg_reg, im_template_seg, mode='3d', zboundaries=True)
    # check
    param_test.output += '\n\nDice[seg_reg,template_seg]: '+str(dice_anat2template)
    if dice_anat2template > param_test.dice_threshold:
        param_test.output += '\n--> PASSED'
    else:
        param_test.status = 99
        param_test.output += '\n--> FAILED'

    # transform results into Pandas structure
    param_test.results = DataFrame(
        index=[param_test.path_data],
        data={'status': param_test.status,
              'output': param_test.output,
              'dice_template2anat': dice_template2anat,
              'dice_anat2template': dice_anat2template,
              'duration [s]': param_test.duration})

    return param_test

#
#     cmd = 'sct_dice_coefficient -i ' + param_test.dict_param_with_path['-s'] + ' -d test_template2anat.nii.gz'
#     status1, output1 = sct.run(cmd, verbose)
#     # parse output and compare to acceptable threshold
#     dice_template2anat = float(output1.split('3D Dice coefficient = ')[1].split('\n')[0])
#     if dice_template2anat < dice_threshold:
#         status1 = 99
#     # compute dice coefficient between segmentation from anat warped into template and template segmentation
#     # N.B. here we use -bmax because the FOV of the anat is smaller than the template
#     cmd = 'sct_dice_coefficient -i ' + fname_template_seg + ' -d ' + path_output + 'test_anat2template.nii.gz -bmax 1'
#     status2, output2 = sct.run(cmd, verbose)
#     # parse output and compare to acceptable threshold
#     dice_anat2template = float(output2.split('3D Dice coefficient = ')[1].split('\n')[0])
#     if dice_anat2template < dice_threshold:
#         status2 = 99
#     # check if at least one integrity status was equal to 99
#     if status1 == 99 or status2 == 99:
#         status = 99
#
#     # concatenate outputs
#     output = output + output1 + output2
#
#     # transform results into Pandas structure
#     results = DataFrame(data={'status': int(status), 'output': output, 'dice_template2anat': dice_template2anat,
#                           'dice_anat2template': dice_anat2template, 'duration [s]': duration}, index=[path_data])
#
#     if abs(mtr_result - param_test.mtr_groundtruth) < param_test.threshold_diff:
#             param_test.output += '--> PASSED'
#         else:
#             param_test.status = 99
#             param_test.output += '--> FAILED'
#     except Exception as err:
#         param_test.output += str(err)
#         raise
#     return param_test
#
#
#
#
# def test(path_data='', parameters=''):
#     verbose = 0
#     dice_threshold = 0.9
#     add_path_for_template = False  # if absolute path or no path to template is provided, then path to data should not be added.
#
#     # initializations
#     dice_template2anat = float('NaN')
#     dice_anat2template = float('NaN')
#     output = ''
#
#     if not parameters:
#         parameters = '-i t2/t2.nii.gz -l t2/labels.nii.gz -s t2/t2_seg.nii.gz ' \
#                      '-param step=1,type=seg,algo=centermassrot,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=5,metric=MeanSquares ' \
#                      '-t template/ -r 0'
#         add_path_for_template = True  # in this case, path to data should be added
#
#     parser = sct_register_to_template.get_parser()
#     dict_param = parser.parse(parameters.split(), check_file_exist=False)
#     if add_path_for_template:
#         dict_param_with_path = parser.add_path_to_file(deepcopy(dict_param), path_data, input_file=True)
#     else:
#         dict_param_with_path = parser.add_path_to_file(deepcopy(dict_param), path_data, input_file=True, do_not_add_path=['-t'])
#     param_with_path = parser.dictionary_to_string(dict_param_with_path)
#
#     # Check if input files exist
#     if not (os.path.isfile(dict_param_with_path['-i']) and
#             os.path.isfile(dict_param_with_path['-l']) and
#             os.path.isfile(dict_param_with_path['-s'])):
#         status = 200
#         output = 'ERROR: the file(s) provided to test function do not exist in folder: ' + path_data
#         return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
#         # return status, output, DataFrame(
#         #     data={'status': status, 'output': output,
#         #           'dice_template2anat': float('nan'), 'dice_anat2template': float('nan')},
#         #     index=[path_data])
#
#     # if template is not specified, use default
#     # if not os.path.isdir(dict_param_with_path['-t']):
#     #     status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
#     #     dict_param_with_path['-t'] = path_sct + default_template
#     #     param_with_path = parser.dictionary_to_string(dict_param_with_path)
#
#     # get contrast folder from -i option.
#     # We suppose we can extract it as the first object when spliting with '/' delimiter.
#     contrast_folder = ''
#     input_filename = ''
#     if dict_param['-i'][0] == '/':
#         dict_param['-i'] = dict_param['-i'][1:]
#     input_split = dict_param['-i'].split('/')
#     if len(input_split) == 2:
#         contrast_folder = input_split[0] + '/'
#         input_filename = input_split[1]
#     else:
#         input_filename = input_split[0]
#     if not contrast_folder:  # if no contrast folder, send error.
#         status = 201
#         output = 'ERROR: when extracting the contrast folder from input file in command line: ' + dict_param['-i'] + ' for ' + path_data
#         return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
#         # return status, output, DataFrame(
#         #     data={'status': status, 'output': output, 'dice_template2anat': float('nan'), 'dice_anat2template': float('nan')}, index=[path_data])
#
#     # create output path
#     # TODO: create function for that
#     import time, random
#     subject_folder = path_data.split('/')
#     if subject_folder[-1] == '' and len(subject_folder) > 1:
#         subject_folder = subject_folder[-2]
#     else:
#         subject_folder = subject_folder[-1]
#     path_output = sct.slash_at_the_end('sct_register_to_template_' + subject_folder + '_' + time.strftime("%y%m%d%H%M%S") + '_'+str(random.randint(1, 1000000)), slash=1)
#     param_with_path += ' -ofolder ' + path_output
#     sct.create_folder(path_output)
#
#     # log file
#     # TODO: create function for that
#     import sys
#     fname_log = path_output + 'output.log'
#
#     sct.pause_stream_logger()
#     file_handler = sct.add_file_handler_to_logger(filename=fname_log, mode='w', log_format="%(message)s")
#     #
#     # stdout_log = file(fname_log, 'w')
#     # redirect to log file
#     # stdout_orig = sys.stdout
#     # sys.stdout = stdout_log
#
#     cmd = 'sct_register_to_template ' + param_with_path
#     output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
#     time_start = time.time()
#     try:
#         status, o = sct.run(cmd, verbose)
#     except:
#         status, o = 1, 'ERROR: Function crashed!'
#     output += o
#     duration = time.time() - time_start
#
#     # if command ran without error, test integrity
#     if status == 0:
#
#     sct.log.info(output)
#     sct.remove_handler(file_handler)
#     sct.start_stream_logger()
#
#     return status, output, results
#
#
# if __name__ == "__main__":
#     # call main function
#     test()
#
