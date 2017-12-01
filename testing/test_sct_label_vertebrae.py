#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_label_vertebrae
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import shutil

def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2.nii.gz -s t2/t2_seg.nii.gz -c t2 -initfile t2/init_label_vertebrae.txt -t template/']
    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    param_test.output += '\nNot implemented.'
    # TODO: implement integrity testing
    return param_test


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
#
# import sct_utils as sct
# from sct_testing import write_to_log_file
# from sct_label_utils import ProcessLabels
# from numpy import linalg
# from math import sqrt
# import sct_label_vertebrae
# from pandas import DataFrame
# import os.path
# from copy import deepcopy
#
#
# def test(path_data='', parameters=''):
#
#     # initializations
#     output = ''
#     file_init_label_vertebrae = 'init_label_vertebrae.txt'
#     rmse = float('NaN')
#     max_dist = float('NaN')
#     diff_manual_result = float('NaN')
#
#     if not parameters:
#         parameters = '-i t2/t2.nii.gz -s t2/t2_seg.nii.gz -c t2 -initfile t2/init_label_vertebrae.txt'
#
#     # retrieve flags
#     try:
#         parser = sct_label_vertebrae.get_parser()
#         dict_param = parser.parse(parameters.split(), check_file_exist=False)
#         dict_param_with_path = parser.add_path_to_file(deepcopy(dict_param), path_data, input_file=True)
#         # update template path because the previous command wrongly adds path to testing data
#         dict_param_with_path['-t'] = dict_param['-t']
#         param_with_path = parser.dictionary_to_string(dict_param_with_path)
#     # in case not all mandatory flags are filled
#     except SyntaxError as err:
#         print err
#         status = 1
#         output = err
#         return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
#
#     # create output folder to deal with multithreading (i.e., we don't want to have outputs from several subjects in the current directory)
#     import time, random
#     subject_folder = path_data.split('/')
#     if subject_folder[-1] == '' and len(subject_folder) > 1:
#         subject_folder = subject_folder[-2]
#     else:
#         subject_folder = subject_folder[-1]
#     path_output = sct.slash_at_the_end('sct_label_vertebrae_' + subject_folder + '_' + time.strftime("%y%m%d%H%M%S") + '_'+str(random.randint(1, 1000000)), slash=1)
#     os.mkdir(path_output)
#     param_with_path += ' -ofolder ' + path_output
#     # log file
#     fname_log = path_output + 'output.log'
#
#     # Extract contrast
#     contrast = ''
#     if dict_param['-i'][0] == '/':
#         dict_param['-i'] = dict_param['-i'][1:]
#     input_split = dict_param['-i'].split('/')
#     if len(input_split) == 2:
#         contrast = input_split[0]
#     if not contrast:  # if no contrast folder, send error.
#         status = 1
#         output += '\nERROR: when extracting the contrast folder from input file in command line: ' + dict_param['-i'] + ' for ' + path_data
#         write_to_log_file(fname_log, output, 'w')
#         return status, output, DataFrame(data={'status': status, 'output': output, 'dice_segmentation': float('nan')}, index=[path_data])
#
#     # Check if input files exist
#     if not os.path.isfile(dict_param_with_path['-i']):
#         status = 200
#         output += '\nERROR: This file does not exist: ' + dict_param_with_path['-i']
#         write_to_log_file(fname_log, output, 'w')
#         return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
#     if not os.path.isfile(dict_param_with_path['-s']):
#         status = 200
#         output += '\nERROR: This file does not exist: ' + dict_param_with_path['-s']
#         write_to_log_file(fname_log, output, 'w')
#         return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
#
#     # open ground truth
#     fname_labels_manual = path_data + contrast + '/' + contrast + '_labeled_center_manual.nii.gz'
#     try:
#         label_manual = ProcessLabels(fname_labels_manual)
#         list_label_manual = label_manual.image_input.getNonZeroCoordinates(sorting='value')
#     except:
#         status = 201
#         output += '\nERROR: cannot file: ' + fname_labels_manual
#         write_to_log_file(fname_log, output, 'w')
#         return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
#
#     cmd = 'sct_label_vertebrae ' + param_with_path
#     output = '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
#     time_start = time.time()
#     try:
#         status, o = sct.run(cmd, 0)
#     except:
#         status, o = 1, '\nERROR: Function crashed!'
#     output += o
#     duration = time.time() - time_start
#
#     # initialization of results: must be NaN if test fails
#     result_mse = float('nan'), float('nan')
#
#     if status == 0:
#         # copy input data (for easier debugging)
#         shutil.copy(dict_param_with_path['-i'], path_output)
#         # extract center of vertebral labels
#         path_seg, file_seg, ext_seg = sct.extract_fname(dict_param['-s'])
#         try:
#             sct.run('sct_label_utils -i '+os.path.join(path_output, file_seg+'_labeled.nii.gz) + ' -vert-body 0 -o '+ os.path.join(path_output, contrast + '_seg_labeled_center.nii.gz'), verbose=0)
#             label_results = ProcessLabels(os.path.join(path_output, contrast + '_seg_labeled_center.nii.gz'))
#             list_label_results = label_results.image_input.getNonZeroCoordinates(sorting='value')
#             # get dimension
#             # from msct_image import Image
#             # img = Image(os.path.join(path_output, contrast+'_seg_labeled.nii.gz'))
#             nx, ny, nz, nt, px, py, pz, pt = label_results.image_input.dim
#         except:
#             status = 1
#             output += '\nERROR: cannot open file: ' + os.path.join(path_output, contrast + '_seg_labeled.nii.gz')
#             write_to_log_file(fname_log, output, 'w')
#             return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
#
#         mse = 0.0
#         max_dist = 0.0
#         for coord_manual in list_label_manual:
#             for coord in list_label_results:
#                 if round(coord.value) == round(coord_manual.value):
#                     # Calculate MSE
#                     mse += (((coord_manual.x - coord.x)*px) ** 2 + ((coord_manual.y - coord.y)*py) ** 2 + ((coord_manual.z - coord.z)*pz) ** 2) / float(3)
#                     # Calculate distance (Frobenius norm)
#                     dist = linalg.norm([(coord_manual.x - coord.x)*px, (coord_manual.y - coord.y)*py, (coord_manual.z - coord.z)*pz])
#                     if dist > max_dist:
#                         max_dist = dist
#                     break
#         rmse = sqrt(mse / len(list_label_manual))
#         # calculate number of label mismatch
#         diff_manual_result = len(list_label_manual) - len(list_label_results)
#
#         # check if MSE is superior to threshold
#         th_rmse = 2
#         if rmse > th_rmse:
#             status = 99
#             output += '\nWARNING: RMSE = '+str(rmse)+' > '+str(th_rmse)
#         th_max_dist = 4
#         if max_dist > th_max_dist:
#             status = 99
#             output += '\nWARNING: Max distance = '+str(max_dist)+' > '+str(th_max_dist)
#         th_diff_manual_result = 3
#         if abs(diff_manual_result) > th_diff_manual_result:
#             status = 99
#             output += '\nWARNING: Diff manual-result = '+str(diff_manual_result)+' > '+str(th_diff_manual_result)
#
#     # transform results into Pandas structure
#     results = DataFrame(data={'status': int(status), 'output': output, 'rmse': rmse, 'max_dist': max_dist, 'diff_man': diff_manual_result, 'duration [s]': duration}, index=[path_data])
#
#     # write log file
#     write_to_log_file(fname_log, output, 'w')
#
#     return status, output, results
#
# if __name__ == "__main__":
#     # call main function
#     test()
