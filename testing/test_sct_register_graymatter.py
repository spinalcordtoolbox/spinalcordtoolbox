#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_graymatter
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont, Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-t mt/label/ -w mt/warp_template2mt.nii.gz -gm mt/mt1_gmseg.nii.gz -wm mt/mt1_wmseg.nii.gz -manual-gm mt/mt1_gmseg_goldstandard.nii.gz -sc mt/mt1_seg.nii.gz -qc 0 -param step=1,type=seg,algo=centermassrot,metric=MeanSquares:step=2,type=im,algo=syn,metric=MeanSquares,iter=3,smooth=0,shrink=2']
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
# Test function sct_register_gm_multilabel
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# modified: 2015/11/16
#
# About the license: see the file LICENSE.TXT
#########################################################################################
#
#
# import commands
# import sys
# import os
# import time
# from pandas import DataFrame
# import sct_register_graymatter
# from msct_image import Image
# import sct_utils as sct
# from numpy import sum, mean
#
#
# def test(path_data, parameters=''):
#
#     if not parameters:
#         parameters = ' -t mt/label/ -w mt/warp_template2mt.nii.gz -gm mt/mt1_gmseg.nii.gz -wm mt/mt1_wmseg.nii.gz -manual-gm mt/mt1_gmseg_goldstandard.nii.gz -sc mt/mt1_seg.nii.gz -qc 0' #-d mt/mt0.nii.gz
#
#     parser = sct_register_graymatter.get_parser()
#     dict_param = parser.parse(parameters.split(), check_file_exist=False)
#     dict_param_with_path = parser.add_path_to_file(dict_param, path_data, input_file=True)
#     param_with_path = parser.dictionary_to_string(dict_param_with_path)
#
#     # Check if input files exist
#     if not (os.path.isfile(dict_param_with_path['-w']) and os.path.isfile(dict_param_with_path['-gm']) and os.path.isfile(dict_param_with_path['-wm']) and os.path.isfile(dict_param_with_path['-manual-gm']) and os.path.isfile(dict_param_with_path['-sc'])): #(os.path.isfile(dict_param_with_path['-d']) and
#         status = 200
#         output = 'ERROR: the file(s) provided to test function do not exist in folder: ' + path_data
#         return status, output, DataFrame(data={'status': status, 'output': output, 'dice_gm': float('nan'), 'diff_dc_gm': float('nan'), 'dice_wm': float('nan'), 'diff_dc_wm': float('nan'), 'hausdorff': float('nan'), 'diff_hd': float('nan'), 'med_dist': float('nan'), 'diff_md': float('nan'), 'duration_[s]': float('nan')}, index=[path_data])
#
#     import time, random
#     subject_folder = path_data.split('/')
#     if subject_folder[-1] == '' and len(subject_folder) > 1:
#         subject_folder = subject_folder[-2]
#     else:
#         subject_folder = subject_folder[-1]
#     path_output = sct.slash_at_the_end('sct_register_graymatter_' + subject_folder + '_' + time.strftime("%y%m%d%H%M%S") + '_'+str(random.randint(1, 1000000)), slash=1)
#     param_with_path += ' -ofolder ' + path_output
#
#     cmd = 'sct_register_graymatter ' + param_with_path
#     time_start = time.time()
#     status, output = sct.run(cmd, 0)
#     duration = time.time() - time_start
#
#
#     # initialization of results: must be NaN if test fails
#     result_dice_gm, result_diff_dc_gm, result_dice_wm, result_diff_dc_wm, result_hausdorff, result_diff_hd, result_median_dist, result_diff_md = float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
#     if status == 0:
#         dice_fname = os.path.join(path_output, 'dice_multilabel_reg.txt')
#         hausdorff_fname = os.path.join(path_output, 'hd_md_multilabel_reg.txt')
#
#         # Extracting dice results:
#         dice = open(dice_fname, 'r')
#         dice_lines = dice.readlines()
#         dice.close()
#         dc_start = dice_lines.index('#Slice, WM DC, WM diff, GM DC, GM diff\n')
#         dice_lines = dice_lines[dc_start+1:]
#
#         null_slices = []
#         wm_dice_list = []
#         wm_diff_list = []
#         gm_dice_list = []
#         gm_diff_list = []
#         for line in dice_lines:
#             dc_list = line.split(', ')
#             if dc_list[1][:2] == 'NO':
#                 null_slices.append(dc_list[0])
#             else:
#                 n_slice, wm_dc, wm_diff, gm_dc, gm_diff = dc_list
#                 gm_diff = gm_diff[:-1]
#                 if wm_dc != 'nan' and gm_dc != 'nan' and wm_dc != '0' and gm_dc != '0':
#                     wm_dice_list.append(float(wm_dc))
#                     wm_diff_list.append(float(wm_diff))
#                     gm_dice_list.append(float(gm_dc))
#                     gm_diff_list.append(float(gm_diff))
#
#         # use try to avoid computing mean in empty lists
#         if not wm_dice_list == []:
#             result_dice_wm = mean(wm_dice_list)
#         if not wm_diff_list == []:
#             result_diff_dc_wm = mean(wm_diff_list)
#         if not gm_dice_list == []:
#             result_dice_gm = mean(gm_dice_list)
#         if not gm_diff_list == []:
#             result_diff_dc_gm = mean(gm_diff_list)
#
#         # Extracting hausdorff distance results
#         hd_file = open(hausdorff_fname, 'r')
#         hd_lines = hd_file.readlines()
#         hd_file.close()
#
#         hd_start = hd_lines.index('#Slice, HD, HD diff, MD, MD diff\n')
#         hd_lines = hd_lines[hd_start+1:]
#
#         hausdorff_list = []
#         hd_diff_list = []
#         max_med_list = []
#         md_diff_list = []
#         for line in hd_lines:
#             values = line.split(', ')
#             if values[0] not in null_slices:
#                 n_slice, hd, hd_diff, md, md_diff = values
#
#                 if hd != 'nan' and md != 'nan' and hd != '0.0' and md != '0.0':
#                     hausdorff_list.append(float(hd))
#                     hd_diff_list.append(float(hd_diff))
#                     max_med_list.append(float(md))
#                     md_diff_list.append(float(md_diff))
#
#         if not hausdorff_list == []:
#             result_hausdorff = mean(hausdorff_list)
#         if not hd_diff_list == []:
#             result_diff_hd = mean(hd_diff_list)
#         if not max_med_list == []:
#             result_median_dist = mean(max_med_list)
#         if not md_diff_list == []:
#             result_diff_md = mean(md_diff_list)
#
#         # Integrity check
#         hd_threshold = 2  # in mm
#         wm_dice_threshold = 0.7
#         if result_hausdorff > hd_threshold or result_dice_wm < wm_dice_threshold:
#             status = 99
#             output += '\nResulting registration correction is too different from manual segmentation:\n' \
#                       'WM dice: '+str(result_dice_wm)+'\n' \
#                       'Hausdorff distance: '+str(result_hausdorff)+'\n'
#
#     # transform results into Pandas structure
#     results = DataFrame(data={'status': status, 'output': output, 'dice_gm': result_dice_gm, 'diff_dc_gm': result_diff_dc_gm, 'dice_wm': result_dice_wm, 'diff_dc_wm': result_diff_dc_wm, 'hausdorff': result_hausdorff, 'diff_hd': result_diff_hd, 'med_dist': result_median_dist, 'diff_md': result_diff_md, 'duration_[s]': duration}, index=[path_data])
#
#     return status, output, results
#
#
# if __name__ == "__main__":
#     # call main function
#     test(os.path.join(path_sct, 'data'))
