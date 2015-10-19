#!/usr/bin/env python
#########################################################################################
#
# Test function sct_segment_graymatter
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# modified: 2015/08/31
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import commands
import sys
import os
from pandas import DataFrame
import sct_segment_graymatter
from msct_image import Image
import sct_utils as sct
from numpy import sum
# append path that contains scripts, to be able to load modules
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
sys.path.append(path_sct + '/scripts')



def test(path_data, parameters=''):

    if not parameters:
        parameters = '-i mt/mt0.nii.gz -s mt/mt0_seg.nii.gz -l mt/label/template/MNI-Poly-AMU_level.nii.gz -normalize 1 -ref mt/mt0_gmseg.nii.gz'

    parser = sct_segment_graymatter.get_parser()
    dict_param = parser.parse(parameters.split(), check_file_exist=False)
    dict_param_with_path = parser.add_path_to_file(dict_param, path_data, input_file=True)
    param_with_path = parser.dictionary_to_string(dict_param_with_path)

    # Check if input files exist
    if not (os.path.isfile(dict_param_with_path['-i']) and os.path.isfile(dict_param_with_path['-s']) and os.path.isfile(dict_param_with_path['-l']) and os.path.isfile(dict_param_with_path['-ref'])):
        status = 200
        output = 'ERROR: the file(s) provided to test function do not exist in folder: ' + path_data
        return status, output, DataFrame(data={'status': status, 'output': output, 'dice': float('nan'), 'hausdorff': float('nan')}, index=[path_data])

    cmd = 'sct_segment_graymatter ' + param_with_path
    status, output = sct.run(cmd, 0)


    # initialization of results: must be NaN if test fails
    result_dice, result_hausdorff = float('nan'), float('nan')
    if status == 0:
        pass
        '''
        # extraction of results
        output_split = output.split('Maximum x-y error = ')[1].split(' mm')
        result_dist_max = float(output_split[0])
        result_mse = float(output_split[1].split('Accuracy of straightening (MSE) = ')[1])

        # integrity testing - straightening has been tested with v2.0.6 on several images.
        # mse is less than 1.5 and dist_max is less than 4
        if result_dist_max > 4.0 or result_mse > 1.5:
            status = 99
        '''

    # transform results into Pandas structure
    results = DataFrame(data={'status': status, 'output': output, 'dice': result_dice, 'hausdorff': result_hausdorff}, index=[path_data])

    return status, output, results



'''
# ##################################################################################################################
output = ''
status = 0

# parameters
folder_data = 'mt/'
file_data = ['mt0.nii.gz', 'mt0_seg.nii.gz', 'label/template/MNI-Poly-AMU_level.nii.gz', 'mt0_gmseg.nii.gz']

# define command
cmd = 'sct_segment_graymatter -i ' + path_data + folder_data + file_data[0] \
    + ' -s ' + path_data + folder_data + file_data[1] \
    + ' -l ' + path_data + folder_data + file_data[2] \
    + ' -normalize 1 '\
    + ' -v 1'

output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
# run command
s, o = commands.getstatusoutput(cmd)
status += s
output += o

# if command ran without error, test integrity
if status == 0:
    threshold = 1e-3
    # compare with gold-standard labeling
    data_original = Image(path_data + folder_data + file_data[-1]).data
    data_totest = Image('mt0_gmseg.nii.gz').data
    # check if non-zero elements are present when computing the difference of the two images
    diff = data_original - data_totest


    if abs(sum(diff))> threshold:
        Image(param=diff, absolutepath='res_differences_from_gold_standard.nii.gz').save()
        status = 99
        output += '\nResulting image differs from gold-standard.'

return status, output
'''


if __name__ == "__main__":
    # call main function
    test(path_sct+'/data')
