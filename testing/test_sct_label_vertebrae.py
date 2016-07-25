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
from copy import deepcopy

# import commands
# # append path that contains scripts, to be able to load modules
# status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# sys.path.append(path_sct + '/scripts')

def test(path_data='', parameters=''):

    # initializations
    file_init_label_vertebrae = 'init_label_vertebrae.txt'
    rmse = float('NaN')
    max_dist = float('NaN')
    diff_manual_result = float('NaN')

    if not parameters:
        parameters = '-i t2/t2.nii.gz -s t2/t2_seg.nii.gz -c t2 -o t2_seg_labeled.nii.gz'

    parser = sct_label_vertebrae.get_parser()
    dict_param = parser.parse(parameters.split(), check_file_exist=False)
    dict_param_with_path = parser.add_path_to_file(deepcopy(dict_param), path_data, input_file=True)
    # update template path because the previous command wrongly adds path to testing data
    dict_param_with_path['-t'] = dict_param['-t']
    param_with_path = parser.dictionary_to_string(dict_param_with_path)

    # Check if input files exist
    if not (os.path.isfile(dict_param_with_path['-i']) and
            os.path.isfile(dict_param_with_path['-s'])):
        status = 200
        output = 'ERROR: the file(s) provided to test function do not exist in folder: ' + path_data
        return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])

    # create output folder to deal with multithreading (i.e., we don't want to have outputs from several subjects in the current directory)
    import time, random
    subject_folder = path_data.split('/')
    if subject_folder[-1] == '' and len(subject_folder) > 1:
        subject_folder = subject_folder[-2]
    else:
        subject_folder = subject_folder[-1]
    path_output = sct.slash_at_the_end('sct_label_vertebrae_' + subject_folder + '_' + time.strftime("%y%m%d%H%M%S") + '_'+str(random.randint(1, 1000000)), slash=1)
    param_with_path += ' -ofolder ' + path_output

    # add initialization parameter contained in file: init_label_vertebrae.txt
    if not os.path.isfile(path_data+'t2/'+file_init_label_vertebrae):
        status = 200
        output = 'ERROR: the file init_label_vertebrae.txt does not exist in folder: ' + path_data
        return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])
        # return status, output, DataFrame(data={'status': status, 'output': output, 'mse': float('nan')}, index=[path_data])
    else:
        file = open(path_data+'t2/'+file_init_label_vertebrae, 'r')
        param_with_path += ' '+file.read().replace('\n', '')

    cmd = 'sct_label_vertebrae ' + param_with_path
    output = '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    time_start = time.time()
    try:
        status, o = sct.run(cmd, 0)
    except:
        status, o = 1, 'ERROR: Function crashed!'
    output += o
    duration = time.time() - time_start

    # initialization of results: must be NaN if test fails
    result_mse = float('nan'), float('nan')

    if status == 0:
        # copy input data (for easier debugging)
        sct.run('cp '+dict_param_with_path['-i']+' '+path_output, verbose=0)
        # extract center of vertebral labels
        sct.run('sct_label_utils -i '+path_output+'t2_seg_labeled.nii.gz -vert-body 0 -o '+path_output+'t2_seg_labeled_center.nii.gz', verbose=0)
        from sct_label_utils import ProcessLabels
        from numpy import linalg
        from math import sqrt
        # get dimension
        from msct_image import Image
        img = Image(path_output+'t2_seg_labeled.nii.gz')
        nx, ny, nz, nt, px, py, pz, pt = img.dim
        # open labels
        label_results = ProcessLabels(path_output+'t2_seg_labeled_center.nii.gz')
        list_label_results = label_results.image_input.getNonZeroCoordinates(sorting='value')
        label_manual = ProcessLabels(path_data+'t2/t2_labeled_center_manual.nii.gz')
        list_label_manual = label_manual.image_input.getNonZeroCoordinates(sorting='value')
        mse = 0.0
        max_dist = 0.0
        for coord_manual in list_label_manual:
            for coord in list_label_results:
                if round(coord.value) == round(coord_manual.value):
                    # Calculate MSE
                    mse += (((coord_manual.x - coord.x)*px) ** 2 + ((coord_manual.y - coord.y)*py) ** 2 + ((coord_manual.z - coord.z)*pz) ** 2) / float(3)
                    # Calculate distance (Frobenius norm)
                    dist = linalg.norm([(coord_manual.x - coord.x)*px, (coord_manual.y - coord.y)*py, (coord_manual.z - coord.z)*pz])
                    if dist > max_dist:
                        max_dist = dist
                    break
        rmse = sqrt(mse / len(list_label_manual))
        # calculate number of label mismatch
        diff_manual_result = len(list_label_manual) - len(list_label_results)

        # # display results
        # sct.printv('RMSE = ' + str(rmse) + ' mm')
        # sct.printv('Max distance = ' + str(max_dist) + ' mm')
        # sct.printv('Diff manual-test = ' + str(diff_manual_result))

        # check if MSE is superior to threshold
        th_rmse = 2
        if rmse > th_rmse:
            status = 99
            output += '\nWARNING: RMSE = '+str(rmse)+' > '+str(th_rmse)
        th_max_dist = 4
        if max_dist > th_max_dist:
            status = 99
            output += '\nWARNING: Max distance = '+str(max_dist)+' > '+str(th_max_dist)
        th_diff_manual_result = 3
        if abs(diff_manual_result) > th_diff_manual_result:
            status = 99
            output += '\nWARNING: Diff manual-result = '+str(diff_manual_result)+' > '+str(th_diff_manual_result)

    # transform results into Pandas structure
    results = DataFrame(data={'status': int(status), 'output': output, 'rmse': rmse, 'max_dist': max_dist, 'diff_man': diff_manual_result, 'duration [s]': duration}, index=[path_data])

    return status, output, results

if __name__ == "__main__":
    # call main function
    test()
