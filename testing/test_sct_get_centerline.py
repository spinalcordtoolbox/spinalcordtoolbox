#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_get_centerline script
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

import os.path
import sys
import commands
from msct_image import Image
from sct_get_centerline import ind2sub
import math
import sct_utils as sct
import numpy as np
from pandas import DataFrame
import sct_get_centerline


def test(path_data='', parameters=''):

    # parameters
    output = ''
    # folder_data = 't2/'
    # file_data = ['t2.nii.gz', 't2_centerline_init.nii.gz', 't2_centerline_labels.nii.gz', 't2_seg_manual.nii.gz']

    if not parameters:
        parameters = '-i t2/t2.nii.gz -c t2'

    parser = sct_get_centerline.get_parser()
    dict_param = parser.parse(parameters.split(), check_file_exist=False)
    contrast = dict_param['-c']
    dict_param_with_path = parser.add_path_to_file(dict_param, path_data, input_file=True)
    param_with_path = parser.dictionary_to_string(dict_param_with_path)

    # log file
    fname_log = path_output + 'output.log'
    stdout_log = file(fname_log, 'w')
    # redirect to log file
    stdout_orig = sys.stdout
    sys.stdout = stdout_log

    # Check if input files exist
    if not (os.path.isfile(dict_param_with_path['-i'])):
        status = 200
        output += '\nERROR: the file(s) provided to test function do not exist in folder: ' + path_data
        write_to_log_file(fname_log, output, 'w')
        return status, output, DataFrame(
            data={'status': status, 'output': output, 'mse': float('nan'), 'dist_max': float('nan')}, index=[path_data])

    # Check if ground truth files exist
    if not os.path.isfile(path_data + contrast + '/' + contrast + '_seg_manual.nii.gz'):
        status = 201
        output += '\nERROR: the file *_seg_manual.nii.gz does not exist in folder: ' + path_data
        write_to_log_file(fname_log, output, 'w')
        return status, output, DataFrame(
            data = {'status': status, 'output': output, 'mse': float('nan'), 'dist_max': float('nan')}, index = [path_data])


    cmd = 'sct_get_centerline '+param_with_path
    status, output = sct.run(cmd, 0)
    scad_centerline = Image(contrast+"_centerline.nii.gz")
    manual_seg = Image(path_data + folder_data + contrast +'_seg_manual.nii.gz')

    max_distance = 0
    standard_deviation = 0
    average = 0
    root_mean_square = 0
    overall_distance = 0
    max_distance = 0
    overall_std = 0
    rmse = 0

    try:
        if status == 0:
            manual_seg.change_orientation()
            scad_centerline.change_orientation()
            from scipy.ndimage.measurements import center_of_mass
            # find COM
            iterator = range(manual_seg.data.shape[2])
            com_x = [0 for ix in iterator]
            com_y = [0 for iy in iterator]

            for iz in iterator:
                com_x[iz], com_y[iz] = center_of_mass(manual_seg.data[:, :, iz])
            max_distance = {}
            distance = {}
            for iz in range(1, scad_centerline.data.shape[2]-1):
                ind1 = np.argmax(scad_centerline.data[:, :, iz])
                X,Y = ind2sub(scad_centerline.data[:, :, iz].shape,ind1)
                com_phys = np.array(manual_seg.transfo_pix2phys([[com_x[iz], com_y[iz], iz]]))
                scad_phys = np.array(scad_centerline.transfo_pix2phys([[X, Y, iz]]))
                distance_magnitude = np.linalg.norm([com_phys[0][0]-scad_phys[0][0], com_phys[0][1]-scad_phys[0][1], 0])
                if math.isnan(distance_magnitude):
                    print "Value is nan"
                else:
                    distance[iz] = distance_magnitude

            max_distance = max(distance.values())
            standard_deviation = np.std(np.array(distance.values()))
            average = sum(distance.values())/len(distance)
            root_mean_square = np.sqrt(np.mean(np.square(distance.values())))
            overall_distance = average
            max_distance = max(distance.values())
            overall_std = standard_deviation
            rmse = root_mean_square

    except Exception, e:
        sct.printv("Exception found while testing scad integrity")
        output = e.message

    result_mse, result_dist_max = rmse, max_distance
    results = DataFrame(data={'status': status, 'output': output, 'mse': result_mse, 'dist_max': result_dist_max}, index=[path_data])

    # define command
    cmd = 'sct_get_centerline -i ' + path_data + folder_data + file_data[0] \
        + ' -p labels ' \
        + ' -l ' + path_data + folder_data + file_data[2] \
        + ' -v 1'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    return status, output, results


if __name__ == "__main__":
    # call main function
    test()