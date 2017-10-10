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

# import commands
# import sys
import os
from pandas import DataFrame
import sct_segment_graymatter
# from msct_image import Image
import sct_utils as sct
from numpy import sum, mean
# import time
from sct_warp_template import get_file_label
# append path that contains scripts, to be able to load modules
# status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# sys.path.append(path_sct + '/scripts')


def test(path_data, parameters=''):

    if not parameters:
        # get file name of vertebral labeling from template
        # file_vertfile = get_file_label(path_data+'mt/label/template', 'vertebral', output='file')
        parameters = '-i t2s/t2s.nii.gz -s t2s/t2s_seg.nii.gz -vertfile t2s/MNI-Poly-AMU_level_crop.nii.gz -ref t2s/t2s_gmseg_manual.nii.gz -qc 0 -ratio level'

    parser = sct_segment_graymatter.get_parser()
    dict_param = parser.parse(parameters.split(), check_file_exist=False)
    dict_param_with_path = parser.add_path_to_file(dict_param, path_data, input_file=True)

    #if -model is used : do not add the path before.
    if '-model' in dict_param_with_path.keys():
        dict_param_with_path['-model'] = dict_param_with_path['-model'][len(path_data):]
    if '-vertfile' in dict_param_with_path.keys():
        dict_param_with_path['-vertfile'] = sct.slash_at_the_end(path_data, slash=1)+dict_param_with_path['-vertfile']

    param_with_path = parser.dictionary_to_string(dict_param_with_path)

    # Check if input files exist
    if not (os.path.isfile(dict_param_with_path['-i']) and os.path.isfile(dict_param_with_path['-s'])):
        status = 200
        output = 'ERROR: the file(s) provided to test function do not exist in folder: ' + path_data
        return status, output, DataFrame(data={'status': status, 'output': output, 'dice_gm': float('nan'), 'dice_wm': float('nan'), 'hausdorff': float('nan'), 'med_dist': float('nan'), 'duration_[s]': float('nan')}, index=[path_data])

    import time, random
    subject_folder = path_data.split('/')
    if subject_folder[-1] == '' and len(subject_folder) > 1:
        subject_folder = subject_folder[-2]
    else:
        subject_folder = subject_folder[-1]
    path_output = sct.slash_at_the_end('sct_segment_graymatter_' + subject_folder + '_' + time.strftime("%y%m%d%H%M%S") + '_'+str(random.randint(1, 1000000)), slash=1)
    param_with_path += ' -ofolder ' + path_output

    cmd = 'sct_segment_graymatter ' + param_with_path
    time_start = time.time()
    status, output = sct.run(cmd, 0)
    duration = time.time() - time_start

    # initialization of results: must be NaN if test fails
    result_dice_gm, result_dice_wm, result_hausdorff, result_median_dist = float('nan'), float('nan'), float('nan'), float('nan')
    if status == 0 and "-ref" in dict_param_with_path.keys()    :
        target_name = sct.extract_fname(dict_param_with_path["-i"])[1]

        dice_fname = path_output+'dice_coefficient_'+target_name+'.txt'
        hausdorff_fname = path_output+'hausdorff_dist_'+target_name+'.txt'

        # Extracting dice results:
        dice = open(dice_fname, 'r')
        dice_lines = dice.readlines()
        dice.close()
        gm_start = dice_lines.index('Dice coefficient on the Gray Matter segmentation:\n')
        wm_start = dice_lines.index('Dice coefficient on the White Matter segmentation:\n')

        # extracting dice on GM
        gm_dice_lines = dice_lines[gm_start:wm_start-1]
        gm_dice_lines = gm_dice_lines[gm_dice_lines.index('2D Dice coefficient by slice:\n')+1:-1]

        null_slices = []
        gm_dice = []
        for line in gm_dice_lines:
            n_slice, dc = line.split(' ')
            # remove \n from dice result
            dc = dc[:-1]
            dc = dc[:-4] if '[0m' in dc else dc

            if dc == '0' or dc == 'nan':
                null_slices.append(n_slice)
            else:
                try:
                    gm_dice.append(float(dc))
                except ValueError:
                    gm_dice.append(float(dc[:-4]))
        result_dice_gm = mean(gm_dice)

        # extracting dice on WM
        wm_dice_lines = dice_lines[wm_start:]
        wm_dice_lines = wm_dice_lines[wm_dice_lines.index('2D Dice coefficient by slice:\n')+1:]
        wm_dice = []
        for line in wm_dice_lines:
            n_slice, dc = line.split(' ')
            # remove \n from dice result
            if line is not wm_dice_lines[-1]:
                dc = dc[:-1]
            if n_slice not in null_slices:
                try:
                    wm_dice.append(float(dc))
                except ValueError:
                    wm_dice.append(float(dc[:-4]))
        result_dice_wm = mean(wm_dice)

        # Extracting hausdorff distance results
        hd = open(hausdorff_fname, 'r')
        hd_lines = hd.readlines()
        hd.close()

        # remove title of columns and last empty/non important lines
        hd_lines = hd_lines[1:-4]

        hausdorff = []
        max_med = []
        for line in hd_lines:
            slice_id, res = line.split(':')
            slice, n_slice = slice_id.split(' ')
            if n_slice not in null_slices:
                hd, med1, med2 = res[:-1].split(' - ')
                hd, med1, med2 = float(hd), float(med1), float(med2)
                hausdorff.append(hd)
                max_med.append(max(med1, med2))

        result_hausdorff = mean(hausdorff)
        result_median_dist = mean(max_med)

        # Integrity check
        hd_threshold = 3 # in mm
        wm_dice_threshold = 0.8
        if result_hausdorff > hd_threshold or result_dice_wm < wm_dice_threshold:
            status = 99
            output += '\nResulting segmentation is too different from manual segmentation:\n' \
                      'WM dice: '+str(result_dice_wm)+'\n' \
                      'Hausdorff distance: '+str(result_hausdorff)+'\n'

    # transform results into Pandas structure
    results = DataFrame(data={'status': status, 'output': output, 'dice_gm': result_dice_gm, 'dice_wm': result_dice_wm, 'hausdorff': result_hausdorff, 'med_dist': result_median_dist, 'duration_[s]': duration}, index=[path_data])

    return status, output, results


if __name__ == "__main__":
    # call main function
    test(path_sct+'/data')
