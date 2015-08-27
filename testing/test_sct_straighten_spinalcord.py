#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_sctraighten_spinalcord script
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


def test(path_data):

    folder_data = 't2/'
    file_data = ['t2.nii.gz', 't2_seg.nii.gz', 't2_straight.nii.gz', 't2_seg_straight.nii.gz']
    dice_threshold = 0.99

    cmd = 'sct_straighten_spinalcord -i ' + path_data + folder_data + file_data[0] \
          + ' -c ' + path_data + folder_data + file_data[1] \
          + ' -r 0' \
          + ' -v 1'

    status, output = sct.run(cmd, 0)

    if status == 0:
        cmd_c2s = 'sct_apply_transfo -i ' + path_data + folder_data + file_data[1] + \
                  ' -d ' + file_data[2] + \
                  ' -w warp_curve2straight.nii.gz' + \
                  ' -o ' + file_data[3] + \
                  '  -x nn'
        status_c2s, output_c2s = sct.run(cmd_c2s, 0)
        if status_c2s != 0:
            return status_c2s, output_c2s

        cmd_s2c = 'sct_apply_transfo -i ' + file_data[3] + \
                  ' -d ' + path_data + folder_data + file_data[0] + \
                  ' -w warp_straight2curve.nii.gz' + \
                  ' -o image_test.nii.gz ' \
                  '-x nn'
        status_s2c, output_s2c = sct.run(cmd_s2c, 0)
        if status_s2c != 0:
            return status_s2c, output_s2c

        cmd_dice = 'sct_dice_coefficient ' + path_data + folder_data + file_data[1] + \
                   ' image_test.nii.gz -bzmax'
        status_dice, output_dice = sct.run(cmd_dice, 0)
        if float(output_dice.split('3D Dice coefficient = ')[1]) < dice_threshold:
            output += output_c2s + output_s2c + output_dice
            status = 5

    return status, output

if __name__ == "__main__":
    # call main function
    test()