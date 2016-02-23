#!/usr/bin/env python
#########################################################################################
#
# Test function sct_propseg
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/09
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands


def test(path_data):

    # parameters
    folder_data = 't2/'
    file_data = ['t2.nii.gz', 't2_seg.nii.gz']
    dice_threshold = 0.99

    # define command
    cmd = 'sct_propseg -i ' + path_data + folder_data + file_data[0] \
        + ' -c t2' \
        + ' -mesh'\
        + ' -cross'\
        + ' -centerline-binary'\
        + ' -v 1'

    # run command
    status, output = commands.getstatusoutput(cmd)

    # if command ran without error, test integrity
    if status == 0:
        # compute dice coefficient between generated image and image from database
        cmd = 'sct_dice_coefficient -i ' + path_data + folder_data + file_data[1] + ' -d ' + file_data[1]
        status, output = commands.getstatusoutput(cmd)
        # parse output and compare to acceptable threshold
        dice = float(output.split('3D Dice coefficient = ')[1].split('\n')[0])
        if dice < dice_threshold:
            status = 99

    return status, output


if __name__ == "__main__":
    # call main function
    test()