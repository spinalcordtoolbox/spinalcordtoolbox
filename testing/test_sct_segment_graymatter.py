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

#import sct_utils as sct
import commands


def test(path_data):

    # parameters
    folder_data = 'mt/'
    file_data = ['mt1.nii.gz', 'mt1_seg.nii.gz', 'label/template/MNI-Poly-AMU_level.nii.gz']
    dice_threshold = 0.99

    # define command
    cmd = 'sct_segment_graymatter -i ' + path_data + folder_data + file_data[0] \
        + ' -s ' + path_data + folder_data + file_data[1] \
        + ' -l ' + path_data + folder_data + file_data[2] \
        + ' -normalize 1 '\
        + ' -v 1'

    # run command
    status, output = commands.getstatusoutput(cmd)

    # if command ran without error, test integrity
    if status == 0:
        pass
        '''
        # compute dice coefficient between generated image and image from database
        cmd = 'sct_dice_coefficient ' + path_data + folder_data + file_data[1] + ' ' + file_data[1]
        status, output = commands.getstatusoutput(cmd)
        # parse output and compare to acceptable threshold
        if float(output.split('3D Dice coefficient = ')[1]) < dice_threshold:
            status = 99
        '''
    return status, output


if __name__ == "__main__":
    # call main function
    test()