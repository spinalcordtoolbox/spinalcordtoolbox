#!/usr/bin/env python
#########################################################################################
#
# Test function sct_create_mask
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# create: 2014/10/19
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands


def test(data_path):

    # parameters
    folder_data = 't2/'
    file_data = ['t2.nii.gz','labels.nii.gz','t2_centerline_init.nii.gz']

    # define command

    # method coord
    cmd = 'sct_create_mask -i ' + data_path + folder_data + file_data[0] \
          + ' -m coord,30x27'\
          + ' -s 10'
    status, output = commands.getstatusoutput(cmd)

    # method point
    cmd = 'sct_create_mask -i ' + data_path + folder_data + file_data[0] \
          + ' -m point,' + file_data[1] \
          + ' -s 10'
    s, o = commands.getstatusoutput(cmd)

    status += s
    output += o

    # method center
    cmd = 'sct_create_mask -i ' + data_path + folder_data + file_data[0] \
          + ' -m center' \
          + ' -s 10'
    s, o = commands.getstatusoutput(cmd)

    status += s
    output += o

    # method centerline
    cmd = 'sct_create_mask -i ' + data_path + folder_data + file_data[0] \
          + ' -m centerline,' + data_path + folder_data + file_data[2] \
          + ' -s 10'
    s, o = commands.getstatusoutput(cmd)

    status += s
    output += o

    return status, output


if __name__ == "__main__":
    # call main function
    test()

