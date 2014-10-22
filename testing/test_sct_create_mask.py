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
import time


def test(data_path):

    # parameters
    folder_data = 'mt/'
    file_data = ['mt1.nii.gz', 'mt1_point.nii.gz', 'mt1_seg.nii.gz']

    # define command

    # method coord
    cmd = 'sct_create_mask -i ' + data_path + folder_data + file_data[0] \
          + ' -m coord,15x17' \
          + ' -s 10' \
          + ' -r 0'
    status, output = commands.getstatusoutput(cmd)
    time.sleep(1)  # here add one second, otherwise the next test will try to create a temporary folder with the same name (because it is typically ran in less than one second)

    # method point
    # create label

    cmd = 'sct_create_mask -i ' + data_path + folder_data + file_data[0] \
          + ' -m point,' + data_path + folder_data + file_data[1] \
          + ' -s 10' \
          + ' -r 0'

    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    time.sleep(1)

    # method center
    cmd = 'sct_create_mask -i ' + data_path + folder_data + file_data[0] \
          + ' -m center' \
          + ' -s 10' \
          + ' -r 0'

    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    time.sleep(1)

    # method centerline
    cmd = 'sct_create_mask -i ' + data_path + folder_data + file_data[0] \
          + ' -m centerline,' + data_path + folder_data + file_data[2] \
          + ' -s 10' \
          + ' -r 0'

    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    return status, output


if __name__ == "__main__":
    # call main function
    test()

