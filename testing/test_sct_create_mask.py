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
    folder_data = ['mt/', 'dmri/']
    file_data = ['mt1.nii.gz', 'mt1_point.nii.gz', 'mt1_seg.nii.gz', 'dmri.nii.gz']

    output = ''
    status = 0

    # method coord
    cmd = 'sct_create_mask -i ' + data_path + folder_data[0] + file_data[0] \
          + ' -p coord,15x17' \
          + ' -size 10' \
          + ' -r 0'
    output += cmd+'\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    # time.sleep(1)  # here add one second, otherwise the next test will try to create a temporary folder with the same name (because it is typically ran in less than one second)

    # method point
    cmd = 'sct_create_mask -i ' + data_path + folder_data[0] + file_data[0] \
          + ' -p point,' + data_path + folder_data[0] + file_data[1] \
          + ' -size 10' \
          + ' -r 0'
    output += cmd+'\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    # time.sleep(1)

    # method center
    cmd = 'sct_create_mask -i ' + data_path + folder_data[0] + file_data[0] \
          + ' -p center' \
          + ' -size 10' \
          + ' -r 0'
    output += cmd+'\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    # time.sleep(1)

    # method centerline
    cmd = 'sct_create_mask -i ' + data_path + folder_data[0] + file_data[0] \
          + ' -p centerline,' + data_path + folder_data[0] + file_data[2] \
          + ' -size 10' \
          + ' -r 0'
    output += cmd+'\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    # time.sleep(1)

    # method center on 4d data
    cmd = 'sct_create_mask -i ' + data_path + folder_data[1] + file_data[3] \
          + ' -p center' \
          + ' -size 10' \
          + ' -r 0'
    output += cmd+'\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    return status, output


if __name__ == "__main__":
    # call main function
    test()

