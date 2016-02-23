#!/usr/bin/env python
#########################################################################################
#
# Test function sct_apply_transfo
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands
from time import sleep


def test(data_path):

    # parameters
    folder_data = [ 'template/template/', 't2/', 'dmri/']
    file_data = ['MNI-Poly-AMU_T2.nii.gz', 't2.nii.gz', 'warp_template2anat.nii.gz', 'dmri.nii.gz']

    output = ''
    status = 0

    # test function
    cmd = 'sct_apply_transfo -i ' + data_path + folder_data[0] + file_data[0] \
          + ' -d ' + data_path + folder_data[1] + file_data[1] \
          + ' -w ' + data_path + folder_data[1] + file_data[2]
    output += cmd+'\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    sleep(1)  # here one second, otherwise the next test will try to create a temporary folder with the same name

    # test with 4d input
    cmd = 'sct_apply_transfo -i ' + data_path + folder_data[2] + file_data[3] \
          + ' -d ' + data_path + folder_data[1] + file_data[1] \
          + ' -w ' + data_path + folder_data[1] + file_data[2]
    output += cmd+'\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # return
    #return sct.run(cmd, 0)
    return status, output


if __name__ == "__main__":
    # call main function
    test()