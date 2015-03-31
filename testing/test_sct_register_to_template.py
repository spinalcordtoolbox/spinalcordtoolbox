#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_to_template script
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


#import sct_utils as sct
import commands


def test(path_data):
    folder_data = ['t2/', 'template/']
    file_data = ['t2.nii.gz', 'labels.nii.gz', 't2_seg.nii.gz']

    cmd = 'sct_register_to_template -i ' + path_data + folder_data[0] + file_data[0] \
          + ' -l ' + path_data + folder_data[0] + file_data[1] \
          + ' -s ' + path_data + folder_data[0] + file_data[2] \
          + ' -r 0' \
          + ' -t ' + path_data + folder_data[1]

    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()