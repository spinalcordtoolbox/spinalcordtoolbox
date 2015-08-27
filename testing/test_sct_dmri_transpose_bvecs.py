#!/usr/bin/env python
#########################################################################################
#
# Test function sct_dmri_transpose_bvecs
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


def test(data_path):

    # parameters
    folder_data = ['dmri/']
    file_data = ['bvecs.txt']

    # define command
    cmd = 'sct_dmri_transpose_bvecs -i ' + data_path + folder_data[0] + file_data[0]

    # return
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()