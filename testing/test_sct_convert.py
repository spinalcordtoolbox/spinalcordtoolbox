#!/usr/bin/env python
#########################################################################################
#
# Test function sct_convert
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
    folder_data = ['t2/']
    file_data = ['t2.nii.gz']

    cmd = 'sct_convert -i ' + data_path + folder_data[0] + file_data[0] + ' -o data.nii'

    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()