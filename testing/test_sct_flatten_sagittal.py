#!/usr/bin/env python
#########################################################################################
#
# Test function sct_flatten_sagital
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/19
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
import commands


def test(data_path):

    # parameters
    folder_data = 't2/'
    file_data = ['t2.nii.gz','t2_centerline_init.nii.gz']

    # define command
    cmd = 'sct_flatten_sagittal -i ' + data_path + folder_data + file_data[0] \
          + ' -s ' + data_path + folder_data + file_data[1]

    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()
