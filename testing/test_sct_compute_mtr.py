#!/usr/bin/env python
#########################################################################################
#
# Test function sct_compute_mtr
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/20
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands


def test(data_path):

    # parameters
    folder_data = 'mt/'
    file_data = ['mt0.nii.gz','mt1.nii.gz']

    # define command
    cmd = 'sct_compute_mtr -i ' + data_path + folder_data + file_data[0] \
          + ' -j ' + data_path + folder_data + file_data[1]

    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()
