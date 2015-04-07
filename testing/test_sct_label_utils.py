#!/usr/bin/env python
#########################################################################################
#
# Test function sct_label_utils
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
    folder_data = ['t2/']
    file_data = ['t2_seg.nii.gz']

    # define command
    cmd = 'sct_label_utils -i ' + data_path + folder_data[0] + file_data[0] + ' -t create -x 1,1,1,1:2,2,2,2'

    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()