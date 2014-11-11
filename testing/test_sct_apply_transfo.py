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


def test(data_path):

    # parameters
    folder_data = [ 'template/', 't2/']
    file_data = ['MNI-Poly-AMU_T2.nii.gz', 't2.nii.gz', 'warp_template2anat.nii.gz']

    # define command
    cmd = 'sct_apply_transfo -i ' + data_path + folder_data[0] + file_data[0] \
          + ' -d ' + data_path + folder_data[1] + file_data[1] \
          + ' -w ' + data_path + folder_data[1] + file_data[2]

    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()