#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_fmri_compute_tsnr script
#
# The test should be done in a folder containing a fmri.nii.gz and a t2.nii.gz files
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# modified: 2015-03-16
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import commands
import os


def test(data_path):
    #parameters
    data_folders = ['fmri/']
    data_files = ['fmri.nii.gz']

    # define command
    cmd = 'sct_fmri_compute_tsnr -i ' + data_path + data_folders[0] + data_files[0]

    # return
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()


