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


class Param:
    def __init__(self, data_path='', data_files=[]):
        self.data_path = data_path
        self.data_files = data_files
        self.verbose = 1

def test(param):

    # define command
    cmd = 'sct_fmri_compute_tsnr -fmri ' + param.data_path + '/' + param.data_files[1] \
          + ' -anat ' + param.data_path + '/' + param.data_files[0]

    # return
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    param = Param(data_path=os.path.abspath('.'), data_files=['t2.nii.gz', 'fmri.nii.gz'])
    test(param)


