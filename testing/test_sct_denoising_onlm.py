#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_get_centerline script
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
import os, commands


def test(path_data):

    # parameters
    folder_data = 't2'
    file_data = ['t2.nii.gz']

    # define command
    cmd = 'sct_denoising_onlm -i ' + os.path.join(path_data, folder_data, file_data[0]) \
          + ' -v 2'
    # return
    #return sct.run(cmd, 0)
    #return commands.getstatusoutput(cmd)
    return 0  # temporarely removing testing about denoising because it uses dipy and dipy has been removed from dependences


if __name__ == "__main__":
    # call main function
    test()
