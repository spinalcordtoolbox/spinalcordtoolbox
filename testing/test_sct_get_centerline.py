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
import commands


def test(path_data):

    # parameters
    folder_data = 't2/'
    file_data = ['t2.nii.gz', 't2_centerline_init.nii.gz']

    # define command
    cmd = 'sct_get_centerline -i ' + path_data + folder_data + file_data[0] \
          + ' -p ' + path_data + folder_data + file_data[1] \
          + ' -g 1'\
          + ' -k 4'\
          + ' -r 1'
    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()