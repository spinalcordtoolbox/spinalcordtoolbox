#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_spinalcord script
#
#   replace the shell test script in sct 1.0
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014-08-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sct_utils as sct
import commands


def test(path_data):

    # parameters
    folder_data = 't2/'
    file_data = ['t2.nii.gz', 't2_seg.nii.gz']

    # define command
    cmd = 'sct_smooth_spinalcord' \
        ' -i '+path_data+folder_data+file_data[0]+ \
        ' -s '+path_data+folder_data+file_data[1]+ \
        ' -smooth 5'

    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


# call to function
if __name__ == "__main__":
    # call main function
    test()
