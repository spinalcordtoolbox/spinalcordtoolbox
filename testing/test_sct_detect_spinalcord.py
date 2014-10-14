#!/usr/bin/env python
#########################################################################################
#
# Test function sct_detect_spinalcord
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/09
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct


def test(path_data):

    # parameters
    folder_data = ['t2/']
    file_data = ['t2.nii.gz']

    # define command
    cmd = "sct_detect_spinalcord -i " + path_data + folder_data[0] + file_data[0] + \
          " -o " + file_data[0] + "_center.nii.gz -t " + file_data[0]

    # return
    return sct.run(cmd, 0)


if __name__ == "__main__":
    # call main function
    test()