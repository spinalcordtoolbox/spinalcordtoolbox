#!/usr/bin/env python
#########################################################################################
#
# Test function sct_convert_binary_to_trilinear
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014-10-04
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

import os

def test(path_data):

    # parameters
    folder_data = 't2'
    file_data = 't2_seg-manual.nii.gz'

    # define command
    cmd = 'sct_convert_binary_to_trilinear' \
        ' -i ' + os.path.join(path_data, folder_data, file_data) + \
        ' -s 5'

    # return
    #return sct.run(cmd, 0)
    return sct.run(cmd)


if __name__ == "__main__":
    # call main function
    test()
