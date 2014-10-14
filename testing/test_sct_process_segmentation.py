#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_process_segmentation script
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

import sct_utils as sct


def test(path_data):

    # parameters
    folder_data = 't2/'
    file_data = 't2_seg.nii.gz'

    # define command
    cmd = 'sct_process_segmentation -i ' + path_data + folder_data + file_data \
          + ' -p compute_csa' \
          + ' -s 1'\
          + ' -b 1'\
          + ' -r 0'\
          + ' -v 1'
    # return
    return sct.run(cmd, 0)


if __name__ == "__main__":
    # call main function
    test()
