#!/usr/bin/env python
#########################################################################################
#
# Test function sct_dmri_separate_b0_and_dwi
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/09
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct


def test(data_path):

    # parameters
    folder_data = 'dmri/'
    file_data = ['dmri.nii.gz', 'bvecs.txt']

    # define command
    cmd = 'sct_dmri_separate_b0_and_dwi -i ' + data_path + folder_data + file_data[0] \
          + ' -b ' + data_path + folder_data + file_data[1]\
          + ' -a 1'\
          + ' -o ./'\
          + ' -v 1'\
          + ' -r 0'
    # return
    return sct.run(cmd, 0)


if __name__ == "__main__":
    # call main function
    test()