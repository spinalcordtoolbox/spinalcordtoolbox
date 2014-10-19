#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_multimodal script
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

    folder_data = 'mt/'
    file_data = ['mt0.nii.gz', 'mt1.nii.gz']

    cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -o data_reg.nii.gz'  \
          + ' -p 3,SyN,0.5,MI'  \
          + ' -z 1' \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'

    return sct.run(cmd, 0)


if __name__ == "__main__":
    # call main function
    test()