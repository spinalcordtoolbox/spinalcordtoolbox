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

#import sct_utils as sct
import commands


def test(path_data):

    folder_data = 'mt/'
    file_data = ['mt0.nii.gz', 'mt1.nii.gz']

    cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -x 0' \
          + ' -o data_reg.nii.gz'  \
          + ' -n 1' \
          + ' -p 1' \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'

    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()