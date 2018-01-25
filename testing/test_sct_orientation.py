#!/usr/bin/env python
#########################################################################################
#
# Test function sct_orientation
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: CHECK INTEGRITY OF ORIENTATION

#import sct_utils as sct



def test(data_path):

    # parameters
    folder_data = ['t2/', 'dmri/']
    file_data = ['t2.nii.gz', 'dmri.nii.gz']

    # test 3d data
    cmd = 'sct_orientation -i ' + data_path + folder_data[0] + file_data[0]
    status, output = sct.run(cmd)
    from time import sleep
    sleep(1)  # here add one second, otherwise the next test will try to create a temporary folder with the same name (because it runs in less than one second)

    # test 4d data
    if status == 0:
        cmd = 'sct_orientation -i ' + data_path + folder_data[1] + file_data[1]
        status, output = sct.run(cmd)

    # return
    return status, output


if __name__ == "__main__":
    # call main function
    test()