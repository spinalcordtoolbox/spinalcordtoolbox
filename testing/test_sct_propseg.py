#!/usr/bin/env python
#########################################################################################
#
# Test function sct_propseg
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
    folder_data = 't2/'
    file_data = ['t2_seg.nii.gz', 't2_manual_segmentation.nii.gz', 't2_seg.nii.gz']

    # define command
    cmd = 'sct_propseg -i ' + path_data + folder_data + file_data[0] \
        + ' -t t2' \
        + ' -mesh'\
        + ' -cross'\
        + ' -centerline-binary'\
        + ' -verbose'
    '''
    cmd2 = 'sct_dice_coefficient ' + path_data + folder_data + file_data[1] \
                + ' ' + f[2] \
                + ' -bmax'
    '''
    # return
    return sct.run(cmd, 0)


if __name__ == "__main__":
    # call main function
    test()