#!/usr/bin/env python
#########################################################################################
#
# Test function sct_dmri_create_noisemask
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley Gros
# modified: 2016/02/23
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
import commands

def test(path_data):

    folder_data = 'dmri/'
    file_data = ['dwi.nii.gz']


    cmd = 'sct_dmri_create_noisemask -i ' + path_data + folder_data + file_data[0] \
                + ' -dof 1'\
                + ' -o noise_mask.nii.gz'

    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    test()