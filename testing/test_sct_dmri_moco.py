#!/usr/bin/env python
#########################################################################################
#
# Test function sct_dmri_moco
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/06
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
import commands


def test(path_data):

    folder_data = 'dmri/'
    file_data = ['dmri.nii.gz', 'bvecs.txt']


    cmd = 'sct_dmri_moco -i ' + path_data + folder_data + file_data[0] \
                + ' -bvec '+ path_data + folder_data + file_data[1] \
                + ' -v 1'\
                + ' -g 3'\
                + ' -r 0'\
                + ' -x spline'

    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()