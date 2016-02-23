#!/usr/bin/env python
#########################################################################################
#
# Test function test_sct_dmri_compute_dti
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands


def test(path_data):

    folder_data = 'dmri/'
    file_data = ['dmri.nii.gz', 'bvecs.txt', 'bvals.txt']

    output = ''
    status = 0

    cmd = 'sct_dmri_compute_dti -i ' + path_data + folder_data + file_data[0] \
                + ' -bvec ' + path_data + folder_data + file_data[1] \
                + ' -bval ' + path_data + folder_data + file_data[2]
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command

    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    return status, output

if __name__ == "__main__":
    # call main function
    test()