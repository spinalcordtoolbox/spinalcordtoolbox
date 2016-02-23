#!/usr/bin/env python
#########################################################################################
#
# Test function sct_maths
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# modified: 2014-10-06
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands


def test(path_data):

    folder_data = 'mt/'
    file_data = ['mtr.nii.gz']

    output = ''
    status = 0

    cmd = 'sct_maths -i ' + path_data + folder_data + file_data[0] \
                + ' -o test.nii.gz' \
                + ' -percent 95'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command

    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    return status, output

if __name__ == "__main__":
    # call main function
    test()