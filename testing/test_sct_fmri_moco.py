#!/usr/bin/env python
#########################################################################################
#
# Test function sct_fmri_moco
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

    folder_data = 'fmri/'
    file_data = ['fmri.nii.gz']

    output = ''
    status = 0

    cmd = 'sct_fmri_moco -i ' + path_data + folder_data + file_data[0] \
                + ' -g 5' \
                + ' -x spline' \
                + ' -r 0' \
                + ' -v 2'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command

    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    return status, output

if __name__ == "__main__":
    # call main function
    test()