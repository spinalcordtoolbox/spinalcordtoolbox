#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_get_centerline script
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

import commands

def test(path_data):

    # parameters
    folder_data = 't2/'
    file_data = ['t2.nii.gz', 't2_centerline_init.nii.gz', 't2_centerline_labels.nii.gz']

    output = ''
    status = 0

    # define command
    cmd = 'sct_get_centerline -i ' + path_data + folder_data + file_data[0] \
        + ' -method auto' \
        + ' -t t2 ' \
        + ' -v 1'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # define command: DOES NOT RUN IT BECAUSE REQUIRES FSL FLIRT
    # cmd = 'sct_get_centerline -i ' + path_data + folder_data + file_data[0] \
    #     + ' -method point' \
    #     + ' -p ' + path_data + folder_data + file_data[1] \
    #     + ' -g 1'\
    #     + ' -k 4'
    # output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    # s, o = commands.getstatusoutput(cmd)
    # status += s
    # output += o

    # define command
    cmd = 'sct_get_centerline -i ' + path_data + folder_data + file_data[0] \
        + ' -method labels ' \
        + ' -l ' + path_data + folder_data + file_data[2] \
        + ' -v 1'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    return status, output


if __name__ == "__main__":
    # call main function
    test()