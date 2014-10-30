#!/usr/bin/env python
#########################################################################################
#
# Test function sct_check_atlas_integrity
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands


def test(data_path):

    # parameters
    folder_data = ['template/label/atlas/']

    # define command
    cmd = 'sct_check_atlas_integrity -i ' + data_path + folder_data[0]

    #return
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()