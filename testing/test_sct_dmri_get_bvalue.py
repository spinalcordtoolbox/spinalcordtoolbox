#!/usr/bin/env python
#########################################################################################
#
# Test function sct_dmri_get_bvalue
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

    # define command
    cmd = 'sct_dmri_get_bvalue -g 0.04 -b 0.04 -d 0.03'

    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()