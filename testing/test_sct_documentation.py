#!/usr/bin/env python
#########################################################################################
#
# Test function sct_documentation
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct



def test(data_path):

    # define command
    cmd = 'sct_propseg'

    # return
    #return sct.run(cmd, 0)
    return sct.run(cmd)


if __name__ == "__main__":
    # call main function
    test()