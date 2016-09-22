#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_warp_template script
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


#import sct_utils as sct
import commands


def test(path_data):

    # parameters
    folder_data = ['mt/', 'template/']
    file_data = ['mt1.nii.gz', 'warp_template2mt.nii.gz']

    # define command
    cmd = 'sct_warp_template' \
        ' -d '+path_data+folder_data[0]+file_data[0]+ \
        ' -w '+path_data+folder_data[0]+file_data[1]+ \
        ' -a 0 '+ \
        ' -s 0 '+ \
        ' -ofolder label'+ \
        ' -t '+path_data+folder_data[1]+ \
        ' -qc 0'+ \
        ' -v 1'

    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


# call to function
if __name__ == "__main__":
    # call main function
    test()
