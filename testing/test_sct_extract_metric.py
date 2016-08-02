#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_extract_metric script
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/09/28
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add integrity check

#import sct_utils as sct
import commands


def test(path_data):

    # parameters
    folder_data = ['mt/', 'label/atlas']
    file_data = ['mtr.nii.gz']

    # define command
    cmd = 'sct_extract_metric' \
        ' -i '+path_data+folder_data[0]+file_data[0]+ \
        ' -f '+path_data+folder_data[0]+folder_data[1]+ \
        ' -method wath '+ \
        ' -vert 1:3'+ \
        ' -o quantif_'+file_data[0]+ \
        ' -v 1'

    # return
    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


# call to function
if __name__ == "__main__":
    # call main function
    test()
