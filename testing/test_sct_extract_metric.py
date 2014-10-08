#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_extract_metric script
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


import sct_utils as sct


def test(path_data):

    # parameters
    folder_data = ['mt/', 'label/atlas']
    file_data = ['mtr.nii.gz']

    # define command
    cmd = 'sct_extract_metric' \
        ' -i '+path_data+folder_data[0]+file_data[0]+ \
        ' -f '+path_data+folder_data[0]+folder_data[1]+ \
        ' -l 2,17 '+ \
        ' -m wath '+ \
        ' -v 1:3'+ \
        ' -o quantif_'+file_data[0]+'.txt' \
        ' -v 1'

    # return
    return sct.run(cmd, 0)


# call to function
if __name__ == "__main__":
    # call main function
    test()
