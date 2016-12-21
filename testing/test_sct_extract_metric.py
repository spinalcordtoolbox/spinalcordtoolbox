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

import commands
import os


def test(path_data):

    folder_data = ['mt/', 'label/atlas']
    file_data = ['mtr.nii.gz']
    file_output = ['quantif_mtr.txt']

    input_file = os.path.join(path_data, 'mt', 'mtr.nii.gz')
    output = os.path.join(path_data, 'mt', 'label', 'atlas')

    # define command
    cmd = 'sct_extract_metric' \
        ' -i ' + input_file + \
        ' -f ' + output + \
        ' -method wath '+ \
        ' -vert 1:3'+ \
        ' -o quantif_mtr.txt' \
        ' -v 1'

    return commands.getstatusoutput(cmd)


# call to function
if __name__ == "__main__":
    test()
