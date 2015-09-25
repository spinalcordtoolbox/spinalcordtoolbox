#!/usr/bin/env python
#########################################################################################
#
# Test function sct_detect_vertebral_levels
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import commands
import sys
# append path that contains scripts, to be able to load modules
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
sys.path.append(path_sct + '/scripts')
from msct_image import Image
from numpy import any

def test(data_path):

    output = ''
    status = 0

    # parameters
    folder_data = ['t2/']
    file_data = ['t2.nii.gz', 't2_seg.nii.gz', 't2_seg_labeled.nii.gz']

    # define command
    cmd = 'sct_label_vertebrae -i ' + data_path + folder_data[0] + file_data[0] \
        + ' -seg ' + data_path + folder_data[0] + file_data[1] \
        + ' -initz 34,3 ' \
        + ' -o t2_seg_labeled_totest.nii.gz'

    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # if command ran without error, test integrity
    if status == 0:
        # compare with gold-standard labeling
        data_original = Image(data_path + folder_data[0] + file_data[2]).data
        data_totest = Image('t2_seg_labeled_totest.nii.gz').data
        # check if non-zero elements are present when computing the difference of the two images
        if any(data_original - data_totest):
            status = 99
            output += '\nResulting image differs from gold-standard.'

    return status, output


if __name__ == "__main__":
    # call main function
    test()
