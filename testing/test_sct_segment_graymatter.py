#!/usr/bin/env python
#########################################################################################
#
# Test function sct_segment_graymatter
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# modified: 2015/08/31
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import commands
import sys
# append path that contains scripts, to be able to load modules
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
sys.path.append(path_sct + '/scripts')
from msct_image import Image
from numpy import sum


def test(data_path):
    output = ''
    status = 0

    # parameters
    folder_data = 'mt/'
    file_data = ['mt0.nii.gz', 'mt0_seg.nii.gz', 'label/template/MNI-Poly-AMU_level.nii.gz', 'mt0_gmseg.nii.gz']

    # define command
    cmd = 'sct_segment_graymatter -i ' + data_path + folder_data + file_data[0] \
        + ' -s ' + data_path + folder_data + file_data[1] \
        + ' -l ' + data_path + folder_data + file_data[2] \
        + ' -normalize 1 '\
        + ' -v 1'

    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    # run command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # if command ran without error, test integrity
    if status == 0:
	threshold = 1e-2
        # compare with gold-standard labeling
        data_original = Image(data_path + folder_data + file_data[-1]).data
        data_totest = Image('mt0_gmseg.nii.gz').data
        # check if non-zero elements are present when computing the difference of the two images
	diff = data_original - data_totest
	
        if abs(sum(diff))> threshold:
	    Image(param=diff, absolutepath='res_differences_from_gold_standard.nii.gz').save()
            status = 99
            output += '\nResulting image differs from gold-standard.'

    return status, output


if __name__ == "__main__":
    # call main function
    test(path_sct+'/data')
