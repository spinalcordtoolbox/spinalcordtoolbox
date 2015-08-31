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
# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')

def test(data_path):

    output = ''
    status = 0

    # parameters
    folder_data = ['t2/']
    file_data = ['t2.nii.gz', 't2_seg.nii.gz', 'labels.nii.gz']

    # define command
    cmd = 'sct_detect_vertebral_levels -i ' + data_path + folder_data[0] + file_data[0] \
        + ' -seg ' + data_path + folder_data[0] + file_data[1] \
        + ' -initz 34,3 ' \
        + ' -o t2_seg_labeled_totest.nii.gz'

    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # if command ran without error, test integrity
    # if status == 0:
        # compare with gold-standard labeling

        # check if labels are located in the right vertebrae
        # TODO

        # add function that does MSD between two images

        # from sct_average_data_within_mask import average_within_mask
        # mtr_mean, mtr_std = average_within_mask('mtr.nii.gz', data_path+folder_data+file_data[2], verbose=0)
        # if not (mtr_mean > range_mtr[0] and mtr_mean < range_mtr[1]):
        #     status = 99
        #     output += '\nMean MTR = '+str(mtr_mean)+'\nAuthorized range: '+str(range_mtr)

    return status, output


if __name__ == "__main__":
    # call main function
    test()
