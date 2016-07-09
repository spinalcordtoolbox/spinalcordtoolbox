#!/usr/bin/env python
#########################################################################################
#
# Test function sct_label_utils
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add test to other processes.

import commands

def test(data_path):

    # parameters
    folder_data = ['t2/']
    file_data = ['t2_seg.nii.gz', 't2_seg_labeled.nii.gz']
    output = ''
    status = 0
    list_status = []

    # TEST CREATE
    cmd = 'sct_label_utils -i ' + data_path + folder_data[0] + file_data[0] + ' -create 1,1,1,1:2,2,2,2'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    list_status.append(s)
    output += o

    # TEST cubic-to-point
    cmd = 'sct_label_utils -i ' + data_path + folder_data[0] + file_data[1] + ' -cubic-to-point -o test_centerofmass.nii.gz'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    list_status.append(s)
    output += o

    # Integrity testing
    if s == 0:
        # compute center of mass of labeled segmentation
        status_mass, output_mass = commands.getstatusoutput('sct_label_utils -i test_centerofmass.nii.gz -display')
        centers_of_mass = '30,25,25,4:30,42,25,3:31,9,25,5:32,0,25,6:30,52,26,2'
        if output_mass.split('\n')[-1] != centers_of_mass:
            output += 'WARNING: Center of mass different from gold-standard. \n--> Results:   ' + output_mass.split('\n')[-1] + '\n--> Should be: ' + centers_of_mass + '\n'
            list_status.append(99)

    # check if at least one integrity status was equal to 99
    if 99 in list_status:
        status = 99

    return status, output


if __name__ == "__main__":
    # call main function
    test()