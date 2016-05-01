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

    # define command
    cmd = 'sct_label_utils -i ' + data_path + folder_data[0] + file_data[0] + ' -create 1,1,1,1:2,2,2,2'
    status, output = commands.getstatusoutput(cmd)

    # if command ran without error, test integrity
    if status == 0:
        # compute center of mass of
        commands.getstatusoutput('sct_label_utils -i ' + data_path + folder_data[0] + file_data[1] + ' -cubic-to-point -o test_centerofmass.nii.gz')
        status_mass, output_mass = commands.getstatusoutput('sct_label_utils -i test_centerofmass.nii.gz -display')
        centers_of_mass = '30,25,25,4:30,42,25,3:31,9,25,5:32,0,25,6:30,52,26,2'
        if output_mass.split('\n')[-1] != centers_of_mass:
            output_mass = 'WARNING: Center of mass different from gold-standard. \n--> Results:   ' + output_mass.split('\n')[-1] + '\n--> Should be: ' + centers_of_mass + '\n'
            status_mass = 99

        # check if at least one integrity status was equal to 99
        if status_mass == 99:
            status = 99

        # concatenate outputs
        output = output_mass

    return status, output


if __name__ == "__main__":
    # call main function
    test()