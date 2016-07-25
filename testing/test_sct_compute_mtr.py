#!/usr/bin/env python
#########################################################################################
#
# Test function sct_compute_mtr
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/20
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import commands
# import sys
# get path of the toolbox
# status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
# sys.path.append(path_sct + '/scripts')

def test(data_path):

    range_mtr = [31.8, 31.9]
    output = ''
    status = 0

    # parameters
    folder_data = 'mt/'
    file_data = ['mt0_reg_slicereg_goldstandard.nii.gz', 'mt1.nii.gz', 'mt1_seg.nii.gz']

    # define command
    cmd = 'sct_compute_mtr -mt0 ' + data_path + folder_data + file_data[0] \
          + ' -mt1 ' + data_path + folder_data + file_data[1] + ' -r 0'
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # if command ran without error, test integrity
    if status == 0:
        # compute mtr within mask
        from sct_average_data_within_mask import average_within_mask
        mtr_mean, mtr_std = average_within_mask('mtr.nii.gz', data_path+folder_data+file_data[2], verbose=0)
        if not (mtr_mean > range_mtr[0] and mtr_mean < range_mtr[1]):
            status = 99
            output += '\nMean MTR = '+str(mtr_mean)+'\nAuthorized range: '+str(range_mtr)

    return status, output


if __name__ == "__main__":
    # call main function
    test()
