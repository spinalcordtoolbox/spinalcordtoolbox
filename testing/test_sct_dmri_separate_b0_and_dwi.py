#!/usr/bin/env python
#########################################################################################
#
# Test function sct_dmri_separate_b0_and_dwi
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/10/09
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#import sct_utils as sct
import commands

def test(data_path):

    # parameters
    folder_data = 'dmri/'
    file_data = ['dmri.nii.gz', 'bvecs.txt', 'dwi.nii.gz']

    output = ''
    status = 0

    # define command
    cmd = 'sct_dmri_separate_b0_and_dwi -i ' + data_path + folder_data + file_data[0] \
          + ' -b ' + data_path + folder_data + file_data[1]\
          + ' -a 1'\
          + ' -ofolder ./'\
          + ' -v 1'\
          + ' -r 0'
    # return
    #return sct.run(cmd, 0)
    output += '\n====================================================================================================\n'+cmd+'\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    if status == 0:
        from msct_image import Image
        from numpy import sum
        threshold = 1e-3
        ref_dwi = Image(data_path+folder_data+file_data[2])
        new_dwi = Image('dwi.nii.gz')
        diff_dwi = ref_dwi.data-new_dwi.data
        if sum(diff_dwi) > threshold:
            status = 99
            output += '\nResulting DWI image differs from gold-standard.\n'

        ref_b0 = Image(data_path+folder_data+'dmri_T0000.nii.gz')
        new_b0 = Image('b0.nii.gz')
        diff_b0 = ref_b0.data - new_b0.data
        if sum(diff_b0) > threshold:
            status = 99
            output = '\nResulting b0 image differs from gold-standard.\n'

    return status, output

if __name__ == "__main__":
    # call main function
    test()