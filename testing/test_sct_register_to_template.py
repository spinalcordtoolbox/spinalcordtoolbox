#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_to_template script
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


#import sct_utils as sct
import commands


def test(path_data):
    folder_data = ['t2/', 'template/']
    file_data = ['t2.nii.gz', 'labels.nii.gz', 't2_seg.nii.gz']

    cmd = 'sct_register_to_template -i ' + path_data + folder_data[0] + file_data[0] \
          + ' -l ' + path_data + folder_data[0] + file_data[1] \
          + ' -m ' + path_data + folder_data[0] + file_data[2] \
          + ' -r 0' \
          + ' -s superfast' \
          + ' -p ' + path_data + folder_data[1]

    '''
    s, output = sct.run(cmd, 0)
    status += s
    cmd = 'sct_WarpImageMultiTransform' \
          + ' 3 ' + path_data + folder_template + file_template[1] \
          + ' templatecord2anat.nii.gz' \
          + ' --use-NN'
    s, output_buf = sct.run(cmd, 0)
    status += s
    output += output_buf
    cmd = 'sct_dice_coefficient ' \
          + path_data + folder_data + file_data[2] \
          + ' templatecord2anat.nii.gz' \
          + ' -bmax'
    s, output_buf = sct.run(cmd, 0)
    status += s
    output += output_buf
    '''

    #return sct.run(cmd, 0)
    return commands.getstatusoutput(cmd)


if __name__ == "__main__":
    # call main function
    test()