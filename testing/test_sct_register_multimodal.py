#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_multimodal script
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

    folder_data = 'mt/'
    file_data = ['mt0.nii.gz', 'mt1.nii.gz']

    output = ''
    status = 0
    possible_algos = ['slicereg2d_pointwise','slicereg2d_translation','slicereg2d_rigid','slicereg2d_affine','slicereg2d_syn','slicereg2d_bsplinesyn','slicereg']
    for algo in possible_algos:
        cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
              + ' -d ' + path_data + folder_data + file_data[1] \
              + ' -o data_reg.nii.gz'  \
              + ' -p step=1,algo=syn,iter=1,smooth=0,shrink=4,metric=MeanSquares'  \
              + ' -x linear' \
              + ' -r 0' \
              + ' -v 1'
        output += cmd+'\n'  # copy command
        s, o = commands.getstatusoutput(cmd)
        status += s
        output += '*****************************************************************************************************\n' \
                  'OUTPUT FROM TEST '+algo+': '
        output += o
        output += '*****************************************************************************************************\n'

    '''
    cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -o data_reg.nii.gz'  \
          + ' -p step=1,algo=syn,iter=1,smooth=0,shrink=4,metric=MeanSquares'  \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'
    output += cmd+'\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += '*****************************************************************************************************\n' \
              'OUTPUT FROM TEST 1: '
    output += o
    output += '*****************************************************************************************************\n'

    # check other method
    cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -o data_reg.nii.gz'  \
          + ' -p step=1,algo=slicereg,iter=1,smooth=0,shrink=4,metric=MeanSquares'  \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'
    output += cmd+'\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # check other method
    cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -o data_reg.nii.gz'  \
          + ' -p step=1,algo=slicereg2d_affine,iter=1,smooth=0,shrink=4,metric=MeanSquares'  \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'
    output += cmd+'\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    '''

    return status, output

if __name__ == "__main__":
    # call main function
    test()