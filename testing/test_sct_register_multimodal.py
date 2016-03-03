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
    file_data = ['mt0.nii.gz', 'mt1.nii.gz', 'mt0_syn_reg_on_mt1.nii.gz']

    output = ''
    status = 0

    algo_default = 'syn'
    cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -o data_'+algo_default+'_reg.nii.gz'  \
          + ' -param step=1,algo='+algo_default+',iter=1,smooth=0,shrink=4,metric=MeanSquares'  \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'
    output += '\n====================================================================================================\n'\
              +cmd+\
              '\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # check other method
    algo = 'slicereg'
    cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -o data_'+algo+'_reg.nii.gz'  \
          + ' -param step=1,algo='+algo+',iter=1,smooth=0,shrink=4,metric=MeanSquares'  \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'
    output += '\n====================================================================================================\n'\
              +cmd+\
              '\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # check other method
    algo = 'affine'
    cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -o data_'+algo+'_reg.nii.gz'  \
          + ' -param step=1,algo='+algo+',iter=1,smooth=0,shrink=4,metric=MeanSquares,slicewise=1'  \
          + ' -r 0' \
          + ' -v 1'
    output += '\n====================================================================================================\n'\
              +cmd+\
              '\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # if command ran without error, test integrity
    if status == 0:
        from msct_image import Image
        # compare with gold-standard registration
        im_gold = Image(path_data + folder_data + file_data[-1])
        data_gold = im_gold.data
        data_res = Image('data_'+algo_default+'_reg.nii.gz').data

        nx, ny, nz, nt, px, py, pz, pt = im_gold.dim
        threshold = 1e-3 * nx * ny * nz * nt  # set the difference threshold to 1e-3 pe voxel
        # check if non-zero elements are present when computing the difference of the two images
        diff = data_gold - data_res
        import numpy as np
        if abs(np.sum(diff))>threshold:
            Image(param=diff, absolutepath='res_differences_from_gold_standard.nii.gz').save()
            status = 99
            output += '\nResulting image differs from gold-standard (sum of the difference of intensity: '+str(np.sum(diff))+').'

    return status, output

if __name__ == "__main__":
    # call main function
    test()