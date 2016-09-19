#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_multimodal script
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/09/28
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import commands
import sct_utils as sct

def test(path_data):

    folder_data = 'mt/'
    file_data = ['mt0.nii.gz', 'mt1.nii.gz', 'mt0_seg.nii.gz', 'mt1_seg.nii.gz']
    output = ''
    status = 0

    # check syn
    algo = 'syn'
    cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -o ' + sct.add_suffix(file_data[0], '_reg_'+algo)  \
          + ' -param step=1,algo='+algo+',iter=1,smooth=1,shrink=2,metric=MI'  \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'
    output += '\n====================================================================================================\n'\
              +cmd+\
              '\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    if status == 0:
        s, o = check_integrity(algo=algo, fname_result=sct.add_suffix(file_data[0], '_reg_'+algo), fname_goldstandard=path_data+folder_data+sct.add_suffix(file_data[0], '_reg_'+algo+'_goldstandard'))
        status += s
        output += o

    # check slicereg
    algo = 'slicereg'
    cmd = 'sct_register_multimodal -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -o ' + sct.add_suffix(file_data[0], '_reg_'+algo)  \
          + ' -param step=1,algo='+algo+',iter=5,smooth=0,metric=MeanSquares'  \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'
    output += '\n====================================================================================================\n'\
              +cmd+\
              '\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o
    if status == 0:
        s, o = check_integrity(algo=algo, fname_result=sct.add_suffix(file_data[0], '_reg_'+algo), fname_goldstandard=path_data+folder_data+sct.add_suffix(file_data[0], '_reg_'+algo+'_goldstandard'))
        status += s
        output += o

    # check centermass
    algo = 'centermass'
    cmd = 'sct_register_multimodal'  \
          + ' -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -iseg ' + path_data + folder_data + file_data[2] \
          + ' -dseg ' + path_data + folder_data + file_data[3] \
          + ' -o ' + sct.add_suffix(file_data[0], '_reg_'+algo)  \
          + ' -param step=1,type=seg,algo='+algo+',smooth=1'  \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'
    output += '\n====================================================================================================\n'\
              +cmd+\
              '\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # check centermassrot
    algo = 'centermassrot'
    cmd = 'sct_register_multimodal'  \
          + ' -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -iseg ' + path_data + folder_data + file_data[2] \
          + ' -dseg ' + path_data + folder_data + file_data[3] \
          + ' -o ' + sct.add_suffix(file_data[0], '_reg_'+algo)  \
          + ' -param step=1,type=seg,algo='+algo+',smooth=1'  \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'
    output += '\n====================================================================================================\n'\
              +cmd+\
              '\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    # check columnwise
    algo = 'columnwise'
    cmd = 'sct_register_multimodal'  \
          + ' -i ' + path_data + folder_data + file_data[0] \
          + ' -d ' + path_data + folder_data + file_data[1] \
          + ' -iseg ' + path_data + folder_data + file_data[2] \
          + ' -dseg ' + path_data + folder_data + file_data[3] \
          + ' -o ' + sct.add_suffix(file_data[0], '_reg_'+algo)  \
          + ' -param step=1,type=seg,algo='+algo+',smooth=1'  \
          + ' -x linear' \
          + ' -r 0' \
          + ' -v 1'
    output += '\n====================================================================================================\n'\
              +cmd+\
              '\n====================================================================================================\n\n'  # copy command
    s, o = commands.getstatusoutput(cmd)
    status += s
    output += o

    return status, output


def check_integrity(algo='', fname_result='', fname_goldstandard=''):
    """
    Check integrity between registered image and gold-standard
    :param algo:
    :return:
    """
    status = 0
    output = '\nChecking integrity between: \n  Result: '+fname_result+'\n  Gold-standard: '+fname_goldstandard

    from msct_image import Image
    # compare with gold-standard registration
    im_gold = Image(fname_goldstandard)
    data_gold = im_gold.data
    data_res = Image(fname_result).data
    # get dimensions
    nx, ny, nz, nt, px, py, pz, pt = im_gold.dim
    # set the difference threshold to 1e-3 pe voxel
    threshold = 1e-3 * nx * ny * nz * nt
    # check if non-zero elements are present when computing the difference of the two images
    diff = data_gold - data_res
    # report result
    import numpy as np
    output += '\nDifference between the two images: '+str(abs(np.sum(diff)))
    output += '\nThreshold: '+str(threshold)
    if abs(np.sum(diff)) > threshold:
        Image(param=diff, absolutepath='res_differences_from_gold_standard.nii.gz').save()
        status = 99
        output += '\nWARNING: Difference is higher than threshold.'
    return status, output

if __name__ == "__main__":
    # call main function
    test()