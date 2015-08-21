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
    folder_data = ['t2/', 'template/', 't2/label/template/']
    file_data = ['t2.nii.gz', 'labels.nii.gz', 't2_seg.nii.gz', 'MNI-Poly-AMU_cord.nii.gz']
    dice_threshold = 0.9

    cmd = 'sct_register_to_template -i ' + path_data + folder_data[0] + file_data[0] \
          + ' -l ' + path_data + folder_data[0] + file_data[1] \
          + ' -s ' + path_data + folder_data[0] + file_data[2] \
          + ' -r 0' \
          + ' -p step=1,type=seg,algo=slicereg,metric=MeanSquares,iter=5:step=2,type=seg,algo=bsplinesyn,iter=3' \
          + ' -t ' + path_data + folder_data[1]
    status, output = commands.getstatusoutput(cmd)

    # if command ran without error, test integrity
    if status == 0:
        # apply transformation to binary mask: template --> anat
        commands.getstatusoutput('sct_apply_transfo -i ' + path_data + folder_data[1] + file_data[3] + ' -d ' + path_data + folder_data[0] + file_data[2] + ' -w warp_template2anat.nii.gz -o test_template2anat.nii.gz -x nn')
        # apply transformation to binary mask: anat --> template
        commands.getstatusoutput('sct_apply_transfo -i ' + path_data + folder_data[0] + file_data[2] + ' -d ' + path_data + folder_data[1] + file_data[3] + ' -w warp_anat2template.nii.gz -o test_anat2template.nii.gz -x nn')
        # compute dice coefficient between template segmentation warped into anat and segmentation from anat
        cmd = 'sct_dice_coefficient ' + path_data + folder_data[0] + file_data[2] + ' test_template2anat.nii.gz'
        status1, output1 = commands.getstatusoutput(cmd)
        # parse output and compare to acceptable threshold
        if float(output1.split('3D Dice coefficient = ')[1]) < dice_threshold:
            status1 = 99
        # compute dice coefficient between segmentation from anat warped into template and template segmentation
        # N.B. here we use -bmax because the FOV of the anat is smaller than the template
        cmd = 'sct_dice_coefficient ' + path_data + folder_data[1] + file_data[3] + ' test_anat2template.nii.gz -bmax'
        status2, output2 = commands.getstatusoutput(cmd)
        # parse output and compare to acceptable threshold
        if float(output2.split('3D Dice coefficient = ')[1]) < dice_threshold:
            status2 = 99
        # check if at least one integrity status was equal to 5
        if status1 == 5 or status2 == 5:
            status = 99
        # concatenate outputs
        output = output1+output2

    return status, output


if __name__ == "__main__":
    # call main function
    test()