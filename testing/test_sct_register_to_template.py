#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_to_template
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os

from pandas import DataFrame
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
import sct_apply_transfo

default_args = [
 '-i t2/t2.nii.gz -l t2/labels.nii.gz -s t2/t2_seg.nii.gz -param step=1,type=seg,algo=centermassrot,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=5,metric=MeanSquares -t template -r 0 -igt template/template/PAM50_small_cord.nii.gz -qc testing-qc',
 '-i t2/t2.nii.gz -s t2/t2_seg.nii.gz -ldisc t2/labels.nii.gz -ref subject',
]

def init(param_test):
    """
    Initialize class: param_test
    """

    param_test.dice_threshold = 0.9

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """

    if param_test.args == default_args[1]:
        return param_test # no integrity test

    # apply transformation to binary mask: template --> anat
    sct_apply_transfo.main(args=[
        '-i', param_test.fname_gt,
        '-d', param_test.dict_args_with_path['-s'],
        '-w', os.path.join(param_test.path_output, 'warp_template2anat.nii.gz'),
        '-o', os.path.join(param_test.path_output, 'test_template2anat.nii.gz'),
        '-x', 'nn',
        '-v', '0'])

    # apply transformation to binary mask: anat --> template
    sct_apply_transfo.main(args=[
        '-i', param_test.dict_args_with_path['-s'],
        '-d', param_test.fname_gt,
        '-w', os.path.join(param_test.path_output, 'warp_anat2template.nii.gz'),
        '-o', os.path.join(param_test.path_output, 'test_anat2template.nii.gz'),
        '-x', 'nn',
        '-v', '0'])

    # compute dice coefficient between template segmentation warped to anat and segmentation from anat
    im_seg = Image(param_test.dict_args_with_path['-s'])
    im_template_seg_reg = Image(os.path.join(param_test.path_output, 'test_template2anat.nii.gz'))
    dice_template2anat = msct_image.compute_dice(im_seg, im_template_seg_reg, mode='3d', zboundaries=True)
    # check
    param_test.output += 'Dice[seg,template_seg_reg]: '+str(dice_template2anat)
    if dice_template2anat > param_test.dice_threshold:
        param_test.output += '\n--> PASSED'
    else:
        param_test.status = 99
        param_test.output += '\n--> FAILED'

    # compute dice coefficient between anat segmentation warped to template and segmentation from template
    im_seg_reg = Image(os.path.join(param_test.path_output, 'test_anat2template.nii.gz'))
    im_template_seg = Image(param_test.fname_gt)
    dice_anat2template = msct_image.compute_dice(im_seg_reg, im_template_seg, mode='3d', zboundaries=True)
    # check
    param_test.output += '\n\nDice[seg_reg,template_seg]: '+str(dice_anat2template)
    if dice_anat2template > param_test.dice_threshold:
        param_test.output += '\n--> PASSED'
    else:
        param_test.status = 99
        param_test.output += '\n--> FAILED'

    # update Panda structure
    param_test.results['dice_template2anat'] = dice_template2anat
    param_test.results['dice_anat2template'] = dice_anat2template

    return param_test
