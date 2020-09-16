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

from __future__ import absolute_import

import os

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from spinalcordtoolbox import __sct_dir__

import sct_apply_transfo


def init(param_test):
    """
    Initialize class: param_test
    """
    default_args = [
        '-i t2/t2.nii.gz -s t2/t2_seg-manual.nii.gz -l t2/labels.nii.gz -param step=1,type=seg,algo=centermassrot,metric=MeanSquares:step=2,type=seg,algo=bsplinesyn,iter=5,metric=MeanSquares -t template -qc qc-testing',
        '-i t2/t2.nii.gz -s t2/t2_seg-manual.nii.gz -ldisc t2/labels.nii.gz -ref subject',
    ]
    param_test.file_seg = 't2/t2_seg-manual.nii.gz'
    param_test.fname_gt = ['template/template/PAM50_small_cord.nii.gz', os.path.join(__sct_dir__, 'data/PAM50/template/PAM50_cord.nii.gz')]

    param_test.dice_threshold = 0.9

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # fetch index of the test being performed
    index_args = param_test.default_args.index(param_test.args)

    # apply transformation to binary mask: template --> anat
    sct_apply_transfo.main(args=[
        '-i', param_test.fname_gt[index_args],
        '-d', param_test.file_seg,
        '-w', 'warp_template2anat.nii.gz',
        '-o', 'test_template2anat.nii.gz',
        '-x', 'nn',
        '-v', '0'])

    # apply transformation to binary mask: anat --> template
    sct_apply_transfo.main(args=[
        '-i', param_test.file_seg,
        '-d', param_test.fname_gt[index_args],
        '-w', 'warp_anat2template.nii.gz',
        '-o', 'test_anat2template.nii.gz',
        '-x', 'nn',
        '-v', '0'])

    # compute dice coefficient between template segmentation warped to anat and segmentation from anat
    im_seg = Image(param_test.file_seg)
    im_template_seg_reg = Image('test_template2anat.nii.gz')
    dice_template2anat = msct_image.compute_dice(im_seg, im_template_seg_reg, mode='3d', zboundaries=True)
    # check
    param_test.output += 'Dice[seg,template_seg_reg]: '+str(dice_template2anat)
    if dice_template2anat > param_test.dice_threshold:
        param_test.output += '\n--> PASSED'
    else:
        param_test.status = 99
        param_test.output += '\n--> FAILED'

    # compute dice coefficient between anat segmentation warped to template and segmentation from template
    im_seg_reg = Image('test_anat2template.nii.gz')
    im_template_seg = Image(param_test.fname_gt[index_args])
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
