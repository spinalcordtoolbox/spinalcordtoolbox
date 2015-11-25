#!/usr/bin/env python
#
# This program is a warper for the isct_dice_coefficient binary
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Sara Dupont
# Modified: 2015-05-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################
from msct_parser import Parser
import sys
import sct_utils as sct

#TODO: change name to sct_dice_coefficient


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Compute the Dice Coefficient. ')
    parser.add_option(name='-i',
                      type_value='image_nifti',
                      description='First input image.',
                      mandatory=True,
                      example='t2_seg.nii.gz')
    parser.add_option(name='-d',
                      type_value='image_nifti',
                      description='Second input image.',
                      mandatory=True,
                      example='t2_manual_seg.nii.gz')
    parser.add_option(name='-b',
                      type_value=[[','], 'int'],
                      description='Bounding box with the coordinates of the origin and the size of the box as follow: x_origin,x_size,y_origin,y_size,z_origin,z_size',
                      mandatory=False,
                      example='5,10,5,10,10,15')
    parser.add_option(name='-bmax',
                      type_value='multiple_choice',
                      description='',
                      mandatory=False,
                      example=['0', '1'])
    parser.add_option(name='-bin',
                      type_value='multiple_choice',
                      description='Binarize image before computing DC. (Put non-zero-voxels to 1)',
                      mandatory=False,
                      example=['0', '1'])
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='Verbose.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])

    return parser
