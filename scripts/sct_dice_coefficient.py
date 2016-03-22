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
import sys

from msct_parser import Parser
import sct_utils as sct


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Compute the Dice Coefficient. Note: indexing (in both time and space) starts with 0 not 1! Inputting -1 for a size will set it to the full image extent for that dimension.')
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
    parser.add_option(name='-2d-slices',
                      type_value='multiple_choice',
                      description='Compute DC on 2D slices in the specified dimension',
                      mandatory=False,
                      example=['0', '1', '2'])
    parser.add_option(name='-b',
                      type_value=[[','], 'int'],
                      description='Bounding box with the coordinates of the origin and the size of the box as follow: x_origin,x_size,y_origin,y_size,z_origin,z_size',
                      mandatory=False,
                      example='5,10,5,10,10,15')
    parser.add_option(name='-bmax',
                      type_value='multiple_choice',
                      description='Use maximum bounding box of the images union to compute DC',
                      mandatory=False,
                      example=['0', '1'])
    parser.add_option(name='-bzmax',
                      type_value='multiple_choice',
                      description='Use maximum bounding box of the images union in the "Z" direction to compute DC',
                      mandatory=False,
                      example=['0', '1'])
    parser.add_option(name='-bin',
                      type_value='multiple_choice',
                      description='Binarize image before computing DC. (Put non-zero-voxels to 1)',
                      mandatory=False,
                      example=['0', '1'])
    parser.add_option(name='-o',
                      type_value='file_output',
                      description='Output file with DC results (.txt)',
                      mandatory=False,
                      example='dice_coeff.txt')
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='Verbose.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])

    return parser

if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    fname_input1 = arguments['-i']
    fname_input2 = arguments['-d']

    if '-bin' in arguments:
        fname_input1_bin = sct.add_suffix(fname_input1, '_bin')
        sct.run('sct_maths -i '+fname_input1+' -bin -o '+fname_input1_bin)
        fname_input1 = fname_input1_bin
        fname_input2_bin = sct.add_suffix(fname_input2, '_bin')
        sct.run('sct_maths -i '+fname_input2+' -bin -o '+fname_input2_bin)
        fname_input2 = fname_input2_bin

    cmd = 'isct_dice_coefficient '+fname_input1+' '+fname_input2

    if '-2d-slices' in arguments:
        cmd += ' -2d-slices '+arguments['-2d-slices']
    if '-b' in arguments:
        bounding_box = ' '.join(arguments['-b'])
        cmd += ' -b '+bounding_box
    if '-bmax' in arguments and arguments['-bmax'] == '1':
        cmd += ' -bmax'
    if '-bzmax' in arguments and arguments['-bzmax'] == '1':
        cmd += ' -bzmax'
    if '-o' in arguments:
        cmd += ' -o '+arguments['-o']

    verbose = arguments['-v']
    if verbose == '0':
        cmd += ' -v '

    status, output = sct.run(cmd, verbose)
    sct.printv(output, verbose)
