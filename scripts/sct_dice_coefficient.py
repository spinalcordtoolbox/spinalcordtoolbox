#!/usr/bin/env python
#
# This program is a warper for the isct_dice_coefficient binary
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Sara Dupont
# Modified: 2017-07-05 (charley)
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import sys
import shutil
import os

from msct_parser import Parser
import sct_utils as sct
from msct_image import Image
from sct_image import copy_header

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
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files.",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description='Verbose.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])

    return parser

if __name__ == "__main__":
    sct.start_stream_logger()
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    fname_input1 = arguments['-i']
    fname_input2 = arguments['-d']

    verbose = arguments['-v']
    if verbose == '0':
        cmd += ' -v '

    tmp_dir = sct.tmp_create(verbose=verbose)  # create tmp directory
    tmp_dir = os.path.abspath(tmp_dir)

    # copy input files to tmp directory
    # for fname in [fname_input1, fname_input2]:
    shutil.copy(fname_input1, tmp_dir)
    shutil.copy(fname_input2, tmp_dir)
    fname_input1 = ''.join(sct.extract_fname(fname_input1)[1:])
    fname_input2 = ''.join(sct.extract_fname(fname_input2)[1:])

    os.chdir(tmp_dir) # go to tmp directory

    if '-bin' in arguments:
        fname_input1_bin = sct.add_suffix(fname_input1, '_bin')
        sct.run('sct_maths -i ' + fname_input1 + ' -bin 0 -o ' + fname_input1_bin)
        fname_input1 = fname_input1_bin
        fname_input2_bin = sct.add_suffix(fname_input2, '_bin')
        sct.run('sct_maths -i ' + fname_input2 + ' -bin 0 -o ' + fname_input2_bin)
        fname_input2 = fname_input2_bin

    # copy header of im_1 to im_2
    im_1, im_2 = Image(fname_input1), Image(fname_input2)
    im_2_cor = copy_header(im_1, im_2)
    im_2_cor.save()

    cmd = 'isct_dice_coefficient ' + fname_input1 + ' ' + fname_input2

    if '-2d-slices' in arguments:
        cmd += ' -2d-slices ' + arguments['-2d-slices']
    if '-b' in arguments:
        bounding_box = ' '.join(arguments['-b'])
        cmd += ' -b ' + bounding_box
    if '-bmax' in arguments and arguments['-bmax'] == '1':
        cmd += ' -bmax'
    if '-bzmax' in arguments and arguments['-bzmax'] == '1':
        cmd += ' -bzmax'
    if '-o' in arguments:
        path_output, fname_output, ext = sct.extract_fname(arguments['-o'])
        cmd += ' -o ' + fname_output + ext

    if '-r' in arguments:
        rm_tmp = bool(int(arguments['-r']))

    # # Computation of Dice coefficient using Python implementation.
    # # commented for now as it does not cover all the feature of isct_dice_coefficient
    # #from msct_image import Image, compute_dice
    # #dice = compute_dice(Image(fname_input1), Image(fname_input2), mode='3d', zboundaries=False)
    # #sct.printv('Dice (python-based) = ' + str(dice), verbose)

    status, output = sct.run(cmd, verbose)

    os.chdir('..') # go back to original directory

    # copy output file into original directory
    if '-o' in arguments:
        shutil.copy(tmp_dir+'/'+fname_output+ext, path_output+fname_output+ext)

    # remove tmp_dir
    if rm_tmp:
        shutil.rmtree(tmp_dir)    

    sct.printv(output, verbose)