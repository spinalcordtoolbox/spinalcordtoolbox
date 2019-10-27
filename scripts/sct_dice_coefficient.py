#!/usr/bin/env python
#
# This program is a wrapper for the isct_dice_coefficient binary
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Sara Dupont
# Modified: 2017-07-05 (charley)
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

import sys
import os
import argparse

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar, SmartFormatter


def get_parser():
    parser = argparse.ArgumentParser(
        description='Compute the Dice Coefficient. '
                    'N.B.: indexing (in both time and space) starts with 0 not 1! Inputting -1 for a '
                    'size will set it to the full image extent for that dimension.',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        required=True,
        metavar=Metavar.file,
        help='First input image. Example: t2_seg.nii.gz',
        )
    mandatory.add_argument(
        '-d',
        required=True,
        help='Second input image. Example: t2_manual_seg.nii.gz',
        metavar=Metavar.file,
        )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        '-2d-slices',
        type=int,
        help='Compute DC on 2D slices in the specified dimension',
        required=False,
        choices=(0, 1, 2))
    optional.add_argument(
        '-b',
        metavar=Metavar.list,
        help='Bounding box with the coordinates of the origin and the size of the box as follow: '
             'x_origin,x_size,y_origin,y_size,z_origin,z_size. Example: 5,10,5,10,10,15',
        required=False)
    optional.add_argument(
        '-bmax',
        type=int,
        help='Use maximum bounding box of the images union to compute DC.',
        required=False,
        choices=(0, 1))
    optional.add_argument(
        '-bzmax',
        type=int,
        help='Use maximum bounding box of the images union in the "Z" direction to compute DC.',
        required=False,
        choices=(0, 1))
    optional.add_argument(
        '-bin',
        type=int,
        help='Binarize image before computing DC. (Put non-zero-voxels to 1)',
        required=False,
        choices=(0, 1))
    optional.add_argument(
        '-o',
        metavar=Metavar.str,
        help='Output file with DC results (.txt). Example: dice_coeff.txt',
        required=False)
    optional.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        required=False,
        default=1,
        choices=(0, 1))
    optional.add_argument(
        '-v',
        type=int,
        help='Verbose.',
        required=False,
        default=1,
        choices=(0, 1))

    return parser

if __name__ == "__main__":
    sct.init_sct()
    parser = get_parser()
    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    fname_input1 = arguments.i
    fname_input2 = arguments.d

    verbose = arguments.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    tmp_dir = sct.tmp_create(verbose=verbose)  # create tmp directory
    tmp_dir = os.path.abspath(tmp_dir)

    # copy input files to tmp directory
    # for fname in [fname_input1, fname_input2]:
    sct.copy(fname_input1, tmp_dir)
    sct.copy(fname_input2, tmp_dir)
    fname_input1 = ''.join(sct.extract_fname(fname_input1)[1:])
    fname_input2 = ''.join(sct.extract_fname(fname_input2)[1:])

    curdir = os.getcwd()
    os.chdir(tmp_dir) # go to tmp directory

    if '-bin' in arguments:
        fname_input1_bin = sct.add_suffix(fname_input1, '_bin')
        sct.run(['sct_maths', '-i', fname_input1, '-bin', '0', '-o', fname_input1_bin])
        fname_input1 = fname_input1_bin
        fname_input2_bin = sct.add_suffix(fname_input2, '_bin')
        sct.run(['sct_maths', '-i', fname_input2, '-bin', '0', '-o', fname_input2_bin])
        fname_input2 = fname_input2_bin

    # copy header of im_1 to im_2
    from spinalcordtoolbox.image import Image
    im_1 = Image(fname_input1)
    im_2 = Image(fname_input2)
    im_2.header = im_1.header
    im_2.save()

    cmd = ['isct_dice_coefficient', fname_input1, fname_input2]

    if vars(arguments)["2d_slices"] is not None:
        cmd += ['-2d-slices', str(vars(arguments)["2d_slices"])]
    if arguments.b is not None:
        bounding_box = arguments.b
        cmd += ['-b'] + bounding_box
    if arguments.bmax is not None and arguments.bmax == 1:
        cmd += ['-bmax']
    if arguments.bzmax is not None and arguments.bzmax == 1:
        cmd += ['-bzmax']
    if arguments.o is not None:
        path_output, fname_output, ext = sct.extract_fname(arguments.o)
        cmd += ['-o', fname_output + ext]

    rm_tmp = bool(arguments.r)

    # # Computation of Dice coefficient using Python implementation.
    # # commented for now as it does not cover all the feature of isct_dice_coefficient
    # #from spinalcordtoolbox.image import Image, compute_dice
    # #dice = compute_dice(Image(fname_input1), Image(fname_input2), mode='3d', zboundaries=False)
    # #sct.printv('Dice (python-based) = ' + str(dice), verbose)

    status, output = sct.run(cmd, verbose, is_sct_binary=True)

    os.chdir(curdir) # go back to original directory

    # copy output file into original directory
    if arguments.o is not None:
        sct.copy(os.path.join(tmp_dir, fname_output + ext), os.path.join(path_output, fname_output + ext))

    # remove tmp_dir
    if rm_tmp:
        sct.rmtree(tmp_dir)

    sct.printv(output, verbose)
