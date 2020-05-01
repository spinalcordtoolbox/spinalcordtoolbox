#!/usr/bin/env python
# -*- coding: utf-8
#
# CLI script to crop an image.
#
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT


import sys
import os
import argparse

from spinalcordtoolbox.cropping import ImageCropper, BoundingBox
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import Metavar, SmartFormatter
import sct_utils as sct


def get_parser():

    # Mandatory arguments
    parser = argparse.ArgumentParser(
        description="Tools to crop an image. Either via command line or via a Graphical User Interface (GUI). See "
                    "example usage at the end.",
        epilog="EXAMPLES:\n"
               "- To crop an image using the GUI (this does not allow to crop along the right-left dimension):\n"
               "sct_crop_image -i t2.nii.gz -g 1\n\n"
               "- To crop an image using a binary mask:\n"
               "sct_crop_image -i t2.nii.gz -m mask.nii.gz\n\n"
               "- To crop an image using a reference image:\n"
               "sct_crop_image -i t2.nii.gz -ref mt1.nii.gz\n\n"
               "- To crop an image by specifying min/max (you don't need to specify all dimensions). In the example "
               "below, cropping will occur between x=5 and x=60, and between z=5 and z=zmax-1\n"
               "sct_crop_image -i t2.nii.gz -xmin 5 -xmax 60 -zmin 5 -zmax -2\n\n",
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip('.py'))

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-i',
        required=True,
        help="Input image. Example: t2.nii.gz",
        metavar=Metavar.file,
        )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        help="Show this help message and exit")
    optional.add_argument(
        '-o',
        help="Output image. By default, the suffix '_crop' will be added to the input image.",
        metavar=Metavar.str,
    )
    optional.add_argument(
        '-g',
        type=int,
        help="0: Cropping via command line | 1: Cropping via GUI. Has priority over -m.",
        choices=(0, 1),
        default=0,
    )
    optional.add_argument(
        '-m',
        help="Binary mask that will be used to extract bounding box for cropping the image. Has priority over -ref.",
        metavar=Metavar.file,
        )
    optional.add_argument(
        '-ref',
        help="Image which dimensions (in the physical coordinate system) will be used as a reference to crop the "
             "input image. Only works for 3D images. Has priority over min/max method.",
        metavar=Metavar.file,
    )
    optional.add_argument(
        '-xmin',
        type=int,
        default=0,
        help="Lower bound for cropping along X.",
        metavar=Metavar.int,
        )
    optional.add_argument(
        '-xmax',
        type=int,
        default=-1,
        help="Higher bound for cropping along X. Setting '-1' will crop to the maximum dimension (i.e. no change), "
             "'-2' will crop to the maximum dimension minus 1 slice, etc.",
        metavar=Metavar.int,
        )
    optional.add_argument(
        '-ymin',
        type=int,
        default=0,
        help="Lower bound for cropping along Y.",
        metavar=Metavar.int,
        )
    optional.add_argument(
        '-ymax',
        type=int,
        default=-1,
        help="Higher bound for cropping along Y. Follows the same rules as xmax.",
        metavar=Metavar.int,
        )
    optional.add_argument(
        '-zmin',
        type=int,
        default=0,
        help="Lower bound for cropping along Z.",
        metavar=Metavar.int,
        )
    optional.add_argument(
        '-zmax',
        type=int,
        default=-1,
        help="Higher bound for cropping along Z. Follows the same rules as xmax.",
        metavar=Metavar.int,
        )
    optional.add_argument(
        '-b',
        type=int,
        default=None,
        help="If this flag is declared, the image will not be cropped (i.e. the dimension will not change). Instead, "
             "voxels outside the bounding box will be set to the value specified by this flag. For example, to have "
             "zeros outside the bounding box, use: '-b 0'",
        metavar=Metavar.int,
        )
    optional.add_argument(
        "-v",
        type=int,
        help="0: Verbose off | 1: Verbose on",
        choices=(0, 1),
        default=1
    )

    return parser


def main(args=None):
    """
    Main function
    :param args:
    :return:
    """
    # get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    parser = get_parser()
    arguments = parser.parse_args(args=args)

    # initialize ImageCropper
    cropper = ImageCropper(Image(arguments.i))
    cropper.verbose = arguments.v
    sct.init_sct(log_level=cropper.verbose, update=True)  # Update log level

    # Switch across cropping methods
    if arguments.g:
        cropper.get_bbox_from_gui()
    elif arguments.m:
        cropper.get_bbox_from_mask(Image(arguments.m))
    elif arguments.ref:
        cropper.get_bbox_from_ref(Image(arguments.ref))
    else:
        cropper.get_bbox_from_minmax(
            BoundingBox(arguments.xmin, arguments.xmax,
                        arguments.ymin, arguments.ymax,
                        arguments.zmin, arguments.zmax)
        )

    # Crop image
    img_crop = cropper.crop(background=arguments.b)

    # Write cropped image to file
    if arguments.o is None:
        fname_out = sct.add_suffix(arguments.i, '_crop')
    else:
        fname_out = arguments.o
    img_crop.save(fname_out)

    sct.display_viewer_syntax([arguments.i, fname_out])


if __name__ == "__main__":
    sct.init_sct()
    main()
