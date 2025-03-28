#!/usr/bin/env python
#
# CLI script to crop an image
#
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
from typing import Sequence
import textwrap

from spinalcordtoolbox.cropping import ImageCropper
from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, display_viewer_syntax, list_type
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel


def get_parser():
    parser = SCTArgumentParser(
        description="Tools to crop an image. Either via command line or via a Graphical User Interface (GUI). See "
                    "example usage at the end.",
        epilog=textwrap.dedent("""
            EXAMPLES:

            - To crop an image using the GUI (this does not allow to crop along the right-left dimension):
              ```
              sct_crop_image -i t2.nii.gz -g 1
              ```
            - To crop an image using a binary mask:
              ```
              sct_crop_image -i t2.nii.gz -m mask.nii.gz
              ```
            - To crop an image using a reference image:
              ```
              sct_crop_image -i t2.nii.gz -ref mt1.nii.gz
              ```
            - To crop an image by specifying min/max (you don't need to specify all dimensions). In the example below, cropping will occur between x=5 and x=60, and between z=5 and z=zmax-1
              ```
              sct_crop_image -i t2.nii.gz -xmin 5 -xmax 60 -zmin 5 -zmax -2
              ```
            - To crop an image using a binary mask, and keep a margin of 5 voxels on each side in the x and y directions only:
              ```
              sct_crop_image -i t2.nii.gz -m mask.nii.gz -dilate 5x5x0
              ```
        """),  # noqa: E501 (line too long)
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        help="Input image. Example: `t2.nii.gz`",
        metavar=Metavar.file,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-o',
        help="Output image. By default, the suffix '_crop' will be added to the input image.",
        metavar=Metavar.str,
    )
    optional.add_argument(
        '-dilate',
        type=list_type('x', int),
        help=textwrap.dedent("""
            Number of extra voxels to keep around the bounding box on each side. Can be specified as a single number, or a list of 3 numbers separated by `x`. For example:

              - `-dilate 5` will add a margin of 5 voxels in each direction
              - `-dilate 2x3x0` will add margin of 2 voxels on each side in the x-axis, 3 voxels on each side in the y-axis, and no extra margin in the z-axis.
        """),
        metavar=Metavar.list,
    ),
    optional.add_argument(
        '-g',
        type=int,
        help="`0`: Cropping via command line | `1`: Cropping via GUI. Has priority over `-m`.",
        choices=(0, 1),
        default=0,
    )
    optional.add_argument(
        '-m',
        help="Binary mask that will be used to extract bounding box for cropping the image. Has priority over `-ref`.",
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
        help="Higher bound for cropping along X. Setting `-1` will crop to the maximum dimension (i.e. no change), "
             "`-2` will crop to the maximum dimension minus 1 slice, etc.",
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
        help="If this flag is declared, the image will not be cropped (i.e. the dimension will not change). Instead, "
             "voxels outside the bounding box will be set to the value specified by this flag. For example, to have "
             "zeros outside the bounding box, use: '-b 0'",
        metavar=Metavar.int,
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def main(argv: Sequence[str]):
    """
    Main function
    :param argv:
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    dilate = arguments.dilate
    if dilate is not None:
        if len(dilate) == 1:
            dilate *= 3
        elif len(dilate) == 3:
            pass  # use dilate as-is
        else:
            parser.error(
                f"Option '-dilate' expected either 1 or 3 numbers, but got "
                f"{len(dilate)} numbers: {'x'.join(str(d) for d in dilate)}."
            )

    # initialize ImageCropper
    cropper = ImageCropper(Image(arguments.i))
    cropper.verbose = verbose

    # Switch across cropping methods
    if arguments.g:
        cropper.get_bbox_from_gui()
    elif arguments.m:
        cropper.get_bbox_from_mask(Image(arguments.m))
    elif arguments.ref:
        cropper.get_bbox_from_ref(Image(arguments.ref))
    else:
        cropper.get_bbox_from_minmax(
            arguments.xmin, arguments.xmax,
            arguments.ymin, arguments.ymax,
            arguments.zmin, arguments.zmax)

    # Crop image
    img_crop = cropper.crop(background=arguments.b, dilate=dilate)

    # Write cropped image to file
    if arguments.o is None:
        fname_out = add_suffix(arguments.i, '_crop')
    else:
        fname_out = arguments.o
    img_crop.save(fname_out)

    display_viewer_syntax([arguments.i, fname_out], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
