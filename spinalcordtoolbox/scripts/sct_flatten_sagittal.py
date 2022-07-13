#!/usr/bin/env python
#########################################################################################
#
# Flatten spinal cord in sagittal plane.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2014-06-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import logging

from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, display_viewer_syntax, set_loglevel
from spinalcordtoolbox.flattening import flatten_sagittal

logger = logging.getLogger(__name__)


# Default parameters
class Param:
    def __init__(self):
        self.debug = 0
        self.interp = 'sinc'  # final interpolation
        self.remove_temp_files = 1  # remove temporary files
        self.verbose = 1


def main(argv=None):
    """
    Main function
    :param fname_anat:
    :param fname_centerline:
    :param verbose:
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    fname_anat = arguments.i
    fname_centerline = arguments.s

    # load input images
    im_anat = Image(fname_anat)
    im_centerline = Image(fname_centerline)

    # flatten sagittal
    im_anat_flattened = flatten_sagittal(im_anat, im_centerline, verbose)

    # save output
    fname_out = add_suffix(fname_anat, '_flatten')
    im_anat_flattened.save(fname_out)

    display_viewer_syntax([fname_anat, fname_out])


def get_parser():
    parser = SCTArgumentParser(
        description="Flatten the spinal cord such within the medial sagittal plane. Useful to make nice pictures. "
                    "Output data has suffix _flatten. Output type is float32 (regardless of input type) to minimize "
                    "loss of precision during conversion."
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Input volume. Example: t2.nii.gz"
    )
    mandatory.add_argument(
        '-s',
        metavar=Metavar.file,
        required=True,
        help="Spinal cord segmentation or centerline. Example: t2_seg.nii.gz"
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
    )

    return parser


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

