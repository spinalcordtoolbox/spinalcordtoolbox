#!/usr/bin/env python
#
# Flatten spinal cord in sagittal plane.
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import logging
from typing import Sequence

from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser, display_viewer_syntax
from spinalcordtoolbox.flattening import flatten_sagittal

logger = logging.getLogger(__name__)


# Default parameters
class Param:
    def __init__(self):
        self.debug = 0
        self.interp = 'sinc'  # final interpolation
        self.remove_temp_files = 1  # remove temporary files
        self.verbose = 1


def main(argv: Sequence[str]):
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
    set_loglevel(verbose=verbose, caller_module_name=__name__)

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

    display_viewer_syntax([fname_anat, fname_out], verbose=verbose)


def get_parser():
    parser = SCTArgumentParser(
        description="Flatten the spinal cord such within the medial sagittal plane. Useful to make nice pictures. "
                    "Output data has suffix _flatten.\n"
                    "\n"
                    "Notes:\n"
                    " - It is recommended to use this image for visualization purposes only.\n"
                    " - The magnitude of the voxel values will be close, but won't precisely match the input image due "
                    "to the linear interpolation performed during the flattening process.\n"
    )
    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Input volume. Example: `t2.nii.gz`"
    )
    mandatory.add_argument(
        '-s',
        metavar=Metavar.file,
        help="Spinal cord segmentation or centerline. Example: `t2_seg.nii.gz`"
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
