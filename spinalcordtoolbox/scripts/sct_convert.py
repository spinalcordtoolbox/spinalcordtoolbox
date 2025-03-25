#!/usr/bin/env python
#
# Module converting image files
#
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

# TODO: add output check in convert

import sys
from typing import Sequence

from spinalcordtoolbox.utils.sys import init_sct, set_loglevel
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser
import spinalcordtoolbox.image as image


# PARSER
# ==========================================================================================
def get_parser():
    parser = SCTArgumentParser(
        description='Convert image file to another type.'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        help='File input. Example: `data.nii.gz`',
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-o",
        help="File output (including the file's extension). Example: `data.nii`",
        metavar=Metavar.str,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-squeeze",
        type=int,
        help='Squeeze data dimension (remove unused dimension)',
        choices=(0, 1),
        default=1)

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_profiling_args()

    return parser


def main(argv: Sequence[str]):
    """
    Main function
    :param args:
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # Building the command, do sanity checks
    fname_in = arguments.i
    fname_out = arguments.o
    squeeze_data = bool(arguments.squeeze)

    # convert file
    img = image.Image(fname_in)
    img = image.convert(img, squeeze_data=squeeze_data)
    img.save(fname_out, mutable=True, verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
