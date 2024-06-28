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

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        "-i",
        required=True,
        help='File input. Example: data.nii.gz',
        metavar=Metavar.file,
    )
    mandatoryArguments.add_argument(
        "-o",
        required=True,
        help='File output (indicate new extension). Example: data.nii',
        metavar=Metavar.str,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-squeeze",
        type=int,
        help='Sueeze data dimension (remove unused dimension)',
        required=False,
        choices=(0, 1),
        default=1)
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

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
