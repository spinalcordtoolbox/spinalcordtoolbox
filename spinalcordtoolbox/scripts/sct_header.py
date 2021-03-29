#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""
Utility functions for processing the headers of Nifti1 images.
"""

__author__ = "Joshua Newton"
__email__ = "joshuacwnewton@gmail.com"
__copyright__ = "Copyright (c) 2021 Polytechnique Montreal <www.neuro.polymtl.ca>"

# --------------------------------------------------------------------------------

import sys
from typing import Sequence

import nibabel as nib

from spinalcordtoolbox.header import format_header, DISPLAY_FORMATS
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar
from spinalcordtoolbox.utils.sys import init_sct, set_global_loglevel, printv


def get_parser():
    """Set up a command-line argument parser for the main function."""
    parser = SCTArgumentParser(
        description=__doc__  # Reuse the module docstring for the argparse help description
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit"
    )
    parser.add_argument(
        '-v',
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
    )

    subparsers = parser.add_subparsers(dest='command')
    # required=True can't be passed to parser.addsubparsers until Py3.7, see https://stackoverflow.com/a/55834365
    subparsers.required = True

    display_parser = subparsers.add_parser('display', help="Subcommand for displaying the header.")
    display_parser.add_argument(
        "image",
        metavar=Metavar.file,
        help="Input image to get header from."
    )
    display_parser.add_argument(
        '-format',
        choices=DISPLAY_FORMATS,
        default='sct',
        help="Which output format to use for the header. Choose 'nibabel' or 'fslhd' if you already have "
             "header information from those programs and you want to compute the 'diff' between two headers."
    )

    return parser


def main(argv: Sequence[str]):
    """
    Main function. When this script is run via CLI, the main function is called using main(sys.argv[1:]).

    :param argv: A list of unparsed arguments, which is passed to ArgumentParser.parse_args()
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    im_in = nib.load(arguments.image)

    if arguments.command == "display":
        printv(format_header(image=im_in, output_format=arguments.format), verbose=verbose)


if __name__ == '__main__':
    init_sct()
    main(sys.argv[1:])
