#!/usr/bin/env python
#########################################################################################
#
# Module converting image files
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add output check in convert

import sys
import os
import argparse

from spinalcordtoolbox.utils import Metavar, SmartFormatter, init_sct, printv
import spinalcordtoolbox.image as image


# DEFAULT PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.verbose = 1


# PARSER
# ==========================================================================================
def get_parser():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description='Convert image file to another type.',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

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

    return parser


# conversion
# ==========================================================================================
def convert(fname_in, fname_out, squeeze_data=True, dtype=None, verbose=1):
    """
    Convert data
    :return True/False
    """
    printv('sct_convert -i ' + fname_in + ' -o ' + fname_out, verbose, 'code')

    img = image.Image(fname_in)
    img = image.convert(img, squeeze_data=squeeze_data, dtype=dtype)
    img.save(fname_out, mutable=True, verbose=verbose)


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
    # Building the command, do sanity checks
    fname_in = arguments.i
    fname_out = arguments.o
    squeeze_data = bool(arguments.squeeze)

    # convert file
    convert(fname_in, fname_out, squeeze_data=squeeze_data)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    init_sct()
    # initialize parameters
    param = Param()
    # call main function
    main()
