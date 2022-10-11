#!/usr/bin/env python
##############################################################################
#
# Download data using http.
#
# ----------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
###############################################################################

import os
import sys
from typing import Sequence

from spinalcordtoolbox.download import install_data, DATASET_DICT
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel


def get_parser():
    parser = SCTArgumentParser(
        description="Download binaries from the web."
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-d',
        required=True,
        choices=list(DATASET_DICT.keys()),
        help="Name of the dataset."
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="Path to a directory to save the downloaded data.\n"
             "(If not provided, the dataset will be downloaded to the SCT installation directory by default. Directory will be created if it does not exist. Warning: existing "
             "data in the directory will be erased unless -k is provided.)\n"
    )
    optional.add_argument(
        '-k',
        action="store_true",
        help="Keep existing data in destination directory."
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


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    data_name = arguments.d
    if arguments.o is not None:
        dest_folder = arguments.o
    else:
        dest_folder = DATASET_DICT[data_name]['default_location']

    url = DATASET_DICT[data_name]["mirrors"]
    install_data(url, dest_folder, keep=arguments.k)

    printv('Done!\n', verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
