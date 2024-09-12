#!/usr/bin/env python
#
# Download data using http.
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
from typing import Sequence
import textwrap

from spinalcordtoolbox.download import install_named_dataset, DATASET_DICT, list_datasets, install_default_datasets
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel


def get_parser():
    parser = SCTArgumentParser(
        description="Download binaries from the web.",
        epilog=list_datasets(),
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-d',
        required=True,
        choices=['default'] + sorted(list(DATASET_DICT.keys()), key=str.casefold),
        metavar="<dataset>",
        help="Name of the dataset, as listed in the table below. If 'default' is specified, then all default datasets "
             "will be re-downloaded. (Default datasets are critical datasets downloaded during installation.)"
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
        help=textwrap.dedent("""
            Path to a directory to save the downloaded data.
            (If not provided, the dataset will be downloaded to the SCT installation directory by default. Directory will be created if it does not exist. Warning: existing data in the directory will be erased unless -k is provided.)
        """),  # noqa: E501 (line too long)
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
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    if arguments.d == "default":
        install_default_datasets(keep=arguments.k)
    else:
        install_named_dataset(arguments.d, dest_folder=arguments.o, keep=arguments.k)

    printv('Done!\n', verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
