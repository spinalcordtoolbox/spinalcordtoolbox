#!/usr/bin/env python
#
# Download data using http.
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
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

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-d',
        choices=['default'] + sorted(list(DATASET_DICT.keys()), key=str.casefold),
        metavar="<dataset>",
        help="Name of the dataset, as listed in the table below. If 'default' is specified, then all default datasets "
             "will be re-downloaded. (Default datasets are critical datasets downloaded during installation.)"
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        '-o',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help=textwrap.dedent("""
            Path to a directory to save the downloaded data.

            (If not provided, the dataset will be downloaded to the SCT installation directory by default. Directory will be created if it does not exist. Warning: existing data in the directory will be erased unless `-k` is provided.)
        """),  # noqa: E501 (line too long)
    )
    optional.add_argument(
        '-k',
        action="store_true",
        help="Keep existing data in destination directory."
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    if arguments.d == "default":
        install_default_datasets(keep=arguments.k)
    else:
        keep = arguments.k
        dest_folder = arguments.o

        # Make sure we don't accidentally overwrite a critical user folder
        if dest_folder is not None and os.path.isdir(dest_folder) and os.listdir(dest_folder) and not keep:
            printv(f"Output directory '{dest_folder}' exists and is non-empty. Contents will be erased.",
                   type="warning")

            while True:
                answer = input("Do you wish to overwrite this directory? ([Y]es/[N]o): ")
                if answer.lower() in ["y", "yes"]:
                    break  # keep = False
                elif answer.lower() in ["n", "no"]:
                    keep = True
                    break
                else:
                    printv(f"Invalid input '{answer}'", type="warning")

            printv("Note: You can suppress this message by specifying `-k` (keep) or by deleting the "
                   "directory in advance.", type="warning")

        install_named_dataset(arguments.d, dest_folder=dest_folder, keep=keep)

    printv('Done!\n', verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
