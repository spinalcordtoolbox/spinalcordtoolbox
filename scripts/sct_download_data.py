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
import argparse

from spinalcordtoolbox.download import install_data
from spinalcordtoolbox.utils.shell import Metavar, SmartFormatter, ActionCreateFolder
from spinalcordtoolbox.utils.sys import init_sct, printv


def get_parser(dict_url):
    parser = argparse.ArgumentParser(
        description="Download binaries from the web.",
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-d',
        required=True,
        choices=list(dict_url.keys()),
        help=f"Name of the dataset."
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
        help="R|Path to a directory to save the downloaded data.\n"
             "(Defaults to ./${dataset-name}. Directory will be created if it does not exist. Warning: existing "
             "data in the directory will be erased unless -k is provided.)\n"
    )
    optional.add_argument(
        '-k',
        action="store_true",
        help="Keep existing data in destination directory."
    )
    optional.add_argument(
        '-v',
        choices=['0', '1', '2'],
        default='1',
        help="Verbose. 0: nothing. 1: basic. 2: extended."
    )

    return parser


def main(args=None):

    # Dictionary containing list of URLs for data names.
    # Mirror servers are listed in order of decreasing priority.
    # If exists, favour release artifact straight from github
    dict_url = {
        "sct_example_data": [
            "https://github.com/sct-data/sct_example_data/releases/download/r20180525/20180525_sct_example_data.zip",
            "https://osf.io/kjcgs/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20180525_sct_example_data.zip",
        ],
        "sct_testing_data": [
            "https://github.com/sct-data/sct_testing_data/releases/download/r20201030/sct_testing_data-r20201030.zip",
            "https://osf.io/download/5f9c271187b7df04593b03e0/"],
        "PAM50": [
            "https://github.com/sct-data/PAM50/releases/download/r20201104/PAM50-r20201104.zip", 
            "https://osf.io/download/5fa21326a5bb9d00610a5a21/"],
        "MNI-Poly-AMU": [
            "https://github.com/sct-data/MNI-Poly-AMU/releases/download/r20170310/20170310_MNI-Poly-AMU.zip",
            "https://osf.io/sh6h4/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20170310_MNI-Poly-AMU.zip",
        ],
        "gm_model": [
            "https://osf.io/ugscu/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20160922_gm_model.zip",
        ],
        "optic_models": [
            "https://github.com/sct-data/optic_models/releases/download/r20170413/20170413_optic_models.zip",
            "https://osf.io/g4fwn/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20170413_optic_models.zip",
        ],
        "pmj_models": [
            "https://github.com/sct-data/pmj_models/releases/download/r20170922/20170922_pmj_models.zip",
            "https://osf.io/4gufr/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20170922_pmj_models.zip",
        ],
        "binaries_linux": [
            "https://osf.io/cs6zt/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20200801_sct_binaries_linux.tar.gz",
        ],
        "binaries_osx": [
            "https://osf.io/874cy?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20200801_sct_binaries_osx.tar.gz",
        ],
        "course_hawaii17": "https://osf.io/6exht/?action=download",
        "course_paris18": [
            "https://osf.io/9bmn5/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20180612_sct_course-paris18.zip",
        ],
        "course_london19": [
            "https://osf.io/4q3u7/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20190121_sct_course-london19.zip",
        ],
        "course_beijing19": [
            "https://osf.io/ef4xz/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20190802_sct_course-beijing19.zip",
        ],
        "deepseg_gm_models": [
            "https://github.com/sct-data/deepseg_gm_models/releases/download/r20180205/20180205_deepseg_gm_models.zip",
            "https://osf.io/b9y4x/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20180205_deepseg_gm_models.zip",
        ],
        "deepseg_sc_models": [
            "https://github.com/sct-data/deepseg_sc_models/releases/download/r20180610/20180610_deepseg_sc_models.zip",
            "https://osf.io/avf97/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20180610_deepseg_sc_models.zip",
        ],
        "deepseg_lesion_models": [
            "https://github.com/sct-data/deepseg_lesion_models/releases/download/r20180613/20180613_deepseg_lesion_models.zip",
            "https://osf.io/eg7v9/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20180613_deepseg_lesion_models.zip",
        ],
        "c2c3_disc_models": [
            "https://github.com/sct-data/c2c3_disc_models/releases/download/r20190117/20190117_c2c3_disc_models.zip",
            "https://osf.io/t97ap/?action=download",
            "https://www.neuro.polymtl.ca/_media/downloads/sct/20190117_c2c3_disc_models.zip",
        ],
    }

    parser = get_parser(dict_url)
    if args:
        arguments = parser.parse_args(args)
    else:
        arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    data_name = arguments.d
    verbose = int(arguments.v)
    init_sct(log_level=verbose, update=True)  # Update log level
    if arguments.o is not None:
        dest_folder = arguments.o
    else:
        dest_folder = os.path.join(os.path.abspath(os.curdir), data_name)

    url = dict_url[data_name]
    install_data(url, dest_folder, keep=arguments.k)

    printv('Done!\n', verbose)
    return 0


if __name__ == "__main__":
    init_sct()
    res = main()
    raise SystemExit(res)
