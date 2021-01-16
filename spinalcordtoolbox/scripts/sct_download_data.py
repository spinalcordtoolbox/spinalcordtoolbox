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

from spinalcordtoolbox.download import install_data
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder
from spinalcordtoolbox.utils.sys import init_sct, printv, set_global_loglevel


# Dictionary containing list of URLs for data names.
# Mirror servers are listed in order of decreasing priority.
# If exists, favour release artifact straight from github
DICT_URL = {
    "sct_example_data": [
        "https://github.com/sct-data/sct_example_data/releases/download/r20180525/20180525_sct_example_data.zip",
        "https://osf.io/kjcgs/?action=download",
        "https://www.neuro.polymtl.ca/_media/downloads/sct/20180525_sct_example_data.zip",
    ],
    "sct_testing_data": [
        "https://github.com/sct-data/sct_testing_data/releases/download/r20210114/sct_testing_data-r20210114.zip",
        "https://osf.io/download/6000caf286541a05da149de5/"
    ],
    "PAM50": [
        "https://github.com/sct-data/PAM50/releases/download/r20201104/PAM50-r20201104.zip",
        "https://osf.io/download/5fa21326a5bb9d00610a5a21/"
    ],
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


def get_parser():
    parser = SCTArgumentParser(
        description="Download binaries from the web."
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-d',
        required=True,
        choices=list(DICT_URL.keys()),
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
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
    )

    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    data_name = arguments.d
    if arguments.o is not None:
        dest_folder = arguments.o
    else:
        dest_folder = os.path.join(os.path.abspath(os.curdir), data_name)

    url = DICT_URL[data_name]
    install_data(url, dest_folder, keep=arguments.k)

    printv('Done!\n', verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

