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
from spinalcordtoolbox.utils.sys import (init_sct, printv, set_loglevel,
                                         __sct_dir__, __bin_dir__)


# Dictionary containing list of URLs and locations for datasets.
# Mirror servers are listed in order of decreasing priority.
# If exists, favour release artifact straight from github
# For the location field, this is where the dataset will be
# downloaded to (relative to the repo) if a location isn't passed by
# the user.
DATASET_DICT = {
    "sct_example_data": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/sct_example_data/releases/download/r20180525/20180525_sct_example_data.zip",
            "https://osf.io/kjcgs/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "sct_example_data"),
    },
    "sct_testing_data": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/sct_testing_data/releases/download/r20210330230310/sct_testing_data-r20210330230310.zip",
            "https://osf.io/download/60629509229503022e6f107d/",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "sct_testing_data"),
    },
    "PAM50": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/PAM50/releases/download/r20201104/PAM50-r20201104.zip",
            "https://osf.io/download/5fa21326a5bb9d00610a5a21/",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "PAM50"),
    },
    "MNI-Poly-AMU": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/MNI-Poly-AMU/releases/download/r20170310/20170310_MNI-Poly-AMU.zip",
            "https://osf.io/sh6h4/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "MNI-Poly-AMU"),
    },
    "gm_model": {
        "mirrors": [
            "https://osf.io/ugscu/?action=download"
        ],
        "default_location": os.path.join(__sct_dir__, "data", "gm_model"),
    },
    "optic_models": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/optic_models/releases/download/r20170413/20170413_optic_models.zip",
            "https://osf.io/g4fwn/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "optic_models"),
    },
    "pmj_models": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/pmj_models/releases/download/r20170922/20170922_pmj_models.zip",
            "https://osf.io/4gufr/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "pmj_models"),
    },
    "binaries_linux": {
        "mirrors": [
            "https://osf.io/cs6zt/?action=download",
        ],
        "default_location": __bin_dir__,
    },
    "binaries_osx": {
        "mirrors": [
            "https://osf.io/874cy?action=download",
        ],
        "default_location": __bin_dir__,
    },
    "binaries_win": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/spinalcordtoolbox-binaries/releases/download/test-release/binaries_win.zip",
        ],
        "default_location": __bin_dir__,
    },
    "course_hawaii17": {
        "mirrors": [
            "https://osf.io/6exht/?action=download",
            "https://github.com/spinalcordtoolbox/sct_tutorial_data/releases/download/SCT-Course/hawaii17.zip",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "course_hawaii17"),
    },
    "course_paris18": {
        "mirrors": [
            "https://osf.io/9bmn5/?action=download",
            "https://github.com/spinalcordtoolbox/sct_tutorial_data/releases/download/SCT-Course/paris18.zip",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "course_paris18"),
    },
    "course_london19": {
        "mirrors": [
            "https://osf.io/4q3u7/?action=download",
            "https://github.com/spinalcordtoolbox/sct_tutorial_data/releases/download/SCT-Course/london19.zip",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "course_london19"),
    },
    "course_beijing19": {
        "mirrors": [
            "https://osf.io/ef4xz/?action=download",
            "https://github.com/spinalcordtoolbox/sct_tutorial_data/releases/download/SCT-Course/beijing19.zip",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "course_beijing19"),
    },
    "course_london20": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/sct_tutorial_data/releases/download/SCT-Course/london20.zip",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "course_london20"),
    },
    "course_harvard21": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/sct_tutorial_data/archive/refs/tags/SCT-Course-20211116.zip",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "course_harvard21"),
    },
    "deepseg_gm_models": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/deepseg_gm_models/releases/download/r20180205/20180205_deepseg_gm_models.zip",
            "https://osf.io/b9y4x/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "deepseg_gm_models"),
    },
    "deepseg_sc_models": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/deepseg_sc_models/releases/download/r20180610/20180610_deepseg_sc_models.zip",
            "https://osf.io/avf97/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "deepseg_sc_models"),
    },
    "deepseg_lesion_models": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/deepseg_lesion_models/releases/download/r20180613/20180613_deepseg_lesion_models.zip",
            "https://osf.io/eg7v9/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "deepseg_lesion_models"),
    },
    "c2c3_disc_models": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/c2c3_disc_models/releases/download/r20190117/20190117_c2c3_disc_models.zip",
            "https://osf.io/t97ap/?action=download",
        ],
        "default_location": os.path.join(__sct_dir__, "data", "c2c3_disc_models"),
    },
    "exvivo_template": {
        "mirrors": [
            "https://github.com/spinalcordtoolbox/exvivo-template/archive/refs/tags/r20210317.zip"
        ],
        "default_location": os.path.join(__sct_dir__, "data", "exvivo_template"),
    }
}


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


def main(argv=None):
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

