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

from __future__ import absolute_import

import os
import sys

from spinalcordtoolbox.download import install_data

from msct_parser import Parser
import sct_utils as sct


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('''Download binaries from the web.''')
    parser.add_option(
        name="-d",
        type_value="multiple_choice",
        description="Name of the dataset.",
        mandatory=True,
        # TODO: replace with key dict items
        example=[
            'sct_example_data',
            'sct_testing_data',
            'course_hawaii17',
            'course_paris18',
            'course_london19',
            'course_beijing19',
            'PAM50',
            'MNI-Poly-AMU',
            'gm_model',
            'optic_models',
            'pmj_models',
            'binaries_debian',
            'binaries_centos',
            'binaries_osx',
            'deepseg_gm_models',
            'deepseg_sc_models',
            'deepseg_lesion_models',
            'c2c3_disc_models'
        ])
    parser.add_option(
        name="-v",
        type_value="multiple_choice",
        description="Verbose. 0: nothing. 1: basic. 2: extended.",
        mandatory=False,
        default_value=1,
        example=['0', '1', '2'])
    parser.add_option(
        name="-o",
        type_value="folder_creation",
        description="path to save the downloaded data",
        mandatory=False)
    parser.add_option(
        name="-h",
        type_value=None,
        description="Display this help",
        mandatory=False)
    return parser


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    data_name = arguments['-d']
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level
    dest_folder = arguments.get('-o', os.path.abspath(os.curdir))

    install_data(data_name, dest_folder)

    sct.printv('Done!\n', verbose)
    return 0


if __name__ == "__main__":
    sct.init_sct()
    res = main()
    raise SystemExit(res)
