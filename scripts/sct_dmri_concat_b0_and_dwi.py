#!/usr/bin/env python
#
# Merge b=0 and dMRI data and output appropriate bvecs/bvals files.
#
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT


from __future__ import absolute_import

import os
import sys
import argparse

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar, SmartFormatter


def get_parser():
    parser = argparse.ArgumentParser(
        description="Concatenate b=0 scans with DWI time series and update the bvecs and bvals files. Note that you can"
                    "concatenate more than two files (e.g.: b0 dwi1 dw2 dw3).",
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py"))
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        help="Input 4d files, separated by space, listed in the right order of concatenation. Example: b0.nii dmri.nii",
        nargs='+',
        metavar=Metavar.file,
        required=True)
    mandatory.add_argument(
        '-bval',
        help="Bvals file. Example: bvals.txt",
        metavar=Metavar.file,
        required=True)
    mandatory.add_argument(
        '-bvec',
        help="Bvecs file. Example: bvecs.txt",
        metavar=Metavar.file,
        required=True)
    mandatory.add_argument(
        '-o',
        help="Output 4d concatenated file. Example: b0_dmri_concat.nii",
        metavar=Metavar.file)
    mandatory.add_argument(
        '-obval',
        help="Output concatenated bval file. Example: bval_concat.txt",
        metavar=Metavar.file)
    mandatory.add_argument(
        '-obvec',
        help="Output concatenated bvec file. Example: bvec_concat.txt",
        metavar=Metavar.file)
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        help="Show this help message and exit")

    return parser


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

    # Open files and concatenate
    # Save files


if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
