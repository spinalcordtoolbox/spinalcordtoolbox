#!/usr/bin/env python
#########################################################################################
#
# Concatenate bval files in time.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Simon LEVY
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys
import argparse

from spinalcordtoolbox.utils import Metavar, SmartFormatter, init_sct, extract_fname, set_global_loglevel


def get_parser():
    # Initialize the parser

    parser = argparse.ArgumentParser(
        description='Concatenate bval files in time.',
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py"))
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        nargs='+',
        required=True,
        help='List of the bval files to concatenate. Example: dmri_b700.bval dmri_b2000.bval',
        metavar=Metavar.file,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-o",
        help='Output file with bvals merged. Example: dmri_b700_b2000_concat.bval',
        metavar=Metavar.file)
    optional.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = nothing, 1 = classic, 2 = expended",
        required=False,
        choices=(0, 1, 2),
        default=1)

    return parser


# MAIN
# ==========================================================================================
def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv if argv else ['--help'])
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    fname_bval_list = arguments.i
    # Build fname_out
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path_in, file_in, ext_in = extract_fname(fname_bval_list[0])
        fname_out = path_in + 'bvals_concat' + ext_in

    # Open bval files and concatenate
    bvals_concat = ''
    # for file_i in fname_bval_list:
    #     f = open(file_i, 'r')
    #     for line in f:
    #         bvals_concat += line
    #     f.close()
    from dipy.data.fetcher import read_bvals_bvecs
    for i_fname in fname_bval_list:
        bval_i, bvec_i = read_bvals_bvecs(i_fname, None)
        bvals_concat += ' '.join(str(v) for v in bval_i)
        bvals_concat += ' '

    # Write new bval
    new_f = open(fname_out, 'w')
    new_f.write(bvals_concat)
    new_f.close()


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
