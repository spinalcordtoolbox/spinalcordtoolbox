#!/usr/bin/env python
#########################################################################################
#
# Concatenate bvecs files in time.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Simon LEVY
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

import sys, os, argparse

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar, SmartFormatter


# PARSER
# ==========================================================================================


def get_parser():
    # Initialize the parser

    parser = argparse.ArgumentParser(
        description='Concatenate bvec files in time. You can either use bvecs in lines or columns. '
                    'N.B.: Return bvecs in lines. If you need it in columns, please use '
                    'sct_dmri_transpose_bvecs afterwards.',
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py"))

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        nargs='+',
        required=True,
        help='List of the bvec files to concatenate. Example: dmri_b700.bvec dmri_b2000.bvec',
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
        metavar=Metavar.file,
        help='Output file with bvecs concatenated. Example: dmri_b700_b2000_concat.bvec')

    return parser


# MAIN
# ==========================================================================================
def main():
    # Get parser info
    parser = get_parser()
    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    fname_bvecs_list = arguments.i
    # Build fname_out
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path_in, file_in, ext_in = sct.extract_fname(fname_bvecs_list[0])
        fname_out = path_in + 'bvecs_concat' + ext_in

    # # Open bvec files and collect values
    # nb_files = len(fname_bvecs_list)
    # bvecs_all = []
    # for i_fname in fname_bvecs_list:
    #     bvecs = []
    #     with open(i_fname) as f:
    #         for line in f:
    #             bvec_line = map(float, line.split())
    #             bvecs.append(bvec_line)
    #     bvecs_all.append(bvecs)
    #     f.close()
    # # Concatenate
    # bvecs_concat = ''
    # for i in range(0, 3):
    #     for j in range(0, nb_files):
    #         bvecs_concat += ' '.join(str(v) for v in bvecs_all[j][i])
    #         bvecs_concat += ' '
    #     bvecs_concat += '\n'
    #

    # Open bvec files and collect values
    bvecs_all = ['', '', '']
    for i_fname in fname_bvecs_list:
        from dipy.data.fetcher import read_bvals_bvecs
        bval_i, bvec_i = read_bvals_bvecs(None, i_fname)
        for i in range(0, 3):
            bvecs_all[i] += ' '.join(str(v) for v in map(lambda n: '%.16f' % n, bvec_i[:, i]))
            bvecs_all[i] += ' '

    # Concatenate
    bvecs_concat = '\n'.join(str(v) for v in bvecs_all)

    # Write new bvec
    new_f = open(fname_out, 'w')
    new_f.write(bvecs_concat)
    new_f.close()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
