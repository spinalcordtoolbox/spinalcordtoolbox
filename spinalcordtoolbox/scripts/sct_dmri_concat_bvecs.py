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

import sys
from typing import Sequence

from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, extract_fname, set_loglevel


def get_parser():
    parser = SCTArgumentParser(
        description='Concatenate bvec files in time. You can either use bvecs in lines or columns. '
                    'N.B.: Return bvecs in lines. If you need it in columns, please use '
                    'sct_dmri_transpose_bvecs afterwards.'
    )

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
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    fname_bvecs_list = arguments.i
    # Build fname_out
    if arguments.o is not None:
        fname_out = arguments.o
    else:
        path_in, file_in, ext_in = extract_fname(fname_bvecs_list[0])
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


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
