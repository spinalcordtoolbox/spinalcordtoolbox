#!/usr/bin/env python
#
# Concatenate bval files in time
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
from typing import Sequence

from spinalcordtoolbox.utils.fs import extract_fname
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser


def get_parser():
    parser = SCTArgumentParser(
        description='Concatenate bval files in time.'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        "-i",
        nargs='+',
        help='List of the bval files to concatenate. Example: `dmri_b700.bval dmri_b2000.bval`',
        metavar=Metavar.file,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-o",
        help='Output file with bvals merged. Example: `dmri_b700_b2000_concat.bval`',
        metavar=Metavar.file)

    # Arguments which implement shared functionality
    parser.add_common_args()
    parser.add_profiling_args()

    return parser


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

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
