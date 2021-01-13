#!/usr/bin/env python
# =======================================================================================================================
#
# Transpose bvecs file (if necessary) to get nx3 structure
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#!/usr/bin/env python
#########################################################################################
#
# Compute DTI.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys

from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, extract_fname, printv, set_global_loglevel


def get_parser():
    parser = SCTArgumentParser(
        description='Transpose bvecs file (if necessary) to get nx3 structure.'
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-bvec',
        metavar=Metavar.file,
        required=True,
        help="Input bvecs file. Example: bvecs.txt"
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
        metavar=Metavar.file,
        default='',
        help="Output bvecs file. By default, input file is overwritten. Example: bvecs_t.txt"
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


# MAIN
# ==========================================================================================
def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    fname_in = arguments.bvec
    fname_out = arguments.o

    # get bvecs in proper orientation
    from dipy.io import read_bvals_bvecs
    bvals, bvecs = read_bvals_bvecs(None, fname_in)

    # # Transpose bvecs
    # printv('Transpose bvecs...', verbose)
    # # from numpy import transpose
    # bvecs = bvecs.transpose()

    # Write new file
    if fname_out == '':
        path_in, file_in, ext_in = extract_fname(fname_in)
        fname_out = path_in + file_in + ext_in
    fid = open(fname_out, 'w')
    for iLine in range(bvecs.shape[0]):
        fid.write(' '.join(str(i) for i in bvecs[iLine, :]) + '\n')
    fid.close()

    # display message
    printv('Created file:\n--> ' + fname_out + '\n', verbose, 'info')


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

