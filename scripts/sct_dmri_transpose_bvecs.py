#!/usr/bin/env python
#=======================================================================================================================
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

from __future__ import absolute_import, division

import os
import sys
import argparse
from spinalcordtoolbox.utils import Metavar, SmartFormatter

from sct_utils import extract_fname, printv
import sct_utils as sct


# PARSER
# ==========================================================================================
def get_parser():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description='Transpose bvecs file (if necessary) to get nx3 structure.',
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
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
        metavar=Metavar.str,
        choices=['0', '1', '2'],
        default='1',
        help="Verbose: 0 = nothing, 1 = basic, 2 = extended."
    )

    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    parser = get_parser()
    if args:
        arguments = parser.parse_args(args)
    else:
        arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    fname_in = arguments.bvec
    fname_out = arguments.o
    verbose = int(arguments.v)
    sct.init_sct(log_level=verbose, update=True)  # Update log level

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


# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
