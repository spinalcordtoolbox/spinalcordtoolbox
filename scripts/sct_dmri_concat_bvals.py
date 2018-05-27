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

import sys
from msct_parser import Parser
from sct_utils import extract_fname
import sct_utils as sct
from dipy.data.fetcher import read_bvals_bvecs

# PARSER
# ==========================================================================================


def get_parser():

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Concatenate bval files in time.')
    parser.add_option(name="-i",
                      type_value=[[','], 'file'],
                      description="List of the bval files to concatenate.",
                      mandatory=True,
                      example="dmri_b700.bval,dmri_b2000.bval")
    parser.add_option(name="-o",
                      type_value="file_output",
                      description='Output file with bvals merged.',
                      mandatory=False,
                      example='dmri_b700_b2000_concat.bval')
    return parser


# MAIN
# ==========================================================================================
def main():

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_bval_list = arguments["-i"]
    # Build fname_out
    if "-o" in arguments:
        fname_out = arguments["-o"]
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
    for i_fname in fname_bval_list:
        bval_i, bvec_i = read_bvals_bvecs(i_fname, None)
        bvals_concat += ' '.join(str(v) for v in bval_i)
        bvals_concat += ' '

    # Write new bval
    new_f = open(fname_out, 'w')
    new_f.write(bvals_concat)
    new_f.close()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
