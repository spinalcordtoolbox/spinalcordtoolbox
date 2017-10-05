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
from msct_parser import Parser
from sct_utils import extract_fname
import sct_utils as sct
from dipy.data.fetcher import read_bvals_bvecs

# PARSER
# ==========================================================================================


def get_parser():

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Concatenate bvec files in time. You can either use bvecs in lines or columns.\nN.B.: Return bvecs in lines. If you need it in columns, please use sct_dmri_transpose_bvecs afterwards.')
    parser.add_option(name="-i",
                      type_value=[[','], 'file'],
                      description="List of the bvec files to concatenate.",
                      mandatory=True,
                      example="dmri_b700.bvec,dmri_b2000.bvec")
    parser.add_option(name="-o",
                      type_value="file_output",
                      description='Output file with bvecs concatenated.',
                      mandatory=False,
                      example='dmri_b700_b2000_concat.bvec')
    return parser


# MAIN
# ==========================================================================================
def main():

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_bvecs_list = arguments["-i"]
    # Build fname_out
    if "-o" in arguments:
        fname_out = arguments["-o"]
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
        bval_i, bvec_i = read_bvals_bvecs(None, i_fname)
        for i in range(0, 3):
            bvecs_all[i] += ' '.join(str(v) for v in map(lambda n: '%.16f'%n, bvec_i[:, i]))
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
    sct.start_stream_logger()
    # call main function
    main()
