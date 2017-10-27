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

import sys
from msct_parser import Parser
from sct_utils import extract_fname, printv
import sct_utils as sct


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)

    # # initialize parameters
    # param = Param()
    # param_default = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Transpose bvecs file (if necessary) to get nx3 structure.')
    parser.add_option(name='-bvec',
                      type_value='file',
                      description='Input bvecs file.',
                      mandatory=True,
                      example='bvecs.txt')
    parser.add_option(name='-i',
                      type_value='file',
                      description='Input bvecs file.',
                      mandatory=False,
                      example='bvecs.txt',
                      deprecated_by='-bvec')
    parser.add_option(name='-o',
                      type_value='file_output',
                      description='Output bvecs file. By default input file is overwritten.',
                      mandatory=False,
                      example='bvecs_t.txt')
    parser.add_option(name='-v',
                      type_value='multiple_choice',
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1', '2'])
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments['-bvec']
    if '-o' in arguments:
        fname_out = arguments['-o']
    else:
        fname_out = ''
    verbose = int(arguments['-v'])

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
    sct.start_stream_logger()
    # call main function
    main()
