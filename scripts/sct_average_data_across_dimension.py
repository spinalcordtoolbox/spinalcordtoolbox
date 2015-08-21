#!/usr/bin/env python
#########################################################################################
#
# Average across dimension. Replaces "fslmaths -Tmean"
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add check output in average_across_dimension

import sys
from numpy import mean
from msct_parser import Parser
from msct_image import Image
from sct_utils import extract_fname, printv


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.verbose = 1


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)

    # initialize parameters
    param = Param()
    param_default = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Average data across specified dimension.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Input file.",
                      mandatory=True,
                      example="data.nii.gz")
    parser.add_option(name="-dim",
                      type_value="multiple_choice",
                      description="""Dimension for average.""",
                      mandatory=False,
                      default_value='t',
                      example=['x', 'y', 'z', 't'])
    parser.add_option(name="-o",
                      type_value="file_output",
                      description='Output file. If output specified, adding suffix "_mean"',
                      mandatory=False,
                      example=['data_mean.nii.gz'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value=param.verbose,
                      example=['0', '1', '2'])
    return parser


# concatenation
# ==========================================================================================
def average_data_across_dimension(fname_in, fname_out, dim):
    """
    Average data
    :param fname_in: input file.
    :param fname_out: output file
    :param dim: dimension: 0, 1, 2, 3
    :return: True/False
    """
    # Open file.
    im = Image(fname_in)
    data = im.data
    # Average
    data_mean = mean(data, dim)
    # Write output
    im.data = data_mean
    im.setFileName(fname_out)
    im.save()
    return True


# MAIN
# ==========================================================================================
def main(args = None):

    fname_out = ''
    dim_list = ['x', 'y', 'z', 't']

    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments["-i"]
    if '-o' in arguments:
        fname_out = arguments["-o"]
    dim_concat = arguments["-dim"]
    verbose = int(arguments['-v'])

    # convert dim into numerical values:
    dim = dim_list.index(dim_concat)

    # Build fname_out
    if fname_out == '':
        path_in, file_in, ext_in = extract_fname(fname_in)
        fname_out = path_in+file_in+'_mean'+ext_in

    # average data
    if not average_data_across_dimension(fname_in, fname_out, dim):
        printv('ERROR in average_data_across_dimension', 1, 'error')

    # display message
    printv('Created file:\n--> '+fname_out+'\n', verbose, 'info')


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
