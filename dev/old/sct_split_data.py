#!/usr/bin/env python
#########################################################################################
#
# Split data. Replace "fslsplit"
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add check output in split_data

import sys
from numpy import array_split, shape, newaxis
from msct_parser import Parser
from msct_image import Image
from sct_utils import extract_fname


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
    parser.usage.set_description('Split data. By default, output files will have suffix "_0000", "_0002", etc.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Input file.",
                      mandatory=True,
                      example="data.nii.gz")
    parser.add_option(name="-dim",
                      type_value="multiple_choice",
                      description="""Dimension for split.""",
                      mandatory=False,
                      default_value='t',
                      example=['x', 'y', 'z', 't'])
    parser.add_option(name="-suffix",
                      type_value="str",
                      description="""Output suffix.""",
                      mandatory=False,
                      default_value='_',
                      example='_')
    return parser


# concatenation
# ==========================================================================================
def split_data(fname_in, dim, suffix):
    """
    Split data
    :param fname_in: input file.
    :param dim: dimension: 0, 1, 2, 3.
    :return: True/False
    """
    # Parse file name
    path_in, file_in, ext_in = extract_fname(fname_in)
    # Open first file.
    im = Image(fname_in)
    data = im.data
    if dim+1 > len(shape(data)):  # in case input volume is 3d and dim=t
        data = data[..., newaxis]
    # Split data into list
    data_split = array_split(data, data.shape[dim], dim)
    # Write each file
    for i in range(len(data_split)):
        # Build suffix
        suffix_output = suffix+str(i).zfill(4)
        # Write file
        im_split = im
        im_split.data = data_split[i]
        im_split.setFileName(path_in+file_in+suffix_output+ext_in)
        im_split.save()
    return True


# MAIN
# ==========================================================================================
def main(args = None):

    dim_list = ['x', 'y', 'z', 't']

    if not args:
        args = sys.argv[1:]

    # Building the command, do sanity checks
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments["-i"]
    dim_concat = arguments["-dim"]
    suffix = arguments["-suffix"]

    # convert dim into numerical values:
    dim = dim_list.index(dim_concat)

    # convert file
    split_data(fname_in, dim, suffix)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
