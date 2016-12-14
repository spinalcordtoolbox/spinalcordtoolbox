#!/usr/bin/env python
#########################################################################################
#
# Concatenate data. Replace "fsl_merge"
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
from numpy import concatenate, expand_dims
import sct_utils as sct
from msct_parser import Parser
from msct_image import Image


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
    parser.usage.set_description('Concatenate data.')
    parser.add_option(name="-i",
                      type_value=[[','], "file"],
                      description='Multiple files separated with ",".',
                      mandatory=True,
                      example="data1.nii.gz,data2.nii.gz")
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Output file",
                      mandatory=True,
                      example=['data_concat.nii.gz'])
    parser.add_option(name="-dim",
                      type_value="multiple_choice",
                      description="""Dimension for concatenation.""",
                      mandatory=True,
                      example=['x', 'y', 'z', 't'])
    return parser


# concatenation
# ==========================================================================================
def concat_data(fname_in, fname_out, dim):
    """
    Concatenate data
    :param fname_in: list of file names.
    :param fname_out:
    :param dim: dimension: 0, 1, 2, 3.
    :return: none
    """
    # create empty list
    list_data = []

    # loop across files
    for i in range(len(fname_in)):
        # append data to list
        list_data.append(Image(fname_in[i]).data)

    # expand dimension of all elements in the list if necessary
    if dim > list_data[0].ndim-1:
        list_data = [expand_dims(i, dim) for i in list_data]
    # concatenate
    try:
        data_concat = concatenate(list_data, axis=dim)
    except Exception as e:
        sct.printv('\nERROR: Concatenation on line {}'.format(sys.exc_info()[-1].tb_lineno)+'\n'+str(e)+'\n', 1, 'error')

    # write file
    im = Image(fname_in[0])
    im.data = data_concat
    im.setFileName(fname_out)
    im.save()


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
    fname_out = arguments["-o"]
    dim_concat = arguments["-dim"]

    # convert dim into numerical values:
    dim = dim_list.index(dim_concat)

    # convert file
    concat_data(fname_in, fname_out, dim)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
