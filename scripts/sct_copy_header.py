#!/usr/bin/env python
#########################################################################################
#
# Copy header from nifti data. Replace "fslcpgeom"
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
from numpy import concatenate, array_split
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
    parser.usage.set_description('Copy NIFTI header from source to destination.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Source file.",
                      mandatory=True,
                      example="src.nii.gz")
    parser.add_option(name="-d",
                      type_value="file",
                      description="Destination file.",
                      mandatory=True,
                      example='dest.nii.gz')
    return parser


# concatenation
# ==========================================================================================
def copy_header(fname_src, fname_dest):
    """
    Copy header
    :param fname_src: source file name
    :param fname_dest: destination file name
    :return:
    """
    nii_src = Image(fname_src)
    data_dest = Image(fname_dest).data
    nii_src.setFileName(fname_dest)
    nii_src.data = data_dest
    nii_src.save()


# MAIN
# ==========================================================================================
def main(args = None):

    if not args:
        args = sys.argv[1:]

    # Building the command, do sanity checks
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_src = arguments["-i"]
    fname_dest = arguments["-d"]

    # copy header
    copy_header(fname_src, fname_dest)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
