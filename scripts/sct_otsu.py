#!/usr/bin/env python
#########################################################################################
#
# Threshold image using Otsu algorithm.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
from msct_parser import Parser
from msct_image import Image
from sct_utils import extract_fname, printv
from skimage.filters import threshold_otsu


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
    parser.usage.set_description('Threshold image using Otsu algorithm.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Input file.",
                      mandatory=True,
                      example="data.nii.gz")
    parser.add_option(name="-o",
                      type_value="file_output",
                      description='Output file. If output specified, adding suffix "_mean"',
                      mandatory=False,
                      example=['data_mean.nii.gz'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1', '2'])
    return parser


# concatenation
# ==========================================================================================
def otsu(fname_in, fname_out):
    """
    Threshold image using Otsu algorithm.
    :param fname_in: input file.
    :param fname_out: output file
    :return: True/False
    """
    # Open file.
    nii = Image(fname_in)
    data = nii.data
    # Threshold
    thresh = threshold_otsu(data)
    data_binary = data > thresh
    # Write output
    nii.data = data_binary
    nii.setFileName(fname_out)
    nii.save()
    return True


# MAIN
# ==========================================================================================
def main(args = None):

    fname_out = ''

    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments["-i"]
    if '-o' in arguments:
        fname_out = arguments["-o"]
    verbose = int(arguments['-v'])

    # Build fname_out
    if fname_out == '':
        path_in, file_in, ext_in = extract_fname(fname_in)
        fname_out = path_in+file_in+'_mean'+ext_in

    # average data
    if not otsu(fname_in, fname_out):
        printv('ERROR in otsu', 1, 'error')

    # display message
    printv('Created file:\n--> '+fname_out+'\n', verbose, 'info')


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # # initialize parameters
    # param = Param()
    # call main function
    main()
