#!/usr/bin/env python
#######################################################################################################################
#
#
# Merge images
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Dominique Eden, Sara Dupont
# Modified: 2017-03-17
#
# About the license: see the file LICENSE.TXT
########################################################################################################################
import sys
from msct_parser import Parser

def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Merge images to the same space')
    parser.add_option(name="-i",
                      type_value=[[','], 'file'],
                      description="Input images",
                      mandatory=True)
    parser.add_option(name="-d",
                      type_value='file',
                      description="Destination image",
                      mandatory=True)
    parser.add_option(name="-w",
                      type_value=[[','], 'file'],
                      description="List of warping fields from input images to destination image",
                      mandatory=True)

    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False)

    parser.usage.addSection('MISC')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value=str(int(Param().rm_tmp)),
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value=str(Param().verbose))

    return parser




class Param:
    def __init__(self):
        self.rm_tmp = True
        self.verbose = 1




########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # create param objects
    param = Param()

    # get parser
    parser = get_parser()
    arguments = parser.parse(args)

    # set param arguments ad inputted by user
    list_fname_in= arguments["-i"]
    fname_dest = arguments["-d"]
    list_fname_warp = arguments["-w"]

    if '-ofolder' in arguments:
        path_results= arguments['-ofolder']
    if '-r' in arguments:
        param.rm_tmp= bool(int(arguments['-r']))
    if '-v' in arguments:
        param.verbose= arguments['-v']


if __name__ == "__main__":
    main()
