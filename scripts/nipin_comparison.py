#!/usr/bin/env python

import sys, sct_utils as sct
from nicolas_scripts.evaluate_hogancest import main as evaluate_hogancest
from msct_parser import Parser

def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Blablablabla')
    parser.add_option(name="-i",
                      type_value="folder",
                      description="Input folder with data used for the test",
                      mandatory=True,
                      example="/home/data")

    parser.add_option(name="-test",  # TODO find better name
                      type_value="str",
                      description="put name of the test you want to run",
                      mandatory=True,
                      example="src_seg.nii.gz")
    parser.add_option(name="-o",
                      type_value="folder",
                      description="output folder for test results",
                      mandatory=False,
                      example="path/to/output/folder")

    return parser

def main(args=None):

    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    #arguments = parser.parse(args)
    evaluate_hogancest(args=args)

if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
