#!/usr/bin/env python
#########################################################################################
#
# Invert the intensity of the image
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2017-10-17
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
from msct_parser import Parser
from msct_image import Image
import sct_utils as sct


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('This function inverts the image intensity using the maximum intensity in the image and 0.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image to invert.",
                      mandatory=True,
                      example="my_image.nii.gz")

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="output image.",
                      mandatory=False,
                      example="output_image.nii.gz",
                      default_value="inverted_image.nii.gz")

    return parser


def main(args=None):
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    input_filename = arguments["-i"]
    image_input = Image(input_filename)
    image_output = image_input.invert()
    if '-o' in arguments:
        image_output.setFileName(arguments['-o'])
    image_output.save(type='minimize')


if __name__ == "__main__":
    sct.init_sct()
    main()