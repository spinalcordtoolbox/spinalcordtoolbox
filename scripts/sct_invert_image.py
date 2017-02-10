#!/usr/bin/env python
#########################################################################################
#
# Register anatomical image to the template using the spinal cord centerline/segmentation.
# we assume here that we have a RPI orientation, where Z axis is inferior-superior direction
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2014-06-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: currently it seems like cross_radius is given in pixel instead of mm

import sys
import msct_parser
import msct_image

# DEFAULT PARAMETERS
class Param(object):
    ## The constructor
    def __init__(self):
        self.debug = 0

#=======================================================================================================================
# Start program
#=======================================================================================================================
def main(args=None):

    if args is None:
        args = sys.argv[1:]
    else:
        script_name =os.path.splitext(os.path.basename(__file__))[0]
        sct.printv('{0} {1}'.format(script_name, " ".join(args)))
    # initialize parameters
    param = Param()
    # call main function

    # Initialize the parser
    parser = msct_parser.Parser(__file__)
    parser.usage.set_description('Utility function for labels.')
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
    arguments = parser.parse(args)

    input_filename = arguments["-i"]
    image_input = msct_image.Image(input_filename)
    image_output = image_input.invert()
    if "-o" in arguments:
        image_output.setFileName(arguments["-o"])
    image_output.save(data_type='minimize')


if __name__ == "__main__":
    main()
