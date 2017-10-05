#!/usr/bin/env python
#########################################################################################
#
# Compute maximum spinal cord compression.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import sct_utils as sct
from msct_parser import Parser


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)
    parser.usage.set_description('Compute Maximum Spinal Cord Compression (MSCC) as in: Miyanji F, Furlan JC, Aarabi B, Arnold PM, Fehlings MG. Acute cervical traumatic spinal cord injury: MR imaging findings correlated with neurologic outcome--prospective study with 100 consecutive patients. Radiology 2007;243(3):820-827.')
    parser.add_option(name='-di',
                      type_value='float',
                      description='Anteroposterior cord distance at the level of maximum injury',
                      mandatory=True,
                      example=6.85)
    parser.add_option(name='-da',
                      type_value='float',
                      description='Anteroposterior cord distance at the nearest normal level above the level of injury',
                      mandatory=True,
                      example=7.65)
    parser.add_option(name='-db',
                      type_value='float',
                      description='Anteroposterior cord distance at the nearest normal level below the level of injury',
                      mandatory=True,
                      example=7.02)
    parser.add_option(name="-h",
                      type_value=None,
                      description="Display this help",
                      mandatory=False)
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    # initialization
    verbose = 1

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    di = arguments['-di']
    da = arguments['-da']
    db = arguments['-db']

    # Compute MSCC
    MSCC = (1 - float(di) / ((da + db) / float(2))) * 100

    # Display results
    sct.printv('\nMSCC = ' + str(MSCC) + '\n', verbose, 'info')


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    # call main function
    main()
