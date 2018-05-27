#!/usr/bin/env python
#########################################################################################
#
# Calculate b-value.
#
# N.B. SI unit for gyromagnetic ratio is radian per second per tesla, therefore need to multiply by 2pi.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-07-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
import os
import getopt
import math
from msct_parser import Parser
import sct_utils as sct

# main
#=======================================================================================================================
def main():

    # Initialization
    GYRO = float(42.576 * 10 ** 6)  # gyromagnetic ratio (in Hz.T^-1)
    gradamp = []
    bigdelta = []
    smalldelta = []

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    gradamp = arguments['-g']
    bigdelta = arguments['-b']
    smalldelta = arguments['-d']

    # sct.printv(arguments)
    sct.printv('\nCheck parameters:')
    sct.printv('  gradient amplitude ..... ' + str(gradamp) + ' mT/m')
    sct.printv('  big delta .............. ' + str(bigdelta) + ' ms')
    sct.printv('  small delta ............ ' + str(smalldelta) + ' ms')
    sct.printv('  gyromagnetic ratio ..... ' + str(GYRO) + ' Hz/T')
    sct.printv('')

    bvalue = (2 * math.pi * GYRO * gradamp * 0.001 * smalldelta * 0.001) ** 2 * (bigdelta * 0.001 - smalldelta * 0.001 / 3)

    sct.printv('b-value = ' + str(bvalue / 10**6) + ' mm^2/s\n')
    return bvalue


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Calculate b-value (in mm^2/s).')
    parser.add_option(name="-g",
                      type_value="float",
                      description="Amplitude of diffusion gradients (in mT/m)",
                      mandatory=True,
                      example='40')
    parser.add_option(name="-b",
                      type_value="float",
                      description="Big delta: time between both diffusion gradients (in ms)",
                      mandatory=True,
                      example='40')
    parser.add_option(name="-d",
                      type_value="float",
                      description="Small delta: duration of each diffusion gradient (in ms)",
                      mandatory=True,
                      example='30')

    return parser


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
