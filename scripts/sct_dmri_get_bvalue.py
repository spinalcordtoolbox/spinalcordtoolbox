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


import math
import sys

import msct_parser


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    else:
        script_name =os.path.splitext(os.path.basename(__file__))[0]
        sct.printv('{0} {1}'.format(script_name, " ".join(args)))

    # Initialization
    GYRO = float(42.576 * 10 ** 6)  # gyromagnetic ratio (in Hz.T^-1)

    parser = get_parser()
    arguments = parser.parse(args)
    gradamp = arguments['-g']
    bigdelta = arguments['-b']
    smalldelta = arguments['-d']

    # print arguments
    print '\nCheck parameters:'
    print '  gradient amplitude ..... ' + str(gradamp * 1000) + ' mT/m'
    print '  big delta .............. ' + str(bigdelta * 1000) + ' ms'
    print '  small delta ............ ' + str(smalldelta * 1000) + ' ms'
    print '  gyromagnetic ratio ..... ' + str(GYRO) + ' Hz/T'
    print ''

    bvalue = (2 * math.pi * GYRO * gradamp * smalldelta) ** 2 * (bigdelta - smalldelta / 3)

    print 'b-value = ' + str(bvalue / 10 ** 6) + ' mm^2/s\n'
    return bvalue


def get_parser():
    # Initialize the parser
    parser = msct_parser.Parser(__file__)
    parser.usage.set_description('Calculate b-value (in mm^2/s).')
    parser.add_option(name="-g",
                      type_value="float",
                      description="Amplitude of diffusion gradients (in T/m)",
                      mandatory=True,
                      example='0.04')
    parser.add_option(name="-b",
                      type_value="float",
                      description="Big delta: time between both diffusion gradients (in s)",
                      mandatory=True,
                      example='0.04')
    parser.add_option(name="-d",
                      type_value="float",
                      description="Small delta: duration of each diffusion gradient (in s)",
                      mandatory=True,
                      example='0.03')

    return parser


if __name__ == "__main__":
    main()
