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
import math
import argparse

from spinalcordtoolbox.utils import Metavar, SmartFormatter, init_sct, printv, set_global_loglevel


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv if argv else ['--help'])
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    GYRO = float(42.576 * 10 ** 6)  # gyromagnetic ratio (in Hz.T^-1)
    gradamp = []
    bigdelta = []
    smalldelta = []
    gradamp = arguments.g
    bigdelta = arguments.b
    smalldelta = arguments.d

    # printv(arguments)
    printv('\nCheck parameters:')
    printv('  gradient amplitude ..... ' + str(gradamp) + ' mT/m')
    printv('  big delta .............. ' + str(bigdelta) + ' ms')
    printv('  small delta ............ ' + str(smalldelta) + ' ms')
    printv('  gyromagnetic ratio ..... ' + str(GYRO) + ' Hz/T')
    printv('')

    bvalue = (2 * math.pi * GYRO * gradamp * 0.001 * smalldelta * 0.001) ** 2 * (
        bigdelta * 0.001 - smalldelta * 0.001 / 3)

    printv('b-value = ' + str(bvalue / 10 ** 6) + ' mm^2/s\n')
    return bvalue


def get_parser():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description='Calculate b-value (in mm^2/s).',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-g",
        type=float,
        required=True,
        help="Amplitude of diffusion gradients (in mT/m). Example: 40",
        metavar=Metavar.float,
    )
    mandatory.add_argument(
        "-b",
        type=float,
        required=True,
        help="Big delta: time between both diffusion gradients (in ms). Example: 40",
        metavar=Metavar.float,
    )
    mandatory.add_argument(
        "-d",
        type=float,
        required=True,
        help="Small delta: duration of each diffusion gradient (in ms). Example: 30",
        metavar=Metavar.float,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = nothing, 1 = classic, 2 = expended",
        required=False,
        choices=(0, 1, 2),
        default=1)

    return parser


# =======================================================================================================================
# Start program
# =======================================================================================================================
if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

