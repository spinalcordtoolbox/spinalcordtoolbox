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

from __future__ import absolute_import, division

import sys

import sct_utils as sct
import os
import argparse


# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation

    parser = argparse.ArgumentParser(
        description='Compute Maximum Spinal Cord Compression (MSCC) as in: Miyanji F, Furlan JC, Aarabi B, Arnold PM, Fehlings MG. Acute cervical traumatic spinal cord injury: MR imaging findings correlated with neurologic outcome--prospective study with 100 consecutive patients. Radiology 2007;243(3):820-827.',
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
    )
    mandatoryArguments = parser.add_argument_group("\nMandatory arguments")
    mandatoryArguments.add_argument(
        '-di',
        type=float,
        help='Anteroposterior cord distance at the level of maximum injury, (e.g. "6.85")',
        required = True)
    mandatoryArguments.add_argument(
        '-da',
        type=float,
        help='Anteroposterior cord distance at the nearest normal level above the level of injury, (e.g. "7.65")',
        required = True)
    mandatoryArguments.add_argument(
        '-db',
        type=float,
        help='Anteroposterior cord distance at the nearest normal level below the level of injury, (e.g. "7.02")',
        required = True)
    optional = parser.add_argument_group("\nOptional arguments")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit")

    return parser


def mscc(di, da, db):
    return (1 - float(di) / ((da + db) / float(2))) * 100


# MAIN
# ==========================================================================================
def main(arguments):
    # initialization
    verbose = 1
    # Get parser info
    di = arguments.di
    da = arguments.da
    db = arguments.db

    # Compute MSCC
    MSCC = mscc(di, da, db)

    # Display results
    sct.printv('\nMSCC = ' + str(MSCC) + '\n', verbose, 'info')


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    parser = get_parser()
    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    # call main function
    main(arguments)
