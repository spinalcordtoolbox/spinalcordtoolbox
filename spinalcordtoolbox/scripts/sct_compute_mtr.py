#!/usr/bin/env python
#########################################################################################
#
# Compute magnetization transfer ratio (MTR).
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-09-21
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import os
import argparse

from spinalcordtoolbox.utils import Metavar, SmartFormatter, init_sct, display_viewer_syntax, printv, set_global_loglevel
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.qmri.mt import compute_mtr


def get_parser():
    parser = argparse.ArgumentParser(
        description='Compute magnetization transfer ratio (MTR). Output is given in percentage.',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-mt0',
        required=True,
        help='Image without MT pulse (MT0)',
        metavar=Metavar.float,
    )
    mandatoryArguments.add_argument(
        '-mt1',
        required=True,
        help='Image with MT pulse (MT1)',
        metavar=Metavar.float,
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-thr",
        type=float,
        help="Threshold to clip MTR output values in case of division by small number. This implies that the output image"
             "range will be [-thr, +thr]. Default: 100.",
        default=100
    )
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit"
    )
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
    )
    optional.add_argument(
        '-o',
        help='Path to output file.',
        metavar=Metavar.str,
        default=os.path.join('.', 'mtr.nii.gz')
    )
    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv if argv else ['--help'])
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    fname_mtr = arguments.o

    # compute MTR
    printv('\nCompute MTR...', verbose)
    nii_mtr = compute_mtr(nii_mt1=Image(arguments.mt1), nii_mt0=Image(arguments.mt0), threshold_mtr=arguments.thr)

    # save MTR file
    nii_mtr.save(fname_mtr, dtype='float32')

    display_viewer_syntax([arguments.mt0, arguments.mt1, fname_mtr])


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

