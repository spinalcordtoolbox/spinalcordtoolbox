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

from __future__ import absolute_import, division

import sys
import os
import argparse

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar, SmartFormatter
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
        type=int,
        choices=(0, 1, 2),
        help='Verbose: 0 = nothing, 1 = classic, 2 = expended',
        default=1
        )
    optional.add_argument(
        '-o',
        help='Path to output file.',
        metavar=Metavar.str,
        default=os.path.join('.', 'mtr.nii.gz')
        )
    return parser


def main():
    # Check input parameters
    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    fname_mtr = args.o
    verbose = args.v

    # compute MTR
    sct.printv('\nCompute MTR...', verbose)
    nii_mtr = compute_mtr(nii_mt1=Image(args.mt1), nii_mt0=Image(args.mt0), threshold_mtr=args.thr)

    # save MTR file
    nii_mtr.save(fname_mtr, dtype='float32')

    sct.display_viewer_syntax([args.mt0, args.mt1, fname_mtr])


if __name__ == "__main__":
    sct.init_sct()
    main()
