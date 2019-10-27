#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Compute MT saturation map and T1 map from a PD-weigthed, a T1-weighted and MT-weighted FLASH images
#
# Reference paper:
#    Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of magnetization transfer with inherent correction
#    for RF inhomogeneity and T1 relaxation obtained from 3D FLASH MRI. Magn Reson Med 2008;60(6):1396-1407.

# ---------------------------------------------------------------------------------------
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import, division

import sys
import os
import argparse

from spinalcordtoolbox.utils import Metavar, SmartFormatter
from spinalcordtoolbox.qmri.mt import compute_mtsat
from spinalcordtoolbox.image import Image

import sct_utils as sct


def get_parser():
    parser = argparse.ArgumentParser(
        description='Compute MTsat and T1map. '
                    'Reference: Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of magnetization '
                    'transfer with inherent correction for RF inhomogeneity and T1 relaxation obtained from 3D FLASH '
                    'MRI. Magn Reson Med 2008;60(6):1396-1407.',
        add_help=False,
        formatter_class=SmartFormatter,
        prog= os.path.basename(__file__).strip(".py")
    )

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        "-mt",
        required=True,
        help="Image with MT_ON",
        metavar=Metavar.file,
        )
    mandatoryArguments.add_argument(
        "-pd",
        required=True,
        help="Image PD weighted (typically, the MT_OFF)",
        metavar=Metavar.file,
        )
    mandatoryArguments.add_argument(
        "-t1",
        required=True,
        help="Image T1-weighted",
        metavar=Metavar.file,
        )
    mandatoryArguments.add_argument(
        "-trmt",
        required=True,
        help="TR [in ms] for mt image.",
        type=float,
        metavar=Metavar.float,
        )
    mandatoryArguments.add_argument(
        "-trpd",
        required=True,
        help="TR [in ms] for pd image.",
        type=float,
        metavar=Metavar.float,
        )
    mandatoryArguments.add_argument(
        "-trt1",
        required=True,
        help="TR [in ms] for t1 image.",
        type=float,
        metavar=Metavar.float,
        )
    mandatoryArguments.add_argument(
        "-famt",
        required=True,
        help="Flip angle [in deg] for mt image.",
        type=float,
        metavar=Metavar.float,
        )
    mandatoryArguments.add_argument(
        "-fapd",
        required=True,
        help="Flip angle [in deg] for pd image.",
        type=float,
        metavar=Metavar.float,
        )
    mandatoryArguments.add_argument(
        "-fat1",
        required=True,
        help="Flip angle [in deg] for t1 image.",
        type=float,
        metavar=Metavar.float,
        )

    optional = parser.add_argument_group('\nOPTIONAL ARGUMENTS')
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-b1map",
        help="B1 map",
        metavar=Metavar.file,
        default=None)
    optional.add_argument(
        "-omtsat",
        metavar=Metavar.str,
        help="Output file for MTsat. Default is mtsat.nii.gz",
        default=None)
    optional.add_argument(
        "-ot1map",
        metavar=Metavar.str,
        help="Output file for T1map. Default is t1map.nii.gz",
        default=None)
    optional.add_argument(
        "-v",
        help="Verbose: 0 = no verbosity, 1 = verbose (default).",
        type=int,
        choices=(0, 1),
        default=1)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    verbose = args.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    sct.printv('Load data...', verbose)
    nii_mt = Image(args.mt)
    nii_pd = Image(args.pd)
    nii_t1 = Image(args.t1)
    if args.b1map is None:
        nii_b1map = None
    else:
        nii_b1map = Image(args.b1map)

    # compute MTsat
    nii_mtsat, nii_t1map = compute_mtsat(nii_mt, nii_pd, nii_t1, args.trmt, args.trpd, args.trt1, args.famt, args.fapd,
                                         args.fat1, nii_b1map=nii_b1map)

    # Output MTsat and T1 maps
    # by default, output in the same directory as the input images
    sct.printv('Generate output files...', verbose)
    if args.omtsat is None:
        fname_mtsat = os.path.join(os.path.dirname(nii_mt.absolutepath), "mtsat.nii.gz")
    else:
        fname_mtsat = args.omtsat
    nii_mtsat.save(fname_mtsat)
    if args.ot1map is None:
        fname_t1map = os.path.join(os.path.dirname(nii_mt.absolutepath), "t1map.nii.gz")
    else:
        fname_t1map = args.ot1map
    nii_t1map.save(fname_t1map)

    sct.display_viewer_syntax([fname_mtsat, fname_t1map],
                              colormaps=['gray', 'gray'],
                              minmax=['-10,10', '0, 3'],
                              opacities=['1', '1'],
                              verbose=verbose)


if __name__ == '__main__':
    main()
