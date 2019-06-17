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

import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description='Compute MTsat and T1map. '
                    'Reference: Helms G, Dathe H, Kallenberg K, Dechent P. High-resolution maps of magnetization '
                    'transfer with inherent correction for RF inhomogeneity and T1 relaxation obtained from 3D FLASH '
                    'MRI. Magn Reson Med 2008;60(6):1396-1407.',
        add_help=False,
        usage= '''sct_compute_mtsat [-h] -mt MT -pd PD -t1 T1 -trmt TRMT -trpd TRPD
                                    -trt1 TRT1 -famt FAMT -fapd FAPD -fat1 FAT1
                                    [-b1map B1MAP] [-omtsat OMTSAT] [-ot1map OT1MAP]
                                    [-v {0,1}]'''
        )
    mandatoryArguments = parser.add_argument_group("\nMandatory arguments")
    mandatoryArguments.add_argument("-mt",
                        help="Image with MT_ON",
                        required=True)
    mandatoryArguments.add_argument("-pd",
                        help="Image PD weighted (typically, the MT_OFF)",
                        required=True)
    mandatoryArguments.add_argument("-t1",
                        help="Image T1-weighted",
                        required=True)
    mandatoryArguments.add_argument("-trmt",
                        help="TR [in ms] for mt image.",
                        type=float,
                        required=True)
    mandatoryArguments.add_argument("-trpd",
                        help="TR [in ms] for pd image.",
                        type=float,
                        required=True)
    mandatoryArguments.add_argument("-trt1",
                        help="TR [in ms] for t1 image.",
                        type=float,
                        required=True)
    mandatoryArguments.add_argument("-famt",
                        help="Flip angle [in deg] for mt image.",
                        type=float,
                        required=True)
    mandatoryArguments.add_argument("-fapd",
                        help="Flip angle [in deg] for pd image.",
                        type=float,
                        required=True)
    mandatoryArguments.add_argument("-fat1",
                        help="Flip angle [in deg] for t1 image.",
                        type=float,
                        required=True)
    optional = parser.add_argument_group('\nOptional arguments')
    optional.add_argument("-h",
                          "--help",
                          action="help",
                          help="show this help message and exit"
                          )
    optional.add_argument("-b1map",
                        help="B1 map",
                        default=None)
    optional.add_argument("-omtsat",
                        help="Output file for MTsat. Default is mtsat.nii.gz",
                        default=None)
    optional.add_argument("-ot1map",
                        help="Output file for T1map. Default is t1map.nii.gz",
                        default=None)
    optional.add_argument("-v",
                        help="Verbose: 0 = no verbosity, 1 = verbose (default).",
                        choices=('0', '1'),
                        type=int,
                        default=1)
    return parser


def main(args):
    import sct_utils as sct
    from spinalcordtoolbox.mtsat import mtsat

    verbose = args.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    fname_mtsat, fname_t1map = mtsat.compute_mtsat_from_file(
        args.mt, args.pd, args.t1, args.trmt, args.trpd, args.trt1, args.famt, args.fapd, args.fat1,
        fname_b1map=args.b1map, fname_mtsat=args.omtsat, fname_t1map=args.ot1map, verbose=verbose)

    sct.display_viewer_syntax([fname_mtsat, fname_t1map],
                              colormaps=['gray', 'gray'],
                              minmax=['-10,10', '0, 3'],
                              opacities=['1', '1'],
                              verbose=verbose)


if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    main(arguments)
