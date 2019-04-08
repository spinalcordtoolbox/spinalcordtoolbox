#!/usr/bin/env python
# -*- coding: utf-8
#
# Generate QC report
#
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT

from __future__ import absolute_import, division

import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generate Quality Control (QC) report following SCT processing.'
    )
    parser.add_argument("-i",
                        help="Input image",
                        required=True)
    parser.add_argument("-p",
                        help="Process (or SCT function) associated with the QC report you would like to generate",
                        required=True)
    parser.add_argument("-s",
                        help="Segmentation",
                        required=False)
    parser.add_argument("-qc",
                        help="Path to output QC folder. Default: ./qc",
                        required=False,
                        default='./qc')
    parser.add_argument("-v",
                        help="Verbose: 0 = no verbosity, 1 = verbose (default).",
                        choices=('0', '1'),
                        type=int,
                        default=1)
    return parser


def main(args):
    from spinalcordtoolbox.reports.qc import generate_qc

    generate_qc(fname_in1=args.i,
                fname_in2=args.d,
                fname_seg=args.s,
                args=None,
                path_qc=args.qc,
                process=args.p)


if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    main(arguments)
