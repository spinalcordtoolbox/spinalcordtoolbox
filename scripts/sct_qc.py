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
        description='Generate Quality Control (QC) report following SCT processing.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='Examples:\n'
               'sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc\n'
               'sct_qc -i t2.nii.gz -s t2_seg_labeled.nii.gz -p sct_label_vertebrae'
    )
    parser.add_argument('-i',
                        metavar='IMAGE',
                        help='Input image',
                        required=True)
    parser.add_argument('-p',
                        help='SCT function associated with the QC report to generate',
                        choices=('sct_propseg', 'sct_deepseg_sc', 'sct_deepseg_gm', 'sct_register_multimodal',
                                 'sct_register_to_template', 'sct_warp_template', 'sct_label_vertebrae',
                                 'sct_detect_pmj'),
                        required=True)
    parser.add_argument('-s',
                        metavar='SEG',
                        help='Segmentation image',
                        required=False)
    parser.add_argument('-d',
                        metavar='DEST',
                        help='Second image to overlay on the first image (requires a segmentation)',
                        required=False)
    parser.add_argument('-qc',
                        metavar='QC',
                        help='Path to output QC folder. Default: ./qc',
                        required=False,
                        default='./qc')
    parser.add_argument('-v',
                        help='Verbosity: 0: no verbosity, 1: verbosity (default).',
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
