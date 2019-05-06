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
import sct_utils as sct


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generate Quality Control (QC) report following SCT processing.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog='Examples:\n'
               'sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc\n'
               'sct_qc -i t2.nii.gz -s t2_seg_labeled.nii.gz -p sct_label_vertebrae\n'
               'sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc -qc-dataset mydata -qc-subject sub-45'
    )
    parser.add_argument('-i',
                        metavar='IMAGE',
                        help='Input image #1 (mandatory)',
                        required=True)
    parser.add_argument('-p',
                        help='SCT function associated with the QC report to generate',
                        choices=('sct_propseg', 'sct_deepseg_sc', 'sct_deepseg_gm', 'sct_register_multimodal',
                                 'sct_register_to_template', 'sct_warp_template', 'sct_label_vertebrae',
                                 'sct_detect_pmj'),
                        required=True)
    parser.add_argument('-s',
                        metavar='SEG',
                        help='Input segmentation',
                        required=False)
    parser.add_argument('-d',
                        metavar='DEST',
                        help='Input image #2 to overlay on image #1 (requires a segmentation), or output of another '
                             'process (e.g., sct_straighten_spinalcord)',
                        required=False)
    parser.add_argument('-qc',
                        metavar='QC',
                        help='Path to save QC report. Default: ./qc',
                        required=False,
                        default='./qc')
    parser.add_argument('-qc-dataset',
                        metavar='DATASET',
                        help='If provided, this string will be mentioned in the QC report as the dataset the process '
                             'was run on',
                        required=False)
    parser.add_argument('-qc-subject',
                        metavar='SUBJECT',
                        help='If provided, this string will be mentioned in the QC report as the subject the process '
                             'was run on',
                        required=False)
    return parser


def main(args):
    from spinalcordtoolbox.reports.qc import generate_qc

    # Build args list (for display)
    args_disp = '-i ' + args.i
    if args.d:
        args_disp += ' -d ' + args.d
    if args.s:
        args_disp += ' -s ' + args.s
    generate_qc(fname_in1=args.i,
                fname_in2=args.d,
                fname_seg=args.s,
                args=args_disp,
                path_qc=args.qc,
                dataset=args.qc_dataset,
                subject=args.qc_subject,
                process=args.p)


if __name__ == '__main__':
    sct.init_sct()
    parser = get_parser()
    arguments = parser.parse_args()
    main(arguments)
