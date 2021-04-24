#!/usr/bin/env python
# -*- coding: utf-8
#
# Generate QC report
#
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT

import sys

from spinalcordtoolbox.utils import init_sct, set_global_loglevel, SCTArgumentParser


def get_parser():
    parser = SCTArgumentParser(
        description='Generate Quality Control (QC) report following SCT processing.',
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
                                 'sct_detect_pmj', 'sct_label_utils', 'sct_get_centerline', 'sct_fmri_moco',
                                 'sct_dmri_moco'),
                        required=True)
    parser.add_argument('-s',
                        metavar='SEG',
                        help='Input segmentation or label',
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
    parser.add_argument('-fps',
                        metavar='float',
                        type=float,
                        help='The number of frames per second for output gif images. Only useful for sct_fmri_moco and '
                             'sct_dmri_moco.',
                        required=False)
    parser.add_argument('-v',
                        action='store_true',
                        help="Verbose")
    parser.add_argument('-h',
                        '--help',
                        action="help",
                        help="show this message and exit")

    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    from spinalcordtoolbox.reports.qc import generate_qc
    # Build args list (for display)
    args_disp = '-i ' + arguments.i
    if arguments.d:
        args_disp += ' -d ' + arguments.d
    if arguments.s:
        args_disp += ' -s ' + arguments.s
    generate_qc(fname_in1=arguments.i,
                fname_in2=arguments.d,
                fname_seg=arguments.s,
                args=args_disp,
                path_qc=arguments.qc,
                dataset=arguments.qc_dataset,
                subject=arguments.qc_subject,
                process=arguments.p,
                fps=arguments.fps,)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

