#!/usr/bin/env python
#
# Generate QC report
#
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import sys
from typing import Sequence

from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.utils.sys import init_sct, list2cmdline, set_loglevel
from spinalcordtoolbox.utils.shell import SCTArgumentParser


def get_parser():
    parser = SCTArgumentParser(
        description='Generate Quality Control (QC) report following SCT processing.',
        epilog='Examples:\n'
               'sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc\n'
               'sct_qc -i t2.nii.gz -s t2_pmj.nii.gz -p sct_detect_pmj\n'
               'sct_qc -i t2.nii.gz -s t2_seg_labeled.nii.gz -p sct_label_vertebrae\n'
               'sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc -qc-dataset mydata -qc-subject sub-45\n'
               'sct_qc -i t2.nii.gz -s t2_seg.nii.gz -d t2_lesion.nii.gz -p sct_deepseg_lesion -plane axial'
    )
    parser.add_argument('-i',
                        metavar='IMAGE',
                        help='Input image #1 (mandatory)',
                        required=True)
    parser.add_argument('-p',
                        help='SCT function associated with the QC report to generate',
                        choices=('sct_propseg', 'sct_deepseg_sc', 'sct_deepseg_gm', 'sct_deepseg_lesion',
                                 'sct_register_multimodal', 'sct_register_to_template', 'sct_warp_template',
                                 'sct_label_vertebrae', 'sct_detect_pmj', 'sct_label_utils', 'sct_get_centerline',
                                 'sct_fmri_moco', 'sct_dmri_moco', 'sct_image_stitch', 'sct_fmri_compute_tsnr'),
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
    parser.add_argument('-plane',
                        help='Plane of the output QC. Only relevant for -p sct_deepseg_lesion.',
                        choices=('axial', 'sagittal'),
                        required=False)
    parser.add_argument('-resample',
                        help='Millimeter resolution to resample the image to. Set to 0 to turn off resampling. You can '
                             'use this option to control the zoom of the QC report: higher values will result in '
                             'smaller images, and lower values will result in larger images.',
                        type=float,
                        required=False)
    parser.add_argument('-text-labels',
                        help="If set to 0, text won't be drawn on top of labels. Only relevant for -p "
                             "sct_label_vertebrae.",
                        choices=(0, 1),
                        default=1,
                        type=int,
                        required=False)
    parser.add_argument('-qc',
                        metavar='QC',
                        help='Path to save QC report. Default: ./qc',
                        required=False,
                        default=os.path.join('.', 'qc'))
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


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    if arguments.p == 'sct_deepseg_lesion' and arguments.plane is None:
        parser.error('Please provide the plane of the output QC with -plane')

    generate_qc(fname_in1=arguments.i,
                fname_in2=arguments.d,
                fname_seg=arguments.s,
                # Internal functions use capitalized strings ('Axial'/'Sagittal')
                plane=arguments.plane.capitalize() if isinstance(arguments.plane, str) else arguments.plane,
                args=f'("sct_qc {list2cmdline(argv)}")',
                path_qc=arguments.qc,
                dataset=arguments.qc_dataset,
                subject=arguments.qc_subject,
                process=arguments.p,
                fps=arguments.fps,
                p_resample=arguments.resample,
                draw_text=bool(arguments.text_labels))


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
