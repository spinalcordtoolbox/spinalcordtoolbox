#!/usr/bin/env python
#
# Generate QC report
#
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import os
import sys
from typing import Sequence
import textwrap

from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.reports import qc2
from spinalcordtoolbox.utils.sys import init_sct, list2cmdline, __sct_dir__, set_loglevel
from spinalcordtoolbox.utils.shell import SCTArgumentParser


def get_parser():
    parser = SCTArgumentParser(
        description='Generate Quality Control (QC) report following SCT processing.',
        epilog=textwrap.dedent("""
            Examples:

            - `sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc`
            - `sct_qc -i t2.nii.gz -s t2_pmj.nii.gz -p sct_detect_pmj`
            - `sct_qc -i t2.nii.gz -s t2_seg_labeled.nii.gz -p sct_label_vertebrae`
            - `sct_qc -i t2.nii.gz -s t2_seg.nii.gz -p sct_deepseg_sc -qc-dataset mydata -qc-subject sub-45`
            - `sct_qc -i t2.nii.gz -s t2_seg.nii.gz -d t2_lesion.nii.gz -p sct_deepseg_lesion -plane axial`
        """),
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-i',
        metavar='IMAGE',
        help='Input image #1')
    mandatory.add_argument(
        '-p',
        help='SCT function associated with the QC report to generate',
        choices=('sct_propseg', 'sct_deepseg_sc', 'sct_deepseg_gm', 'sct_deepseg_lesion',
                 'sct_register_multimodal', 'sct_register_to_template', 'sct_warp_template',
                 'sct_label_vertebrae', 'sct_detect_pmj', 'sct_label_utils', 'sct_get_centerline',
                 'sct_fmri_moco', 'sct_dmri_moco', 'sct_image_stitch', 'sct_fmri_compute_tsnr'))

    optional = parser.optional_arggroup
    optional.add_argument(
        '-s',
        metavar='SEG',
        help='Input segmentation or label')
    optional.add_argument(
        '-d',
        metavar='DEST',
        help='Input image #2 to overlay on image #1 (requires a segmentation), or output of another '
             'process. Example: `sct_straighten_spinalcord`')
    optional.add_argument(
        '-plane',
        help='Plane of the output QC. Only relevant for `-p sct_deepseg_lesion`.',
        choices=('axial', 'sagittal'))
    optional.add_argument(
        '-resample',
        help='Millimeter resolution to resample the image to. Set to 0 to turn off resampling. You can '
             'use this option to control the zoom of the QC report: higher values will result in '
             'smaller images, and lower values will result in larger images.',
        type=float)
    optional.add_argument(
        '-text-labels',
        help="If set to 0, text won't be drawn on top of labels. Only relevant for `-p sct_label_vertebrae`.",
        choices=(0, 1),
        default=1,
        type=int)
    optional.add_argument(
        '-custom-labels',
        metavar="JSON",
        help="Path to a JSON file containing custom region labels. Only relevant for `-p sct_label_vertebrae`.",
        default=os.path.join(__sct_dir__, 'spinalcordtoolbox', 'reports', 'sct_label_vertebrae_regions.json'))
    optional.add_argument(
        '-qc',
        metavar='QC',
        help='Path to save QC report.',
        default=os.path.join('.', 'qc'))
    optional.add_argument(
        '-qc-dataset',
        metavar='DATASET',
        help='If provided, this string will be mentioned in the QC report as the dataset the process '
             'was run on')
    optional.add_argument(
        '-qc-subject',
        metavar='SUBJECT',
        help='If provided, this string will be mentioned in the QC report as the subject the process '
             'was run on')
    optional.add_argument(
        '-fps',
        metavar='float',
        type=float,
        help='The number of frames per second for output gif images. Only useful for sct_fmri_moco and '
             'sct_dmri_moco.')

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    if arguments.p == 'sct_deepseg_lesion' and arguments.plane is None:
        parser.error('Please provide the plane of the output QC with `-plane`')

    # Common arguments for QC reports
    kwargs = dict(
        fname_input=arguments.i,
        fname_output=arguments.d,
        fname_seg=arguments.s,
        argv=argv,
        path_qc=arguments.qc,
        dataset=arguments.qc_dataset,
        subject=arguments.qc_subject,
    )
    if arguments.resample is None:
        assert 'p_resample' not in kwargs  # Use the report's default
    elif arguments.resample == 0:
        kwargs['p_resample'] = None  # Explicitly turn it off
    else:
        kwargs['p_resample'] = arguments.resample

    if arguments.p in ['sct_register_multimodal', 'sct_register_to_template']:
        qc2.sct_register(command=arguments.p, **kwargs)
    elif arguments.p == 'sct_fmri_compute_tsnr':
        qc2.sct_fmri_compute_tsnr(**kwargs)
    elif arguments.p == 'sct_label_vertebrae':
        del kwargs['fname_output']  # not used by this report
        qc2.sct_label_vertebrae(
            command=arguments.p,
            draw_text=bool(arguments.text_labels),
            path_custom_labels=arguments.custom_labels,
            **kwargs
        )
    elif arguments.p == 'sct_label_utils':
        del kwargs['fname_output']  # not used by this report
        qc2.sct_label_utils(command=arguments.p, **kwargs)
    else:
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
                    )


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
