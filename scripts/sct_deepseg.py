#!/usr/bin/env python
# -*- coding: utf-8
# This command-line tool is the interface for the deepseg API that performs segmentation using deep learning from the
# ivadomed package.

from __future__ import absolute_import

import sys
import os
import argparse
import nibabel as nib

from ivadomed.utils import segment_volume

from spinalcordtoolbox.utils import Metavar, SmartFormatter

from sct_utils import add_suffix, init_sct


def get_parser():
    parser = argparse.ArgumentParser(
        description="Segmentation using deep learning.",
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        help="Image to segment.",
        metavar=Metavar.file)
    mandatory.add_argument(
        # TODO: need to find a better strategy here
        "-m",
        help="Path of the model to use.",
        default='')

    paramseg = parser.add_argument_group("\nSEGMENTATION PARAMETERS")
    paramseg.add_argument(
        "-thr",
        type=float,
        help='Threshold to apply in the segmentation predictions, use 0 (zero) to disable it. Example: 0.999',
        metavar=Metavar.float,
        default=0.999)
    paramseg.add_argument(
        "-t",
        help="Enable TTA (test-time augmentation). "
             "Better results, but takes more time and "
             "provides non-deterministic results.",
        metavar='')

    misc = parser.add_argument_group('\nMISC')
    misc.add_argument(
        "-o",
        help="Output segmentation. In case multi-class segmentation, suffixes will be added.",
        metavar=Metavar.file,
        default=None)
    misc.add_argument(
        '-qc',
        help="The path where the quality control generated content will be saved.",
        metavar=Metavar.str,
        default=None)
    misc.add_argument(
        '-qc-dataset',
        help='If provided, this string will be mentioned in the QC report as the dataset the process was run on',
        metavar=Metavar.str)
    misc.add_argument(
        '-qc-subject',
        help='If provided, this string will be mentioned in the QC report as the subject the process was run on',
        metavar=Metavar.str)
    misc.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = no verbosity, 1 = verbose.",
        choices=(0, 1),
        default=1)
    misc.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")

    return parser


def run_main():
    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    nii_seg = segment_volume(args.m, args.i, fname_roi=None)

    # TODO: use args to get output name
    nib.save(nii_seg, add_suffix(args.i, '_seg'))


if __name__ == '__main__':
    init_sct()
    run_main()
