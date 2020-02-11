#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Function to segment the spinal cord using convolutional neural networks
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener & Charley Gros
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import division, absolute_import

import os
import sys
import argparse

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar, SmartFormatter, ActionCreateFolder


def get_parser():
    """Initialize the parser."""

    parser = argparse.ArgumentParser(
        description="Spinal Cord Segmentation using convolutional networks. Reference: Gros et al. Automatic "
                    "segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional "
                    "neural networks. Neuroimage. 2018 Oct 6;184:901-915. ",
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py"))
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        metavar=Metavar.file,
        help='Input image. Example: t1.nii.gz',
    )
    mandatory.add_argument(
        "-c",
        required=True,
        help="Type of image contrast.",
        choices=('t1', 't2', 't2s', 'dwi'),
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit")
    optional.add_argument(
        "-centerline",
        help="R|Method used for extracting the centerline:\n"
             " svm: Automatic detection using Support Vector Machine algorithm.\n"
             " cnn: Automatic detection using Convolutional Neural Network.\n"
             " viewer: Semi-automatic detection using manual selection of a few points with an interactive viewer "
             "followed by regularization.\n"
             " file: Use an existing centerline (use with flag -file_centerline)",
        choices=('svm', 'cnn', 'viewer', 'file'),
        default="svm")
    optional.add_argument(
        "-file_centerline",
        metavar=Metavar.str,
        help='Input centerline file (to use with flag -centerline file). Example: t2_centerline_manual.nii.gz')
    optional.add_argument(
        "-thr",
        type=float,
        help="Binarization threshold (between 0 and 1) to apply to the segmentation prediction. Set to -1 for no "
             "binarization (i.e. soft segmentation output). The default threshold is specific to each contrast and was"
             "estimated using an optimization algorithm. More details at: "
             "https://github.com/sct-pipeline/deepseg-threshold.",
        metavar=Metavar.float,
        default=None)
    optional.add_argument(
        "-brain",
        type=int,
        help='Indicate if the input image contains brain sections (to speed up segmentation). Only use with '
             '"-centerline cnn".',
        choices=(0, 1))
    optional.add_argument(
        "-kernel",
        help="Choice of kernel shape for the CNN. Segmentation with 3D kernels is slower than with 2D kernels.",
        choices=('2d', '3d'),
        default="2d")
    optional.add_argument(
        "-ofolder",
        metavar=Metavar.str,
        help='Output folder. Example: My_Output_Folder/ ',
        action=ActionCreateFolder,
        default=os.getcwd())
    optional.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        choices=(0, 1),
        default=1)
    optional.add_argument(
        "-v",
        type=int,
        help="1: display on (default), 0: display off, 2: extended",
        choices=(0, 1, 2),
        default=1)
    optional.add_argument(
        '-qc',
        metavar=Metavar.str,
        help='The path where the quality control generated content will be saved',
        default=None)
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the dataset the process was run on',)
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the subject the process was run on',)
    optional.add_argument(
        '-igt',
        metavar=Metavar.str,
        help='File name of ground-truth segmentation.',)

    return parser


def main():
    """Main function."""
    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    fname_image = os.path.abspath(args.i)
    contrast_type = args.c

    ctr_algo = args.centerline

    if args.brain is None:
        if contrast_type in ['t2s', 'dwi']:
            brain_bool = False
        if contrast_type in ['t1', 't2']:
            brain_bool = True
    else:
        brain_bool = bool(args.brain)

    if bool(args.brain) and ctr_algo == 'svm':
        sct.printv('Please only use the flag "-brain 1" with "-centerline cnn".', 1, 'warning')
        sys.exit(1)

    kernel_size = args.kernel
    if kernel_size == '3d' and contrast_type == 'dwi':
        kernel_size = '2d'
        sct.printv('3D kernel model for dwi contrast is not available. 2D kernel model is used instead.',
                   type="warning")

    if ctr_algo == 'file' and args.file_centerline is None:
        sct.printv('Please use the flag -file_centerline to indicate the centerline filename.', 1, 'warning')
        sys.exit(1)

    if args.file_centerline is not None:
        manual_centerline_fname = args.file_centerline
        ctr_algo = 'file'
    else:
        manual_centerline_fname = None

    threshold = args.thr
    if threshold is not None:
        if threshold > 1.0 or (threshold < 0.0 and threshold != -1.0):
            raise SyntaxError("Threshold should be between 0 and 1, or equal to -1 (no threshold)")

    remove_temp_files = args.r
    verbose = args.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    path_qc = args.qc
    qc_dataset = args.qc_dataset
    qc_subject = args.qc_subject
    output_folder = args.ofolder

    # check if input image is 2D or 3D
    sct.check_dim(fname_image, dim_lst=[2, 3])

    # Segment image
    from spinalcordtoolbox.image import Image
    from spinalcordtoolbox.deepseg_sc.core import deep_segmentation_spinalcord
    from spinalcordtoolbox.reports.qc import generate_qc

    im_image = Image(fname_image)
    # note: below we pass im_image.copy() otherwise the field absolutepath becomes None after execution of this function
    im_seg, im_image_RPI_upsamp, im_seg_RPI_upsamp = \
        deep_segmentation_spinalcord(im_image.copy(), contrast_type, ctr_algo=ctr_algo,
                                     ctr_file=manual_centerline_fname, brain_bool=brain_bool, kernel_size=kernel_size,
                                     threshold_seg=threshold, remove_temp_files=remove_temp_files, verbose=verbose)

    # Save segmentation
    fname_seg = os.path.abspath(os.path.join(output_folder, sct.extract_fname(fname_image)[1] + '_seg' +
                                             sct.extract_fname(fname_image)[2]))
    im_seg.save(fname_seg)

    # Generate QC report
    if path_qc is not None:
        generate_qc(fname_image, fname_seg=fname_seg, args=sys.argv[1:], path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, process='sct_deepseg_sc')
    sct.display_viewer_syntax([fname_image, fname_seg], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == "__main__":
    sct.init_sct()
    main()
