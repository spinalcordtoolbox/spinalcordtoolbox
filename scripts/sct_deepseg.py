#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Entry point for all sct deep learning models
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener, Charley Gros, Olivier Vincent
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar, SmartFormatter, ActionCreateFolder


def get_parser():
    """Initialize the parser."""


    parent_parser = argparse.ArgumentParser(add_help=False)
    mandatory = parent_parser.add_argument_group("\nMANDATORY ARGUMENTS")
    optional = parent_parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        metavar=Metavar.file,
        help='Input image. Example: t1.nii.gz',
    )

    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit")
    optional.add_argument(
        "-v",
        type=int,
        help="1: display on (default), 0: display off, 2: extended",
        choices=(0, 1, 2),
        default=1)

    parser = argparse.ArgumentParser(
        description="Segmentation using convolutional networks.",
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py"))


    subparsers = parser.add_subparsers(help='sub-command help',dest='subparser_name')
    parser_sc = subparsers.add_parser('sc',  parents = [parent_parser], help='Spinal Cord segmentation')

    parser_sc.add_argument(
        "-c",
        required=True,
        help="Type of image contrast.",
        choices=('t1', 't2', 't2s', 'dwi'),
    )
    parser_sc.add_argument(
        "-centerline",
        help="R|Method used for extracting the centerline:\n"
             " svm: Automatic detection using Support Vector Machine algorithm.\n"
             " cnn: Automatic detection using Convolutional Neural Network.\n"
             " viewer: Semi-automatic detection using manual selection of a few points with an interactive viewer "
             "followed by regularization.\n"
             " file: Use an existing centerline (use with flag -file_centerline)",
        choices=('svm', 'cnn', 'viewer', 'file'),
        default="svm")
    parser_sc.add_argument(
        "-file_centerline",
        metavar=Metavar.str,
        help='Input centerline file (to use with flag -centerline file). Example: t2_centerline_manual.nii.gz')
    parser_sc.add_argument(
        "-thr",
        type=float,
        help="Binarization threshold (between 0 and 1) to apply to the segmentation prediction. Set to -1 for no "
             "binarization (i.e. soft segmentation output). The default threshold is specific to each contrast and was"
             "estimated using an optimization algorithm. More details at: "
             "https://github.com/sct-pipeline/deepseg-threshold.",
        metavar=Metavar.float,
        default=None)
    parser_sc.add_argument(
        "-brain",
        type=int,
        help='Indicate if the input image contains brain sections (to speed up segmentation). Only use with '
             '"-centerline cnn".',
        choices=(0, 1))
    parser_sc.add_argument(
        "-kernel",
        help="Choice of kernel shape for the CNN. Segmentation with 3D kernels is slower than with 2D kernels.",
        choices=('2d', '3d'),
        default="2d")
    parser_sc.add_argument(
        "-ofolder",
        metavar=Metavar.str,
        help='Output folder. Example: My_Output_Folder/ ',
        action=ActionCreateFolder,
        default=os.getcwd())
    parser_sc.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        choices=(0, 1),
        default=1)

    parser_sc.add_argument(
        '-qc',
        metavar=Metavar.str,
        help='The path where the quality control generated content will be saved',
        default=None)
    parser_sc.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the dataset the process was run on',)
    parser_sc.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the subject the process was run on',)
    parser_sc.add_argument(
        '-igt',
        metavar=Metavar.str,
        help='File name of ground-truth segmentation.',)

    parser_lesion = subparsers.add_parser('lesion', parents = [parent_parser], help='MS lesion segmentation')

    parser_lesion.add_argument(
        "-c",
        required=True,
        help='R|Type of image contrast.\n'
             ' t2: T2w scan with isotropic or anisotropic resolution.\n'
             ' t2_ax: T2w scan with axial orientation and thick slices.\n'
             ' t2s: T2*w scan with axial orientation and thick slices.',
        choices=('t2', 't2_ax', 't2s'),
    )

    parser_lesion.add_argument(
        "-centerline",
        help="R|Method used for extracting the centerline:\n"
             " svm: Automatic detection using Support Vector Machine algorithm.\n"
             " cnn: Automatic detection using Convolutional Neural Network.\n"
             " viewer: Semi-automatic detection using manual selection of a few points with an interactive viewer "
             "followed by regularization.\n"
             " file: Use an existing centerline (use with flag -file_centerline)",
        required=False,
        choices=('svm', 'cnn', 'viewer', 'file'),
        default="svm")
    parser_lesion.add_argument(
        "-file_centerline",
        help='Input centerline file (to use with flag -centerline manual). Example: t2_centerline_manual.nii.gz',
        metavar=Metavar.str,
        required=False)
    parser_lesion.add_argument(
        "-brain",
        type=int,
        help='Indicate if the input image contains brain sections (to speed up segmentation). This flag is only '
             'effective with "-centerline cnn".',
        required=False,
        choices=(0, 1),
        default=1)
    parser_lesion.add_argument(
        "-ofolder",
        help='Output folder. Example: My_Output_Folder/ ',
        required=False,
        action=ActionCreateFolder,
        metavar=Metavar.str,
        default=os.getcwd())
    parser_lesion.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        required=False,
        choices=(0, 1),
        default=1)

    parser_lesion.add_argument(
        '-igt',
        metavar=Metavar.str,
        help='File name of ground-truth segmentation.',
        required=False)


    parser_gm = subparsers.add_parser('gm', parents = [parent_parser], help='Gray matter segmentation')

    parser_gm.add_argument(
        "-o",
        help="Output segmentation file name. Example: sc_gm_seg.nii.gz",
        metavar=Metavar.file,
        default=None)

    parser_gm.add_argument(
        '-qc',
        help="The path where the quality control generated content will be saved.",
        metavar=Metavar.str,
        default=None)
    parser_gm.add_argument(
        '-qc-dataset',
        help='If provided, this string will be mentioned in the QC report as the dataset the process was run on',
        metavar=Metavar.str)
    parser_gm.add_argument(
        '-qc-subject',
        help='If provided, this string will be mentioned in the QC report as the subject the process was run on',
        metavar=Metavar.str)
    parser_gm.add_argument(
        "-m",
        help="Model to use (large or challenge). "
             "The model 'large' will be slower but "
             "will yield better results. The model "
             "'challenge' was built using data from "
             "the following challenge: goo.gl/h4AVar.",
        choices=('large', 'challenge'),
        default='large')
    parser_gm.add_argument(
        "-thr",
        type=float,
        help='Threshold to apply in the segmentation predictions, use 0 (zero) to disable it. Example: 0.999',
        metavar=Metavar.float,
        default=0.999)
    parser_gm.add_argument(
        "-t",
        help="Enable TTA (test-time augmentation). "
             "Better results, but takes more time and "
             "provides non-deterministic results.",
        metavar='')

    return parser


def main():
    """Main function."""
    parser = get_parser()
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])


    if args.subparser_name == "sc":
        call_deepseg_sc(args)
    elif args.subparser_name == "lesion":
        call_deepseg_lesion(args)
    elif args.subparser_name == "gm":
        call_deepseg_gm(args)
    else:
        sct.printv("Error, invalid subparser")


def call_deepseg_sc(args):

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

def call_deepseg_lesion(args):
    fname_image = args.i
    contrast_type = args.c

    ctr_algo = args.centerline

    brain_bool = bool(args.brain)
    if args.brain is None and contrast_type in ['t2s', 't2_ax']:
        brain_bool = False

    output_folder = args.ofolder

    if ctr_algo == 'file' and args.file_centerline is None:
        sct.printv('Please use the flag -file_centerline to indicate the centerline filename.', 1, 'error')
        sys.exit(1)

    if args.file_centerline is not None:
        manual_centerline_fname = args.file_centerline
        ctr_algo = 'file'
    else:
        manual_centerline_fname = None

    remove_temp_files = args.r
    verbose = args.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    algo_config_stg = '\nMethod:'
    algo_config_stg += '\n\tCenterline algorithm: ' + str(ctr_algo)
    algo_config_stg += '\n\tAssumes brain section included in the image: ' + str(brain_bool) + '\n'
    sct.printv(algo_config_stg)

    # Segment image
    from spinalcordtoolbox.image import Image
    from spinalcordtoolbox.deepseg_lesion.core import deep_segmentation_MSlesion
    im_image = Image(fname_image)
    im_seg, im_labels_viewer, im_ctr = deep_segmentation_MSlesion(im_image, contrast_type, ctr_algo=ctr_algo, ctr_file=manual_centerline_fname,
                                        brain_bool=brain_bool, remove_temp_files=remove_temp_files, verbose=verbose)

    # Save segmentation
    fname_seg = os.path.abspath(os.path.join(output_folder, sct.extract_fname(fname_image)[1] + '_lesionseg' +
                                             sct.extract_fname(fname_image)[2]))
    im_seg.save(fname_seg)

    if ctr_algo == 'viewer':
        # Save labels
        fname_labels = os.path.abspath(os.path.join(output_folder, sct.extract_fname(fname_image)[1] + '_labels-centerline' +
                                               sct.extract_fname(fname_image)[2]))
        im_labels_viewer.save(fname_labels)

    if verbose == 2:
        # Save ctr
        fname_ctr = os.path.abspath(os.path.join(output_folder, sct.extract_fname(fname_image)[1] + '_centerline' +
                                               sct.extract_fname(fname_image)[2]))
        im_ctr.save(fname_ctr)

    sct.display_viewer_syntax([fname_image, fname_seg], colormaps=['gray', 'red'], opacities=['', '0.7'])

def call_deepseg_gm(arguments):
    input_filename = arguments.i
    if arguments.o is not None:
        output_filename = arguments.o
    else:
        output_filename = sct.add_suffix(input_filename, '_gmseg')

    use_tta = arguments.t
    model_name = arguments.m
    threshold = arguments.thr
    verbose = arguments.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    if threshold > 1.0 or threshold < 0.0:
        raise RuntimeError("Threshold should be between 0.0 and 1.0.")

    # Threshold zero means no thresholding
    if threshold == 0.0:
        threshold = None

    from spinalcordtoolbox.deepseg_gm import deepseg_gm
    deepseg_gm.check_backend()

    out_fname = deepseg_gm.segment_file(input_filename, output_filename,
                                        model_name, threshold, int(verbose),
                                        use_tta)

    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    if path_qc is not None:
        generate_qc(fname_in1=input_filename, fname_seg=out_fname, args=sys.argv[1:], path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, process='sct_deepseg_gm')

    sct.display_viewer_syntax([input_filename, format(out_fname)],
                              colormaps=['gray', 'red'],
                              opacities=['1', '0.7'],
                              verbose=verbose)


if __name__ == "__main__":
    sct.init_sct()
    main()
