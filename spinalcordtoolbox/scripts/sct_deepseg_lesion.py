#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Function to segment the multiple sclerosis lesions using convolutional neural networks
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Charley Gros
# Modified: 2018-06-11
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys

from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.fs import extract_fname


def get_parser():
    parser = SCTArgumentParser(
        description='MS lesion Segmentation using convolutional networks. Reference: Gros C et al. Automatic'
                    'segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional'
                    'neural networks. Neuroimage. 2018 Oct 6;184:901-915.'
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        help='Input image. Example: t1.nii.gz',
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-c",
        required=True,
        help='Type of image contrast.\n'
             ' t2: T2w scan with isotropic or anisotropic resolution.\n'
             ' t2_ax: T2w scan with axial orientation and thick slices.\n'
             ' t2s: T2*w scan with axial orientation and thick slices.',
        choices=('t2', 't2_ax', 't2s'),
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit",
    )
    optional.add_argument(
        "-centerline",
        help="Method used for extracting the centerline:\n"
             " svm: Automatic detection using Support Vector Machine algorithm.\n"
             " cnn: Automatic detection using Convolutional Neural Network.\n"
             " viewer: Semi-automatic detection using manual selection of a few points with an interactive viewer "
             "followed by regularization.\n"
             " file: Use an existing centerline (use with flag -file_centerline)",
        required=False,
        choices=('svm', 'cnn', 'viewer', 'file'),
        default="svm")
    optional.add_argument(
        "-file_centerline",
        help='Input centerline file (to use with flag -centerline manual). Example: t2_centerline_manual.nii.gz',
        metavar=Metavar.str,
        required=False)
    optional.add_argument(
        "-brain",
        type=int,
        help='Indicate if the input image contains brain sections (to speed up segmentation). This flag is only '
             'effective with "-centerline cnn".',
        required=False,
        choices=(0, 1),
        default=1)
    optional.add_argument(
        "-ofolder",
        help='Output folder. Example: My_Output_Folder',
        required=False,
        action=ActionCreateFolder,
        metavar=Metavar.str,
        default=os.getcwd())
    optional.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        required=False,
        choices=(0, 1),
        default=1)
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")
    optional.add_argument(
        '-igt',
        metavar=Metavar.str,
        help='File name of ground-truth segmentation.',
        required=False)

    return parser


def main(argv=None):
    """Main function."""
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    fname_image = arguments.i
    contrast_type = arguments.c

    ctr_algo = arguments.centerline

    brain_bool = bool(arguments.brain)
    if arguments.brain is None and contrast_type in ['t2s', 't2_ax']:
        brain_bool = False

    output_folder = arguments.ofolder

    if ctr_algo == 'file' and arguments.file_centerline is None:
        printv('Please use the flag -file_centerline to indicate the centerline filename.', 1, 'error')
        sys.exit(1)

    if arguments.file_centerline is not None:
        manual_centerline_fname = arguments.file_centerline
        ctr_algo = 'file'
    else:
        manual_centerline_fname = None

    remove_temp_files = arguments.r

    algo_config_stg = '\nMethod:'
    algo_config_stg += '\n\tCenterline algorithm: ' + str(ctr_algo)
    algo_config_stg += '\n\tAssumes brain section included in the image: ' + str(brain_bool) + '\n'
    printv(algo_config_stg)

    # Segment image
    from spinalcordtoolbox.image import Image
    from spinalcordtoolbox.deepseg_.lesion import deep_segmentation_MSlesion
    im_image = Image(fname_image)
    im_seg, im_labels_viewer, im_ctr = deep_segmentation_MSlesion(im_image, contrast_type, ctr_algo=ctr_algo, ctr_file=manual_centerline_fname,
                                        brain_bool=brain_bool, remove_temp_files=remove_temp_files, verbose=verbose)

    # Save segmentation
    fname_seg = os.path.abspath(os.path.join(output_folder, extract_fname(fname_image)[1] + '_lesionseg' +
                                             extract_fname(fname_image)[2]))
    im_seg.save(fname_seg)

    if ctr_algo == 'viewer':
        # Save labels
        fname_labels = os.path.abspath(os.path.join(output_folder, extract_fname(fname_image)[1] + '_labels-centerline' +
                                               extract_fname(fname_image)[2]))
        im_labels_viewer.save(fname_labels)

    if verbose == 2:
        # Save ctr
        fname_ctr = os.path.abspath(os.path.join(output_folder, extract_fname(fname_image)[1] + '_centerline' +
                                               extract_fname(fname_image)[2]))
        im_ctr.save(fname_ctr)

    display_viewer_syntax([fname_image, fname_seg], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

