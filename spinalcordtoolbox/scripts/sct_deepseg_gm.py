#!/usr/bin/env python
# This command-line tool is the interface for the deepseg_gm API
# that implements the model for the Spinal Cord Gray Matter Segmentation.
#
# Reference paper:
#     Perone, C. S., Calabrese, E., & Cohen-Adad, J. (2017).
#     Spinal cord gray matter segmentation using deep dilated convolutions.
#     URL: https://arxiv.org/abs/1710.01269

import sys
import os

from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, display_viewer_syntax, set_loglevel
from spinalcordtoolbox.image import add_suffix
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.deepseg_.gm import segment_file


def get_parser():
    parser = SCTArgumentParser(
        description='Spinal Cord Gray Matter (GM) Segmentation using deep dilated convolutions. The contrast of the '
                    'input image must be similar to a T2*-weighted image: WM dark, GM bright and CSF bright. '
                    'Reference: Perone CS, Calabrese E, Cohen-Adad J. Spinal cord gray matter segmentation using deep '
                    'dilated convolutions. Sci Rep 2018;8(1):5966.'
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        help="Image filename to segment (3D volume). Example: t2s.nii.gz.",
        metavar=Metavar.file
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-o",
        help="Output segmentation file name. Example: sc_gm_seg.nii.gz",
        metavar=Metavar.file,
        default=None)
    misc = parser.add_argument_group('\nMISC')
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
        "-m",
        help="Model to use (large or challenge). "
             "The model 'large' will be slower but "
             "will yield better results. The model "
             "'challenge' was built using data from "
             "the following challenge: goo.gl/h4AVar.",
        choices=('large', 'challenge'),
        default='large')
    misc.add_argument(
        "-thr",
        type=float,
        help='Threshold to apply in the segmentation predictions, use 0 (zero) to disable it. Example: 0.999',
        metavar=Metavar.float,
        default=0.999)
    misc.add_argument(
        "-t",
        help="Enable TTA (test-time augmentation). "
             "Better results, but takes more time and "
             "provides non-deterministic results.",
        metavar='')
    misc.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    input_filename = arguments.i
    if arguments.o is not None:
        output_filename = arguments.o
    else:
        output_filename = add_suffix(input_filename, '_gmseg')

    use_tta = arguments.t
    model_name = arguments.m
    threshold = arguments.thr

    if threshold > 1.0 or threshold < 0.0:
        raise RuntimeError("Threshold should be between 0.0 and 1.0.")

    # Threshold zero means no thresholding
    if threshold == 0.0:
        threshold = None

    out_fname = segment_file(input_filename, output_filename,
                             model_name, threshold, int(verbose),
                             use_tta)

    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    if path_qc is not None:
        generate_qc(fname_in1=input_filename, fname_seg=out_fname, args=sys.argv[1:], path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, process='sct_deepseg_gm')

    display_viewer_syntax([input_filename, format(out_fname)],
                          colormaps=['gray', 'red'],
                          opacities=['1', '0.7'],
                          verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
