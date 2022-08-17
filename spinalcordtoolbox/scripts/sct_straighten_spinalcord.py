#!/usr/bin/env python
#
# This program takes as input an anatomic image and the centerline or segmentation of its spinal cord (that you can get
# using sct_get_centerline.py or sct_segmentation_propagation) and returns the anatomic image where the spinal
# cord was straightened.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Geoffrey Leveque, Julien Touati
# Modified: 2014-09-01
#
# License: see the LICENSE.TXT
# ======================================================================================================================

import sys
import os

from spinalcordtoolbox.straightening import SpinalCordStraightener
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel, removesuffix


def get_parser():
    parser = SCTArgumentParser(
        description="This program takes as input an anatomic image and the spinal cord centerline (or "
                    "segmentation), and returns the an image of a straightened spinal cord. Reference: "
                    "De Leener B, Mangeat G, Dupont S, Martin AR, Callot V, Stikov N, Fehlings MG, "
                    "Cohen-Adad J. Topologically-preserving straightening of spinal cord MRI. J Magn "
                    "Reson Imaging. 2017 Oct;46(4):1209-1219"
    )
    mandatory = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        metavar=Metavar.file,
        help='Input image with curved spinal cord. Example: "t2.nii.gz"',
        required=True)
    mandatory.add_argument(
        "-s",
        metavar=Metavar.file,
        help='Spinal cord centerline (or segmentation) of the input image. To obtain the centerline, you can use '
             'sct_get_centerline. To obtain the segmentation you can use sct_propseg or sct_deepseg_sc. '
             'Example: centerline.nii.gz',
        required=True)
    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-dest",
        metavar=Metavar.file,
        help="Spinal cord centerline (or segmentation) of a destination image (which could be "
             "straight or curved). An algorithm scales the length of the input centerline to match that of the "
             "destination centerline. If using -ldisc-input and -ldisc-dest with this parameter, "
             "instead of linear scaling, the source centerline will be non-linearly matched so "
             "that the inter-vertebral discs of the input image will match that of the "
             "destination image. This feature is particularly useful for registering to a "
             "template while accounting for disc alignment.",
        required=False)
    optional.add_argument(
        "-ldisc-input",
        metavar=Metavar.file,
        help="Labels located at the posterior edge of the intervertebral discs, for the input  "
        "image (-i). All disc covering the region of interest should be provided. Exmaple: if "
        "you are interested in levels C2 to C7, then you should provide disc labels 2,3,4,5,"
        "6,7). More details about label creation at "
        "http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/. "  # TODO (Julien) update this link
        "This option must be used with the -ldisc-dest parameter.",
        required=False)
    optional.add_argument(
        "-ldisc-dest",
        metavar=Metavar.file,
        help="Labels located at the posterior edge of the intervertebral discs, for the destination file (-dest). "
             "The same comments as in -ldisc-input apply. This option must be used with the -ldisc-input parameter.",
        required=False)
    optional.add_argument(
        "-disable-straight2curved",
        action='store_true',
        help="Disable straight to curved transformation computation, in case you do not need the "
             "output warping field straight-->curve (faster).",
        required=False)
    optional.add_argument(
        "-disable-curved2straight",
        action='store_true',
        help="Disable curved to straight transformation computation, in case you do not need the "
             "output warping field curve-->straight (faster).",
        required=False)
    optional.add_argument(
        "-speed-factor",
        metavar=Metavar.float,
        type=float,
        help='Acceleration factor for the calculation of the straightening warping field.'
             ' This speed factor enables an intermediate resampling to a lower resolution, which '
             'decreases the computational time at the cost of lower accuracy.'
             ' A speed factor of 2 means that the input image will be downsampled by a factor 2 '
             'before calculating the straightening warping field. For example, a 1x1x1 mm^3 image '
             'will be downsampled to 2x2x2 mm3, providing a speed factor of approximately 8.'
             ' Note that accelerating the straightening process reduces the precision of the '
             'algorithm, and induces undesirable edges effects. Default=1 (no downsampling).',
        required=False,
        default=1)
    optional.add_argument(
        "-xy-size",
        metavar=Metavar.float,
        type=float,
        help='Size of the output FOV in the RL/AP plane, in mm. The resolution of the destination '
             'image is the same as that of the source image (-i). Default: 35.',
        required=False,
        default=35.0)
    optional.add_argument(
        "-o",
        metavar=Metavar.file,
        help='Straightened file. By default, the suffix "_straight" will be added to the input file name.',
        required=False,
        default='')
    optional.add_argument(
        "-ofolder",
        metavar=Metavar.folder,
        help="Output folder (all outputs will go there).",
        action=ActionCreateFolder,
        required=False,
        default='.')
    optional.add_argument(
        '-centerline-algo',
        help='Algorithm for centerline fitting. Default: nurbs.',
        choices=('bspline', 'linear', 'nurbs'),
        default='nurbs')
    optional.add_argument(
        '-centerline-smooth',
        metavar=Metavar.int,
        type=int,
        help='Degree of smoothing for centerline fitting. Only use with -centerline-algo {bspline, linear}. Default: 10',
        default=10)

    optional.add_argument(
        "-param",
        metavar=Metavar.list,
        help="Parameters for spinal cord straightening. Separate arguments with \",\".\n"
             "  - precision: Float [1, inf) Precision factor of straightening, related to the number of slices. Increasing this parameter increases the precision along with increased computational time. Not taken into account with Hanning fitting method. Default=2\n"
             "  - threshold_distance: Float [0, inf) Threshold at which voxels are not considered into displacement. Increase this threshold if the image is blackout around the spinal cord too much. Default=10\n"
             "  - accuracy_results: {0, 1} Disable/Enable computation of accuracy results after straightening. Default=0\n"
             "  - template_orientation: {0, 1} Disable/Enable orientation of the straight image to be the same as the template. Default=0",
        required=False)

    optional.add_argument(
        "-x",
        help="Final interpolation. Default: spline.",
        choices=("nn", "linear", "spline"),
        default="spline")
    optional.add_argument(
        '-qc',
        metavar=Metavar.str,
        help='The path where the quality control generated content will be saved',
        default=None)
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the dataset the '
             'process was run on',
        default=None)
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help='If provided, this string will be mentioned in the QC report as the subject the '
             'process was run on',
        default=None)
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

    return parser


# MAIN
# ==========================================================================================
def main(argv=None):
    """
    Main function
    :param argv:
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    input_filename = arguments.i
    centerline_file = arguments.s

    sc_straight = SpinalCordStraightener(input_filename, centerline_file)

    if arguments.dest is not None:
        sc_straight.use_straight_reference = True
        sc_straight.centerline_reference_filename = str(arguments.dest)

    if arguments.ldisc_input is not None:
        if not sc_straight.use_straight_reference:
            printv('Warning: discs position are not taken into account if reference is not provided.')
        else:
            sc_straight.discs_input_filename = str(arguments.ldisc_input)
            sc_straight.precision = 4.0
    if arguments.ldisc_dest is not None:
        if not sc_straight.use_straight_reference:
            printv('Warning: discs position are not taken into account if reference is not provided.')
        else:
            sc_straight.discs_ref_filename = str(arguments.ldisc_dest)
            sc_straight.precision = 4.0

    # Handling optional arguments
    sc_straight.remove_temp_files = arguments.r
    sc_straight.interpolation_warp = arguments.x
    sc_straight.output_filename = arguments.o
    sc_straight.path_output = arguments.ofolder
    path_qc = arguments.qc
    sc_straight.verbose = verbose

    # if arguments.cpu_nb is not None:
    #     sc_straight.cpu_number = arguments.cpu-nb)
    if arguments.disable_straight2curved:
        sc_straight.straight2curved = False
    if arguments.disable_curved2straight:
        sc_straight.curved2straight = False

    if arguments.speed_factor:
        sc_straight.speed_factor = arguments.speed_factor

    if arguments.xy_size:
        sc_straight.xy_size = arguments.xy_size

    sc_straight.param_centerline = ParamCenterline(
        algo_fitting=arguments.centerline_algo,
        smooth=arguments.centerline_smooth)
    if arguments.param is not None:
        params_user = arguments.param
        # update registration parameters
        for param in params_user:
            param_split = param.split('=')
            if param_split[0] == 'precision':
                sc_straight.precision = float(param_split[1])
            if param_split[0] == 'threshold_distance':
                sc_straight.threshold_distance = float(param_split[1])
            if param_split[0] == 'accuracy_results':
                sc_straight.accuracy_results = int(param_split[1])
            if param_split[0] == 'template_orientation':
                sc_straight.template_orientation = int(param_split[1])

    fname_straight = sc_straight.straighten()

    printv("\nFinished! Elapsed time: {} s".format(sc_straight.elapsed_time), verbose)

    # Generate QC report
    if path_qc is not None:
        path_qc = os.path.abspath(path_qc)
        qc_dataset = arguments.qc_dataset
        qc_subject = arguments.qc_subject
        generate_qc(fname_straight, args=argv, path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, process=removesuffix(os.path.basename(__file__), ".py"))

    display_viewer_syntax([fname_straight], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
