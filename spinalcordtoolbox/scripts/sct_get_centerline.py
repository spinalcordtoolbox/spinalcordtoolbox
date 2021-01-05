#!/usr/bin/env python

import os
import sys

import numpy as np

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline, _call_viewer_centerline
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_global_loglevel
from spinalcordtoolbox.utils.fs import extract_fname


def get_parser():
    parser = SCTArgumentParser(
        description=(
            "This function extracts the spinal cord centerline. Three methods are available: 'optic' (automatic), "
            "'viewer' (manual), and 'fitseg' (applied on segmented image). These functions output (i) a NIFTI file "
            "with labels corresponding to the discrete centerline, and (ii) a csv file containing the float (more "
            "precise) coordinates of the centerline in the RPI orientation. \n"
            "\n"
            "Reference: C Gros, B De Leener, et al. Automatic spinal cord localization, robust to MRI contrast using "
            "global curve optimization (2017). doi.org/10.1016/j.media.2017.12.001"
        )
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Input image. Example: t1.nii.gz"
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        "-c",
        choices=['t1', 't2', 't2s', 'dwi'],
        help="Type of image contrast. Only with method=optic."
    )
    optional.add_argument(
        "-method",
        choices=['optic', 'viewer', 'fitseg'],
        default='optic',
        help="R|Method used for extracting the centerline.\n"
             "  - optic: automatic spinal cord detection method\n"
             "  - viewer: manual selection a few points followed by interpolation\n"
             "  - fitseg: fit a regularized centerline on an already-existing cord segmentation. It will "
             "interpolate if slices are missing and extrapolate beyond the segmentation boundaries (i.e., every "
             "axial slice will exhibit a centerline pixel)."
    )
    optional.add_argument(
        "-centerline-algo",
        choices=['polyfit', 'bspline', 'linear', 'nurbs'],
        default='bspline',
        help="Algorithm for centerline fitting. Only relevant with -method fitseg"
    )
    optional.add_argument(
        "-centerline-smooth",
        metavar=Metavar.int,
        type=int,
        default=30,
        help="Degree of smoothing for centerline fitting. Only for -centerline-algo {bspline, linear}."
    )
    optional.add_argument(
        "-o",
        metavar=Metavar.file,
        help="File name (without extension) for the centerline output files. By default, output file will be the "
             "input with suffix '_centerline'. Example: 'centerline_optic'"
    )
    optional.add_argument(
        "-gap",
        metavar=Metavar.float,
        type=float,
        default=20.0,
        help="Gap in mm between manually selected points. Only with method=viewer."
    )
    optional.add_argument(
        "-igt",
        metavar=Metavar.file,
        help="File name of ground-truth centerline or segmentation (binary nifti)."
    )
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
    )
    optional.add_argument(
        "-qc",
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved."
    )
    optional.add_argument(
        "-qc-dataset",
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    optional.add_argument(
        "-qc-subject",
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )
    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    # Input filename
    fname_input_data = arguments.i
    fname_data = os.path.abspath(fname_input_data)

    # Method used
    method = arguments.method

    # Contrast type
    contrast_type = arguments.c
    if method == 'optic' and not contrast_type:
        # Contrast must be
        error = "ERROR: -c is a mandatory argument when using 'optic' method."
        printv(error, type='error')
        return

    # Gap between slices
    interslice_gap = arguments.gap

    param_centerline = ParamCenterline(
        algo_fitting=arguments.centerline_algo,
        smooth=arguments.centerline_smooth,
        minmax=True)

    # Output folder
    if arguments.o is not None:
        file_output = arguments.o
    else:
        path_data, file_data, ext_data = extract_fname(fname_data)
        file_output = os.path.join(path_data, file_data + '_centerline')

    if method == 'viewer':
        # Manual labeling of cord centerline
        im_labels = _call_viewer_centerline(Image(fname_data), interslice_gap=interslice_gap)
    elif method == 'fitseg':
        im_labels = Image(fname_data)
    elif method == 'optic':
        # Automatic detection of cord centerline
        im_labels = Image(fname_data)
        param_centerline.algo_fitting = 'optic'
        param_centerline.contrast = contrast_type
    else:
        printv("ERROR: The selected method is not available: {}. Please look at the help.".format(method), type='error')
        return

    # Extrapolate and regularize (or detect if optic) cord centerline
    im_centerline, arr_centerline, _, _ = get_centerline(im_labels,
                                                         param=param_centerline,
                                                         verbose=verbose)

    # save centerline as nifti (discrete) and csv (continuous) files
    im_centerline.save(file_output + '.nii.gz')
    np.savetxt(file_output + '.csv', arr_centerline.transpose(), delimiter=",")

    display_viewer_syntax([fname_input_data, file_output + '.nii.gz'], colormaps=['gray', 'red'], opacities=['', '1'])

    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject

    # Generate QC report
    if path_qc is not None:
        generate_qc(fname_input_data, fname_seg=file_output + '.nii.gz', args=sys.argv[1:], path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, process='sct_get_centerline')
    display_viewer_syntax([fname_input_data, file_output + '.nii.gz'], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

