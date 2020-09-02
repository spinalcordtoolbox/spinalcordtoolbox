#!/usr/bin/env python

from __future__ import division, absolute_import

import os
import sys
import argparse

import numpy as np

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar, SmartFormatter, ActionCreateFolder
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline, _call_viewer_centerline
from spinalcordtoolbox.reports.qc import generate_qc


def get_parser():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description=(
            "This function extracts the spinal cord centerline. Three methods are available: OptiC (automatic), "
            "Viewer (manual), and Fitseg (applied on segmented image). These functions output (i) a NIFTI file with "
            "labels corresponding to the discrete centerline, and (ii) a csv file containing the float (more precise) "
            "coordinates of the centerline in the RPI orientation. \n"
            "\n"
            "Reference: C Gros, B De Leener, et al. Automatic spinal cord localization, robust to MRI contrast using "
            "global curve optimization (2017). doi.org/10.1016/j.media.2017.12.001"
        ),
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Input image. Example: ti.nii.gz"
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
        help=("Method used for extracting the centerline.\n"
              "  - optic: automatic spinal cord detection method\n"
              "  - viewer: manual selection a few points followed by interpolation\n"
              "  - fitseg: fit a regularized centerline on an already-existing cord segmentation. It will "
              "interpolate if slices are missing and extrapolate beyond the segmentation boundaries (i.e., every "
              "axial slice will exhibit a centerline pixel).")
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
        help=("File name (without extension) for the centerline output files. By default, output file will be the "
              "input with suffix '_centerline'. Example: 'centerline_optic'")
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
        "-v",
        choices=['0', '1'],
        default='1',
        help="Verbose. 1: display on, 0: display off (default)"
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


def run_main():
    sct.init_sct()
    parser = get_parser()
    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # Input filename
    fname_input_data = arguments.i
    fname_data = os.path.abspath(fname_input_data)

    # Method used
    method = arguments.method

    # Contrast type
    contrast_type = arguments.c
    if method == 'optic' and not contrast_type:
        # Contrast must be
        error = 'ERROR: -c is a mandatory argument when using Optic method.'
        sct.printv(error, type='error')
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
        path_data, file_data, ext_data = sct.extract_fname(fname_data)
        file_output = os.path.join(path_data, file_data + '_centerline')

    verbose = int(arguments.v)
    sct.init_sct(log_level=verbose, update=True)  # Update log level

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
        sct.printv("ERROR: The selected method is not available: {}. Please look at the help.".format(method), type='error')
        return


    # Extrapolate and regularize (or detect if optic) cord centerline
    im_centerline, arr_centerline, _, _ = get_centerline(im_labels,
                                                         param=param_centerline,
                                                         verbose=verbose)

    # save centerline as nifti (discrete) and csv (continuous) files
    im_centerline.save(file_output + '.nii.gz')
    np.savetxt(file_output + '.csv', arr_centerline.transpose(), delimiter=",")

    sct.display_viewer_syntax([fname_input_data, file_output+'.nii.gz'], colormaps=['gray', 'red'], opacities=['', '1'])
    
    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject

    # Generate QC report
    if path_qc is not None:
        generate_qc(fname_input_data, fname_seg=file_output+'.nii.gz', args=sys.argv[1:], path_qc=os.path.abspath(path_qc),
                    dataset=qc_dataset, subject=qc_subject, process='sct_get_centerline')
    sct.display_viewer_syntax([fname_input_data, file_output+'.nii.gz'], colormaps=['gray', 'red'], opacities=['', '0.7'])

if __name__ == '__main__':
    run_main()
