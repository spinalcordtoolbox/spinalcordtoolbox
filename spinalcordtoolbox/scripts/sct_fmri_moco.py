#!/usr/bin/env python
#########################################################################################
#
# Motion correction of fMRI data.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
import os
from typing import Sequence

from spinalcordtoolbox.moco import ParamMoco, moco_wrapper
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax, list_type
from spinalcordtoolbox.reports.qc import generate_qc


def get_parser():
    # initialize parameters
    # TODO: create a class ParamFmriMoco which inheritates from ParamMoco
    param_default = ParamMoco(group_size=1, metric='MeanSquares', smooth='0')

    # parser initialisation
    parser = SCTArgumentParser(
        description="Motion correction of fMRI data. Some robust features include:\n"
                    "  - group-wise (-g)\n"
                    "  - slice-wise regularized along z using polynomial function (-p)\n"
                    "    (For more info about the method, type: isct_antsSliceRegularizedRegistration)\n"
                    "  - masking (-m)\n"
                    "  - iterative averaging of target volume\n"
                    "\n"
                    "The outputs of the motion correction process are:\n"
                    "  - the motion-corrected fMRI volumes\n"
                    "  - the time average of the corrected fMRI volumes\n"
                    "  - a time-series with 1 voxel in the XY plane, for the X and Y motion direction (two separate "
                    "files), as required for FSL analysis.\n"
                    "  - a TSV file with the slice-wise average of the motion correction for XY (one file), that "
                    "can be used for Quality Control.\n"
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Input data (4D). Example: fmri.nii.gz"
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-g',
        metavar=Metavar.int,
        type=int,
        help="Group nvols successive fMRI volumes for more robustness."
    )
    optional.add_argument(
        '-m',
        metavar=Metavar.file,
        help="Binary mask to limit voxels considered by the registration metric."
    )
    optional.add_argument(
        '-param',
        metavar=Metavar.list,
        type=list_type(',', str),
        help=f"Advanced parameters. Assign value with \"=\"; Separate arguments with \",\".\n"
             f"  - poly [int]: Degree of polynomial function used for regularization along Z. For no regularization "
             f"set to 0. Default={param_default.poly}.\n"
             f"  - smooth [mm]: Smoothing kernel. Default={param_default.smooth}.\n"
             f"  - iter [int]: Number of iterations. Default={param_default.iter}.\n"
             f"  - metric {{MI, MeanSquares, CC}}: Metric used for registration. Default={param_default.metric}.\n"
             f"  - gradStep [float]: Searching step used by registration algorithm. The higher the more deformation "
             f"allowed. Default={param_default.gradStep}.\n"
             f"  - sampling [None or 0-1]: Sampling rate used for registration metric. "
             f"Default={param_default.sampling}.\n"
             f"  - numTarget [int]: Target volume or group (starting with 0). Default={param_default.num_target}.\n"
             f"  - iterAvg [int]: Iterative averaging: Target volume is a weighted average of the "
             f"previously-registered volumes. Default={param_default.iterAvg}.\n"
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default='.',
        help="Output path."
    )
    optional.add_argument(
        '-x',
        choices=['nn', 'linear', 'spline'],
        default='linear',
        help="Final interpolation."
    )
    optional.add_argument(
        '-r',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=1,
        help="Remove temporary files. 0 = no, 1 = yes"
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
        '-qc',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved. (Note: "
             "Both '-qc' and '-qc-seg' are required in order to generate a QC report.)"
    )
    optional.add_argument(
        '-qc-seg',
        metavar=Metavar.file,
        help="Segmentation of spinal cord to improve cropping in qc report. (Note: "
             "Both '-qc' and '-qc-seg' are required in order to generate a QC report.)"
    )
    optional.add_argument(
        '-qc-fps',
        metavar=Metavar.float,
        type=float,
        default=3,
        help="This float number is the number of frames per second for the output gif images."
    )
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    # initialization
    param = ParamMoco(group_size=1, metric='MeanSquares', smooth='0')

    # Fetch user arguments
    param.fname_data = arguments.i
    param.path_out = arguments.ofolder
    param.remove_temp_files = arguments.r
    param.interp = arguments.x
    if arguments.g is not None:
        param.group_size = arguments.g
    if arguments.m is not None:
        param.fname_mask = arguments.m
    if arguments.param is not None:
        param.update(arguments.param)

    path_qc = arguments.qc
    qc_fps = arguments.qc_fps
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject
    qc_seg = arguments.qc_seg

    mutually_inclusive_args = (path_qc, qc_seg)
    is_qc_none, is_seg_none = [arg is None for arg in mutually_inclusive_args]
    if not (is_qc_none == is_seg_none):
        parser.error("Both '-qc' and '-qc-seg' are required in order to generate a QC report.")

    # run moco
    fname_output_image = moco_wrapper(param)

    set_loglevel(verbose)  # moco_wrapper changes verbose to 0, see issue #3341

    # QC report
    if path_qc is not None:
        generate_qc(fname_in1=fname_output_image, fname_in2=param.fname_data, fname_seg=qc_seg,
                    args=argv, path_qc=os.path.abspath(path_qc), fps=qc_fps, dataset=qc_dataset,
                    subject=qc_subject, process='sct_fmri_moco')

    display_viewer_syntax([fname_output_image, param.fname_data], mode='ortho,ortho', verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
