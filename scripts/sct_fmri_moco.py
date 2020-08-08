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
import argparse

from spinalcordtoolbox.moco import ParamMoco, moco_wrapper

import sct_utils as sct
from spinalcordtoolbox.utils import Metavar, SmartFormatter


def get_parser():
    # initialize parameters
    # TODO: create a class ParamFmriMoco which inheritates from ParamMoco
    param_default = ParamMoco(group_size=1, metric='MeanSquares', smooth='0')

    # parser initialisation
    parser = argparse.ArgumentParser(
        description=("Motion correction of fMRI data. Some robust features include:\n"
                     "    - group-wise (-g)\n"
                     "    - slice-wise regularized along z using polynomial function (-p)\n"
                     "      (For more info about the method, type: isct_antsSliceRegularizedRegistration)\n"
                     "    - masking (-m)\n"
                     "    - iterative averaging of target volume\n"
                     "\n"
                     "The outputs of the motion correction process are:\n"
                     "    - the motion-corrected fMRI volumes\n"
                     "    - the time average of the corrected fMRI volumes\n"
                     "    - a time-series with 1 voxel in the XY plane, for the X and Y motion direction (two separate "
                     "files), as required for FSL analysis.\n"
                     "    - a TSV file with the slice-wise average of the motion correction for XY (one file), that "
                     "can be used for Quality Control.\n"),
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py")
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
        metavar=Metavar.str,
        help=(f"R|Advanced parameters. Assign value with \"=\"; Separate arguments with \",\"\n"
              f"    -poly [int]: Degree of polynomial function used for regularization along Z. For no regularization "
              f"set to 0. Default={param_default.poly}.\n"
              f"    -smooth [mm]: Smoothing kernel. Default={param_default.smooth}.\n"
              f"    -iter [int]: Number of iterations. Default={param_default.iter}.\n"
              f"    -metric {{MI, MeanSquares, CC}}: Metric used for registration. Default={param_default.metric}.\n"
              f"    -gradStep [float]: Searching step used by registration algorithm. The higher the more deformation "
              f"allowed. Default={param_default.gradStep}.\n"
              f"    -sampling [None or 0-1]: Sampling rate used for registration metric. "
              f"Default={param_default.sampling}.\n"
              f"    -numTarget [int]: Target volume or group (starting with 0). Default={param_default.num_target}.\n"
              f"    -iterAvg [int]: Iterative averaging: Target volume is a weighted average of the "
              f"previously-registered volumes. Default={param_default.iterAvg}.\n")
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.folder,
        default='./',
        help="Output path."
    )
    optional.add_argument(
        '-x',
        metavar=Metavar.str,
        choices=['nn', 'linear', 'spline'],
        default='linear',
        help="Final interpolation."
    )
    optional.add_argument(
        '-r',
        metavar=Metavar.int,
        choices=['0', '1'],
        default='1',
        help="Remove temporary files. O = no, 1 = yes"
    )
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        choices=['0', '1', '2'],
        default='1',
        help="Verbose: 0 = nothing, 1 = basic, 2 = extended."
    )

    return parser


def main():

    # initialization
    param = ParamMoco(group_size=1, metric='MeanSquares', smooth='0')

    # Fetch user arguments
    parser = get_parser()
    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    param.fname_data = arguments.i
    param.path_out = arguments.ofolder
    param.remove_temp_files = int(arguments.r)
    param.interp = arguments.x
    if '-g' in arguments:
        param.group_size = arguments['-g']
    if '-m' in arguments:
        param.fname_mask = arguments['-m']
    if '-param' in arguments:
        param.update(arguments['-param'])
    param.verbose = int(arguments.v)

    # Update log level
    sct.init_sct(log_level=param.verbose, update=True)

    # run moco
    moco_wrapper(param)


if __name__ == "__main__":
    sct.init_sct()
    main()
