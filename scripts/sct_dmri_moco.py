#!/usr/bin/env python
#########################################################################################
#
# Motion correction of dMRI data.
#
# Inspired by Xu et al. Neuroimage 2013.
#
# Details of the algorithm:
# - grouping of DW data only (every n volumes, default n=5)
# - average all b0
# - average DWI data within each group
# - average DWI of all groups
# - moco on DWI groups
# - moco on b=0, using target volume: last b=0
# - moco on all dMRI data
# _ generating b=0 mean and DWI mean after motion correction
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-08-15
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: if -f, we only need two plots. Plot 1: X params with fitted spline, plot 2: Y param with fitted splines. Each plot will have all Z slices (with legend Z=0, Z=1, ...) and labels: y; translation (mm), xlabel: volume #. Plus add grid.


import sys
import os
from spinalcordtoolbox.moco import ParamMoco, moco_wrapper

import sct_utils as sct
import argparse
from spinalcordtoolbox.utils import Metavar, SmartFormatter, list_type


def get_parser():

    # initialize parameters
    param_default = ParamMoco(is_diffusion=True, group_size=3, metric='MI', smooth='1')

    # Initialize the parser
    parser = argparse.ArgumentParser(
        description=(
            '  Motion correction of dMRI data. Some of the features to improve robustness were proposed in Xu et al. '
            '(http://dx.doi.org/10.1016/j.neuroimage.2012.11.014) and include:\n'
            '    -group-wise (-g)\n'
            '    -slice-wise regularized along z using polynomial function (-param). For more info about the method, '
            'type: isct_antsSliceRegularizedRegistration\n'
            '    -masking (-m)\n'
            '    -iterative averaging of target volume\n'
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
        help="Diffusion data. Example: dmri.nii.gz"
    )
    mandatory.add_argument(
        '-bvec',
        metavar=Metavar.file,
        required=True,
        help='Bvecs file. Example: bvecs.txt'
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-bval',
        metavar=Metavar.file,
        help='Bvals file. Example: bvals.nii.gz',
    )
    optional.add_argument(
        '-bvalmin',
        type=float,
        metavar=Metavar.float,
        help='B-value threshold (in s/mm2) below which data is considered as b=0. Example: 50.0',
    )
    optional.add_argument(
        '-g',
        type=int,
        metavar=Metavar.int,
        help='Group nvols successive dMRI volumes for more robustness. Example: 2',
        default=param_default.group_size,
    )
    optional.add_argument(
        '-m',
        metavar=Metavar.file,
        help='Binary mask to limit voxels considered by the registration metric. Example: dmri_mask.nii.gz',
    )
    optional.add_argument(
        '-param',
        metavar=Metavar.list,
        type=list_type(',', str),
        help=f"R|Advanced parameters. Assign value with \"=\", and separate arguments with \",\".\n"
             f"    -poly [int]: Degree of polynomial function used for regularization along Z. For no regularization set to "
             f"0. Default={param_default.poly}.\n"
             f"    -smooth [mm]: Smoothing kernel. Default={param_default.smooth}.\n"
             f"    -metric {{MI, MeanSquares, CC}}: Metric used for registration. Default={param_default.metric}.\n"
             f"    -gradStep [float]: Searching step used by registration algorithm. The higher the more deformation "
             f"allowed. Default={param_default.gradStep}.\n"
             f"    -sample [None or 0-1]: Sampling rate used for registration metric. Default={param_default.sampling}.\n"
    )
    optional.add_argument(
        '-x',
        metavar=Metavar.str,
        choices=['nn', 'linear', 'spline'],
        default=param_default.interp,
        help="Final interpolation."
    )
    optional.add_argument(
        '-ofolder',
        metavar=Metavar.folder,
        default=param_default.path_out,
        help="Output folder. Example: dmri_moco_results/"
    )
    # TODO: Convert -r to flag using action=store_true
    optional.add_argument(
        "-r",
        metavar=Metavar.str,
        choices=('0', '1'),
        default='1',
        help="Remove temporary files. 0 = no, 1 = yes"
    )
    optional.add_argument(
        "-v",
        metavar=Metavar.str,
        choices=('0', '1', '2'),
        default='1',
        help="Verbose: 0 = nothing, 1 = classic, 2 = expanded",
    )
    return parser


def main():

    # initialization
    param = ParamMoco(is_diffusion=True, group_size=3, metric='MI', smooth='1')

    # Fetch user arguments
    parser = get_parser()
    arguments = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    param.fname_data = arguments.i
    param.fname_bvecs = os.path.abspath(arguments.bvec)
    if arguments.bval is not None:
        param.fname_bvals = os.path.abspath(arguments.bval)
    if arguments.bvalmin is not None:
        param.bval_min = arguments.bvalmin
    if arguments.g is not None:
        param.group_size = arguments.g
    if arguments.m is not None:
        param.fname_mask = arguments.m
    if arguments.param is not None:
        param.update(arguments.param)
    if arguments.x is not None:
        param.interp = arguments.x
    if arguments.ofolder is not None:
        param.path_out = arguments.ofolder
    if arguments.r is not None:
        param.remove_temp_files = arguments.r
    param.verbose = int(arguments.v)

    # Update log level
    sct.init_sct(log_level=param.verbose, update=True)

    # run moco
    moco_wrapper(param)


if __name__ == "__main__":
    sct.init_sct()
    main()
