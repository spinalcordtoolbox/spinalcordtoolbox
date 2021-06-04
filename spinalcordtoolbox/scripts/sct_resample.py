#!/usr/bin/env python
#########################################################################################
#
# Resample data using nibabel.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add possiblity to resample to destination image

import os
import sys

from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, init_sct, printv, set_global_loglevel
import spinalcordtoolbox.resampling


class Param:
    # The constructor
    def __init__(self):
        self.fname_data = ''
        self.fname_out = ''
        self.new_size = ''
        self.new_size_type = ''
        self.interpolation = 'linear'
        self.ref = None
        self.x_to_order = {'nn': 0, 'linear': 1, 'spline': 2}
        self.mode = 'reflect'  # How to fill the points outside the boundaries of the input, possible options: constant, nearest, reflect or wrap
        # constant put the superior edges to 0, wrap does something weird with the superior edges, nearest and reflect are fine
        self.file_suffix = '_resampled'  # output suffix
        self.verbose = 1


# initialize parameters
param = Param()


def get_parser():
    parser = SCTArgumentParser(
        description="Anisotropic resampling of 3D or 4D data."
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Image to segment. Can be 3D or 4D. (Cannot be 2D) Example: dwi.nii.gz"
    )

    resample_types = parser.add_argument_group(
        "\nTYPE OF THE NEW SIZE INPUT: with a factor of resampling, in mm or in number of voxels\n"
        "Please choose only one of the 3 options"
    )
    resample_types.add_argument(
        '-f',
        metavar=Metavar.str,
        help="R|Resampling factor in each dimensions (x,y,z). Separate with 'x'. Example: 0.5x0.5x1\n"
             "For 2x upsampling, set to 2. For 2x downsampling set to 0.5"
    )
    resample_types.add_argument(
        '-mm',
        metavar=Metavar.str,
        help="New resolution in mm. Separate dimension with 'x'. Example: 0.1x0.1x5"
    )
    resample_types.add_argument(
        '-vox',
        metavar=Metavar.str,
        help="Resampling size in number of voxels in each dimensions (x,y,z). Separate with 'x'."
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-ref',
        metavar=Metavar.file,
        help="Reference image to resample input image to. Uses world coordinates."
    )
    optional.add_argument(
        '-x',
        choices=['nn', 'linear', 'spline'],
        default='linear',
        help="Interpolation method."
    )
    optional.add_argument(
        '-header-rescale',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Rescale the affine matrix of an image to accommodate smaller/bigger cord sizes. O = no, 1 = yes"
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help="Output file name. Example: dwi_resampled.nii.gz"
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

    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)

    param.fname_data = arguments.i
    arg = 0
    if arguments.f is not None:
        param.new_size = arguments.f
        param.new_size_type = 'factor'
        arg += 1
    elif arguments.mm is not None:
        param.new_size = arguments.mm
        param.new_size_type = 'mm'
        arg += 1
    elif arguments.vox is not None:
        param.new_size = arguments.vox
        param.new_size_type = 'vox'
        arg += 1
    elif arguments.ref is not None:
        param.ref = arguments.ref
        arg += 1
    else:
        printv(parser.error('ERROR: you need to specify one of those three arguments : -f, -mm or -vox'))

    if arg > 1:
        printv(parser.error('ERROR: you need to specify ONLY one of those three arguments : -f, -mm or -vox'))

    if arguments.o is not None:
        param.fname_out = arguments.o
    if arguments.x is not None:
        if len(arguments.x) == 1:
            param.interpolation = int(arguments.x)
        else:
            param.interpolation = arguments.x

    if arguments.header_rescale:
        spinalcordtoolbox.resampling.rescale_affine(param.fname_data, param.fname_out, param.verbose, param.new_size, param.new_size_type)
    else:
        spinalcordtoolbox.resampling.resample_file(param.fname_data, param.fname_out, param.new_size, param.new_size_type,
                                               param.interpolation, param.verbose, fname_ref=param.ref)

if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])

