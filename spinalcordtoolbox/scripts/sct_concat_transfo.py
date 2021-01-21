#!/usr/bin/env python
#########################################################################################
#
# Concatenate transformations. This function is a wrapper for isct_ComposeMultiTransform
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-07-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: also enable to concatenate reversed transfo

import sys
import os
import functools
import argparse

from spinalcordtoolbox.image import Image, check_dim, generate_output_file
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, SmartFormatter
from spinalcordtoolbox.utils.sys import init_sct, printv, run_proc, set_global_loglevel
from spinalcordtoolbox.utils.fs import tmp_create, extract_fname, check_file_exist


class Param:
    # The constructor
    def __init__(self):
        self.fname_warp_final = 'warp_final.nii.gz'


def main(argv=None):
    """
    Main function
    :param argv:
    :return:
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_global_loglevel(verbose=verbose)
    param = Param()

    # Initialization
    fname_warp_final = ''  # concatenated transformations
    if arguments.o is not None:
        fname_warp_final = arguments.o
    fname_dest = arguments.d
    fname_warp_list = arguments.w
    warpinv_filename = arguments.winv

    # Parse list of warping fields
    printv('\nParse list of warping fields...', verbose)
    use_inverse = []
    fname_warp_list_invert = []
    for idx_warp, path_warp in enumerate(fname_warp_list):
        # Check if this transformation should be inverted
        if path_warp in warpinv_filename:
            use_inverse.append('-i')
            fname_warp_list_invert += [[use_inverse[idx_warp], fname_warp_list[idx_warp]]]
        else:
            use_inverse.append('')
            fname_warp_list_invert += [[path_warp]]
        path_warp = fname_warp_list[idx_warp]
        if path_warp.endswith((".nii", ".nii.gz")) \
                and Image(fname_warp_list[idx_warp]).header.get_intent()[0] != 'vector':
            raise ValueError("Displacement field in {} is invalid: should be encoded"
                             " in a 5D file with vector intent code"
                             " (see https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h"
                             .format(path_warp))

    # check if destination file is 3d
    check_dim(fname_dest, dim_lst=[3])

    # Here we take the inverse of the warp list, because sct_WarpImageMultiTransform concatenates in the reverse order
    fname_warp_list_invert.reverse()
    fname_warp_list_invert = functools.reduce(lambda x, y: x + y, fname_warp_list_invert)

    # Check file existence
    printv('\nCheck file existence...', verbose)
    check_file_exist(fname_dest, verbose)
    for i in range(len(fname_warp_list)):
        check_file_exist(fname_warp_list[i], verbose)

    # Get output folder and file name
    if fname_warp_final == '':
        path_out, file_out, ext_out = extract_fname(param.fname_warp_final)
    else:
        path_out, file_out, ext_out = extract_fname(fname_warp_final)

    # Check dimension of destination data (cf. issue #1419, #1429)
    im_dest = Image(fname_dest)
    if im_dest.dim[2] == 1:
        dimensionality = '2'
    else:
        dimensionality = '3'

    cmd = ['isct_ComposeMultiTransform', dimensionality, 'warp_final' + ext_out, '-R', fname_dest] + fname_warp_list_invert
    _, output = run_proc(cmd, verbose=verbose, is_sct_binary=True)

    # check if output was generated
    if not os.path.isfile('warp_final' + ext_out):
        raise ValueError(f"Warping field was not generated! {output}")

    # Generate output files
    printv('\nGenerate output files...', verbose)
    generate_output_file('warp_final' + ext_out, os.path.join(path_out, file_out + ext_out))


# ==========================================================================================
def get_parser():
    # Initialize the parser

    parser = SCTArgumentParser(
        description='Concatenate transformations. This function is a wrapper for isct_ComposeMultiTransform (ANTs). '
                    'The order of input warping fields is important. For example, if you want to concatenate: '
                    'A->B and B->C to yield A->C, then you have to input warping fields in this order: A->B B->C.',
    )

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        "-d",
        required=True,
        help='Destination image. (e.g. "mt.nii.gz")',
        metavar=Metavar.file,
    )
    mandatoryArguments.add_argument(
        "-w",
        required=True,
        help='Transformation(s), which can be warping fields (nifti image) or affine transformation matrix (text '
             'file). Separate with space. Example: warp1.nii.gz warp2.nii.gz',
        nargs='+',
        metavar=Metavar.file)

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-winv",
        help='Affine transformation(s) listed in flag -w which should be inverted before being used. Note that this '
             'only concerns affine transformation (not warping fields). If you would like to use an inverse warping'
             'field, then directly input the inverse warping field in flag -w.',
        nargs='*',
        metavar=Metavar.file,
        default=[])
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit")
    optional.add_argument(
        "-o",
        help='Name of output warping field (e.g. "warp_template2mt.nii.gz")',
        metavar=Metavar.str)
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


# =======================================================================================================================
# Start program
# =======================================================================================================================
if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
