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

from __future__ import absolute_import, division

import sys, os, functools, argparse

import sct_utils as sct
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import Metavar, SmartFormatter

class Param:
    # The constructor
    def __init__(self):
        self.fname_warp_final = 'warp_final.nii.gz'


def main(args=None):
    """
    Main function
    :param args:
    :return:
    """
    # get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    else:
        # flatten the list of input arguments because -w and -winv carry a nested list
        lst = []
        for line in args:
            lst.append(line) if isinstance(line, str) else lst.extend(line)
        args = lst
    parser = get_parser()
    arguments = parser.parse_args(args=args)

    # Initialization
    fname_warp_final = ''  # concatenated transformations
    fname_dest = arguments.d
    fname_warp_list = arguments.w
    warpinv_filename = arguments.winv

    if arguments.o is not None:
        fname_warp_final = arguments.o
    verbose = arguments.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # Parse list of warping fields
    sct.printv('\nParse list of warping fields...', verbose)
    use_inverse = []
    fname_warp_list_invert = []
    # list_warp = list_warp.replace(' ', '')  # remove spaces
    # list_warp = list_warp.split(",")  # parse with comma
    for idx_warp, path_warp in enumerate(fname_warp_list):
        # Check if this transformation should be inverted
        if path_warp in warpinv_filename:
            use_inverse.append('-i')
            # list_warp[idx_warp] = path_warp[1:]  # remove '-'
            fname_warp_list_invert += [[use_inverse[idx_warp], fname_warp_list[idx_warp]]]
        else:
            use_inverse.append('')
            fname_warp_list_invert += [[path_warp]]
        path_warp = fname_warp_list[idx_warp]
        if path_warp.endswith((".nii", ".nii.gz")) \
                and Image(fname_warp_list[idx_warp]).header.get_intent()[0] != 'vector':
            raise ValueError("Displacement field in {} is invalid: should be encoded" \
                             " in a 5D file with vector intent code" \
                             " (see https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h" \
                             .format(path_warp))
    # need to check if last warping field is an affine transfo
    isLastAffine = False
    path_fname, file_fname, ext_fname = sct.extract_fname(fname_warp_list_invert[-1][-1])
    if ext_fname in ['.txt', '.mat']:
        isLastAffine = True

    # check if destination file is 3d
    if not sct.check_dim(fname_dest, dim_lst=[3]):
        sct.printv('ERROR: Destination data must be 3d')

    # Here we take the inverse of the warp list, because sct_WarpImageMultiTransform concatenates in the reverse order
    fname_warp_list_invert.reverse()
    fname_warp_list_invert = functools.reduce(lambda x, y: x + y, fname_warp_list_invert)

    # Check file existence
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_dest, verbose)
    for i in range(len(fname_warp_list)):
        sct.check_file_exist(fname_warp_list[i], verbose)

    # Get output folder and file name
    if fname_warp_final == '':
        path_out, file_out, ext_out = sct.extract_fname(param.fname_warp_final)
    else:
        path_out, file_out, ext_out = sct.extract_fname(fname_warp_final)

    # Check dimension of destination data (cf. issue #1419, #1429)
    im_dest = Image(fname_dest)
    if im_dest.dim[2] == 1:
        dimensionality = '2'
    else:
        dimensionality = '3'

    cmd = ['isct_ComposeMultiTransform', dimensionality, 'warp_final' + ext_out, '-R', fname_dest] + fname_warp_list_invert
    status, output = sct.run(cmd, verbose=verbose, is_sct_binary=True)

    # check if output was generated
    if not os.path.isfile('warp_final' + ext_out):
        sct.printv('ERROR: Warping field was not generated.\n' + output, 1, 'error')

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file('warp_final' + ext_out, os.path.join(path_out, file_out + ext_out))


# ==========================================================================================
def get_parser():
    # Initialize the parser

    parser = argparse.ArgumentParser(
        description='Concatenate transformations. This function is a wrapper for isct_ComposeMultiTransform (ANTs). '
                    'The order of input warping fields is important. For example, if you want to concatenate: '
                    'A->B and B->C to yield A->C, then you have to input warping fields in this order: A->B B->C.',
        formatter_class=SmartFormatter,
        add_help=None,
        prog=os.path.basename(__file__).strip(".py"))

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
        metavar=Metavar.str,
        required = False)
    optional.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = nothing, 1 = classic, 2 = expended",
        required=False,
        choices=(0, 1, 2),
        default = 1)

    return parser


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    param = Param()
    # call main function
    main()
