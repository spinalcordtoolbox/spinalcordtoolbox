#!/usr/bin/env python
# -*- coding: utf-8
#
# Function to analyze lesions or tumours by computing statistics on binary masks.
#

from __future__ import print_function, absolute_import, division

import os, math, sys
import argparse

from spinalcordtoolbox.process_seg import analyze_lesion
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.utils import Metavar, SmartFormatter, ActionCreateFolder

import sct_utils as sct
from sct_utils import extract_fname, printv, tmp_create


def get_parser():
    # Initialize the parser

    parser = argparse.ArgumentParser(
        description='R|Compute statistics on segmented lesions. The function assigns an ID value to each lesion (1, 2, '
                    '3, etc.) and then outputs morphometric measures for each lesion:\n'
                    '- volume [mm^3]\n'
                    '- length [mm]: length along the Superior-Inferior axis\n'
                    '- max_equivalent_diameter [mm]: maximum diameter of the lesion, when approximating\n'
                    '                                the lesion as a circle in the axial plane.\n\n'
                    'If the proportion of lesion in each region (e.g. WM and GM) does not sum up to 100%, it means '
                    'that the registered template does not fully cover the lesion. In that case you might want to '
                    'check the registration results.',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory_arguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory_arguments.add_argument(
        "-m",
        required=True,
        help='Binary mask of lesions (lesions are labeled as "1").',
        metavar=Metavar.file)
    mandatory_arguments.add_argument(
        "-s",
        required=True,
        help="Spinal cord centerline or segmentation file, which will be used to correct morphometric measures with "
             "cord angle with respect to slice. (e.g.'t2_seg.nii.gz')",
        metavar=Metavar.file)

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="show this help message and exit")
    optional.add_argument(
        "-i",
        help='Image from which to extract average values within lesions (e.g. "t2.nii.gz"). If provided, the function '
             'computes the mean and standard deviation values of this image within each lesion.',
        metavar=Metavar.file,
        default=None,
        required=False)
    optional.add_argument(
        "-t",
        help="Path to folder containing the atlas/template registered to the anatomical image. If provided, the "
             "function computes: (i) the distribution of each lesion depending on each vertebral level and on each"
             "region of the template (e.g. GM, WM, WM tracts) and (ii) the proportion of ROI (e.g. vertebral level, "
             "GM, WM) occupied by lesion.",
        metavar=Metavar.folder,
        default=None,
        required=False)
    optional.add_argument(
        "-ofolder",
        help='Output folder (e.g. "./")',
        metavar=Metavar.folder,
        action=ActionCreateFolder,
        default='./',
        required=False)
    optional.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        required=False,
        default=1,
        choices=(0, 1))
    optional.add_argument(
        "-v",
        type=int,
        help="Verbose: 0 = nothing, 1 = classic, 2 = expended",
        required=False,
        choices=(0, 1, 2),
        default=1)

    return parser


def check_binary(arr, fname, verbose=1):
    """
    Check if data is binary and not empty. If not, print warning or error message.

    :param arr: Input data, array.
    :param fname: Input filename.
    :param verbose: Verbose.
    :return:
    """
    if not np.array_equal(arr, arr.astype(bool)):
        if np.max(arr) == 0:
            printv('WARNING: Empty masked image: {}'.format(fname), verbose, 'warning')
        else:
            printv("ERROR input file %s is not binary file with 0 and 1 values".format(fname), 1, 'error')


# TODO: avoid duplication with: https://github.com/neuropoly/spinalcordtoolbox/blob/d1cdb30b231c83701d81a321ce62eb3980062a6f/spinalcordtoolbox/process_seg.py#L74
def get_angle_correction(im_seg):
    """
    Measure spinal cord angle with respect to slice.

    :param im_seg: Spinal cord segmentation image.
    :return: numpy array, angle for each I-S slice, np.nan when no segmentation.
    """
    nx, ny, nz, nt, px, py, pz, pt = im_seg.dim

    # fit centerline, smooth it and return the first derivative (in physical space)
    _, arr_ctl, arr_ctl_der, _ = get_centerline(im_seg, param=ParamCenterline(), verbose=1)
    x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = arr_ctl_der

    angle_correction = np.full_like(np.empty(nz), np.nan, dtype=np.double)

    # loop across x_centerline_deriv
    # (instead of [min_z_index, max_z_index], which could vary after interpolation)
    for iz in range(x_centerline_deriv.shape[0]):
        # normalize the tangent vector to the centerline (i.e. its derivative)
        vect = np.array([x_centerline_deriv[iz] * px, y_centerline_deriv[iz] * py, pz])
        norm = np.linalg.norm(vect)
        tangent_vect = vect / norm

        # compute the angle between the normal vector of the plane and the vector z
        angle = np.arccos(np.vdot(tangent_vect, np.array([0, 0, 1])))
        angle_correction[iz] = math.degrees(angle)

    return angle_correction


def main(args=None):
    """
    Main function
    :param args:
    :return:
    """
    # get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    parser = get_parser()
    arguments = parser.parse_args(args=args)

    # Input files
    fname_mask = arguments.m  # Mandatory
    fname_sc = arguments.s  # Mandatory
    fname_ref = arguments.i  # Optional

    # Path to registered template
    path_template = arguments.t

    # Output folder
    path_results = arguments.ofolder

    # Verbosity
    verbose = arguments.v
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # Run analysis
    analyze_lesion(fname_mask=fname_mask,
                   fname_voi=fname_sc,
                   fname_ref=fname_ref,
                   path_template=path_template,
                   path_ofolder=path_results,
                   verbose=verbose)

    # TODO charley: to refactor
    """
    # Remove temp folder
    if arguments.r is not None:
        rm_tmp = bool(arguments.r)
    else:
        rm_tmp = True
    
    # remove tmp_dir
    if rm_tmp:
        sct.rmtree(lesion_obj.tmp_dir)

    printv('\nDone! To view the labeled lesion file (one value per lesion), type:', verbose)
    if fname_ref is not None:
        printv('fsleyes ' + fname_mask + ' ' + os.path.join(path_results, lesion_obj.fname_label) + ' -cm red-yellow -a 70.0 & \n', verbose, 'info')
    else:
        printv('fsleyes ' + os.path.join(path_results, lesion_obj.fname_label) + ' -cm red-yellow -a 70.0 & \n', verbose, 'info')
    """


if __name__ == "__main__":
    sct.init_sct()
    main()
