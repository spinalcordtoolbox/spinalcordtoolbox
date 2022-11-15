#!/usr/bin/env python
##############################################################################
#
# Detect the compression level
# The script requires spinal cord segmentation and file with disc labels. If
# not provided, the script performs the spinal cord segmentation and disc
# identification automatically on the input image (not implemented yet).
# For details, see Horakova et al., 2022 (https://pubmed.ncbi.nlm.nih.gov/35371944/)
#
# ----------------------------------------------------------------------------
# Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Jan Valosek
#
# About the license: see the file LICENSE.TXT
##############################################################################

import sys
import math
from typing import Sequence

import numpy as np

from spinalcordtoolbox.process_seg import compute_shape
from spinalcordtoolbox.aggregate_slicewise import aggregate_per_slice_or_level, func_wa, func_std, merge_dict
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.template import get_slices_from_vertebral_levels
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.fs import get_absolute_path
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel, printv


# constants for the normalized model, adapted based on Horakova et al., 2022
# TODO - consider to use Param object instead of global variables
#  (https://github.com/spinalcordtoolbox/spinalcordtoolbox/wiki/Programming%3A-CLI-script-structure#-param--param)
CONSTANT = 56.066
CR_COEFFICIENT = 14.961
CSA_COEFFICIENT = 7.533
SOLIDITY_COEFFICIENT = 38.550
TORSION_COEFFICIENT = 2.118
CUT_OFF = 0.451     # cut-off to determine the presence of the compression
E = math.e      # Euler's Number

# Normative values for healthy subjects, see table 2 in Horakova et al., 2022
# Due to naturally different anatomy between vertebral levels (e.g., C3/C4 has different CSA than C6/C7), quantitative
# metrics must be normalized.
# The convention for disc labels is the following: value=4 -> disc C3/C4, value=5 -> disc C4/C5 etc
normative_values = {
    'CR': {4: 58.7, 5: 55.1, 6: 53.4, 7: 54.8},
    'CSA': {4: 71.7, 5: 75.4, 6: 71.4, 7: 62.3},
    'Solidity': {4: 96.8, 5: 96.4, 6: 96.3, 7: 96.4},
    'Torsion': {4: 0.81, 5: 0.75, 6: 0.88, 7: 1.18}
    }

# Currently, the model works only for discs C3/C4 (4) to C6/7 (7)
supported_discs = [4, 5, 6, 7]


def get_parser():
    """
    :return: Returns the parser with the command line documentation contained in it.
    """
    # Initialize the parser
    parser = SCTArgumentParser(
        description=(
            "Detect the compression level. "
            "For details, see Horakova et al., 2022 (https://pubmed.ncbi.nlm.nih.gov/35371944/). "
            "The script requires spinal cord segmentation and file with disc labels. If not provided, the script "
            "performs the spinal cord segmentation and disc identification automatically on the input image."
        )
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-i',
        metavar=Metavar.file,
        help="Input image. Example: t2star.nii.gz"
    )
    optional.add_argument(
        '-s',
        metavar=Metavar.file,
        help="Mask to compute morphometrics from (e.g., spinal cord segmentation). "
             "Example: t2star_seg.nii.gz. If not provided, segmentation will be done on the input image (flag -i)."
    )
    optional.add_argument(
        '-discfile',
        metavar=Metavar.file,
        help="File with disc labels. The convention for disc labels is the following: "
             "value=3 -> disc C2/C3, value=4 -> disc C3/C4, etc."
             "Such label file can be manually created using: sct_label_utils -i IMAGE_REF -create-viewer 4:7 "
             "or obtained automatically using sct_label_vertebrae function (the file with labeled_discs.nii.gz suffix)."
    )
    optional.add_argument(
        '-torsion-slices',
        metavar=Metavar.int,
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Number of slices above and below the given slice to compute the torsion for. For details, see eq 1-3 in "
             "https://pubmed.ncbi.nlm.nih.gov/35371944/."
             "Options: 1, 2, 3"
    )
    optional.add_argument(
        '-centerline-algo',
        choices=['polyfit', 'bspline', 'linear', 'nurbs'],
        default='bspline',
        help="Algorithm for centerline fitting used within spinalcordtoolbox.process_seg.compute_shape. "
             "Only relevant with -angle-corr 1."
    )
    optional.add_argument(
        '-centerline-smooth',
        metavar=Metavar.int,
        type=int,
        default=30,
        help="Degree of smoothing for centerline fitting within spinalcordtoolbox.process_seg.compute_shape. "
             "Only use with -centerline-algo {bspline, linear}."
    )
    optional.add_argument(
        '-angle-corr',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=1,
        help="Angle correction for computing morphometric measures used within "
             "spinalcordtoolbox.process_seg.compute_shape. When angle correction is used, the cord within "
             "the slice is stretched/expanded by a factor corresponding to the cosine of the angle between the "
             "centerline and the axial plane. If the cord is already quasi-orthogonal to the slab, you can set "
             "-angle-corr to 0."
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


def compute_compression_probability(cr, csa, solidity, torsion, disc):
    """
    # Detect the presence of the compression based on the normalized model.
    # Based on eq 4 from Horakova et al., 2022 (https://pubmed.ncbi.nlm.nih.gov/35371944/)
    :param cr: compression ratio (CR) in %
    :param csa: cross-sectional area (CSA) in mm2
    :param solidity: solidity in %
    :param torsion: torsion in degrees
    :param disc: intervertebral disc level (e.g., 4 - corresponding to disc C3/C4)
    :return: probability
    """
    model = CONSTANT - \
            CR_COEFFICIENT * (cr / normative_values['CR'][disc]) - \
            CSA_COEFFICIENT * (csa / normative_values['CSA'][disc]) - \
            SOLIDITY_COEFFICIENT * (solidity / normative_values['Solidity'][disc]) + \
            TORSION_COEFFICIENT * (torsion / normative_values['Torsion'][disc])
    probability = (E ** model) / (1 + E ** model)
    probability = probability

    return probability


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    # Initialization
    group_funcs = (('MEAN', func_wa), ('STD', func_std))

    # Fetch input arguments
    if arguments.i is None and arguments.s is None:
        parser.error("Either the option -i or option -s is required.")
    if arguments.i is not None:
        fname_in = get_absolute_path(arguments.i)
    if arguments.s is not None:
        fname_seg = get_absolute_path(arguments.s)
    if arguments.discfile is not None:
        fname_disc = get_absolute_path(arguments.discfile)
    torsion_slices = arguments.torsion_slices
    angle_correction = bool(arguments.angle_corr)
    param_centerline = ParamCenterline(
        algo_fitting=arguments.centerline_algo,
        smooth=arguments.centerline_smooth,
        minmax=True)

    # No SC segmentation provided, do it now
    if arguments.s is None:
        printv('Automatic segmentation is not implemented yet.', 1, 'error')
        sys.exit(1)
        # TODO - perform segmentation on fname_in within this script.

    # No disc file provided, do it now
    if arguments.discfile is None:
        printv('Automatic labeling is not implemented yet.', 1, 'error')
        sys.exit(1)
        # TODO - perform disc identification (vertebral labeling) within this script.

    # Compute morphometric metrics
    metrics, fit_results = compute_shape(fname_seg,
                                         angle_correction=angle_correction,
                                         param_centerline=param_centerline,
                                         verbose=verbose)

    # Compute the average and standard deviation across slices
    metrics_agg = {}
    for key in ['area', 'diameter_AP', 'diameter_RL', 'solidity', 'orientation']:
        # Note: we do not need to calculate all the metrics, we need just:
        #   - area (will be CSA)
        #   - diameter_AP and diameter_RL (used to calculate compression ratio)
        #   - solidity
        #   - orientation (used to calculate torsion)
        # Note: we have to calculate metrics across all slices (perslice) to be able to compute orientation
        metrics_agg[key] = aggregate_per_slice_or_level(metrics[key],
                                                        perslice=True,
                                                        perlevel=False, fname_vert_level=fname_disc,
                                                        group_funcs=group_funcs
                                                        )

    metrics_agg_merged = merge_dict(metrics_agg)

    # Compute compression ratio (CR) as 'diameter_AP' / 'diameter_RL'
    # TODO - compression ratio (CR) could be computed directly within the compute_shape function -> consider that
    for key in metrics_agg_merged.keys():           # Loop across slices
        # Ignore slices which have diameter_AP or diameter_RL equal to None (e.g., due to bad SC segmentation)
        if metrics_agg_merged[key]['MEAN(diameter_AP)'] is None or metrics_agg_merged[key]['MEAN(diameter_RL)'] is None:
            metrics_agg_merged[key]['CompressionRatio'] = None
        else:
            metrics_agg_merged[key]['CompressionRatio'] = metrics_agg_merged[key]['MEAN(diameter_AP)'] / \
                                                          metrics_agg_merged[key]['MEAN(diameter_RL)']

    # Compute torsion as the average of absolute differences in orientation between the given slice and x slice(s)
    # above and below. For details see eq 1-3 in https://pubmed.ncbi.nlm.nih.gov/35371944/
    # TODO - torsion could be computed directly within the compute_shape function -> consider that
    # Since the torsion is computed from slices above and below, it cannot be computed for the x first and last x slices
    # --> x first and x last slices will be excluded from the torsion computation
    # For example, if torsion_slices == 3, the first three and last three slices will have torsion = None
    slices = list(metrics_agg_merged.keys())[torsion_slices:-torsion_slices]

    for key in metrics_agg_merged.keys():  # Loop across slices
        if key in slices:
            # Note: the key is a tuple (e.g. `1,`), not an int (e.g., 1), thus key[0] is used to convert tuple to int
            # and `,` is used to convert int back to tuple
            # TODO - the keys could be changed from tuple to int inside the compute_shape function -> consider that
            if torsion_slices == 3:
                metrics_agg_merged[key]['Torsion'] = 1/6 * (abs(metrics_agg_merged[key]['MEAN(orientation)'] -
                                                                metrics_agg_merged[key[0] - 1,]['MEAN(orientation)']) +
                                                            abs(metrics_agg_merged[key]['MEAN(orientation)'] -
                                                                metrics_agg_merged[key[0] + 1,]['MEAN(orientation)']) +
                                                            abs(metrics_agg_merged[key[0] - 1,]['MEAN(orientation)'] -
                                                                metrics_agg_merged[key[0] - 2,]['MEAN(orientation)']) +
                                                            abs(metrics_agg_merged[key[0] + 1,]['MEAN(orientation)'] -
                                                                metrics_agg_merged[key[0] + 2,]['MEAN(orientation)']) +
                                                            abs(metrics_agg_merged[key[0] - 2,]['MEAN(orientation)'] -
                                                                metrics_agg_merged[key[0] - 3,]['MEAN(orientation)']) +
                                                            abs(metrics_agg_merged[key[0] + 2,]['MEAN(orientation)'] -
                                                                metrics_agg_merged[key[0] + 3,]['MEAN(orientation)']))
                # TODO - implement also equations for torsion_slices == 1 and torsion_slices == 2
        else:
            metrics_agg_merged[key]['Torsion'] = None

    # Identify which intervertebral disc compression belongs to
    im_disc = Image(fname_disc).change_orientation('RPI')

    # Identify which discs are included in the im_disc (e.g., 3, 4, 5 etc)
    levels = np.trim_zeros(np.unique(im_disc.data))

    # Make sure levels are int
    levels = [int(level) for level in levels]
    # Find slices corresponding to the discs
    slicegroups = dict()
    for level in levels:
        slice = get_slices_from_vertebral_levels(im_disc, level)
        if isinstance(slice, list):
            # Note: [0] is used because get_slices_from_vertebral_levels returns list; in our case, it is just
            # one-item list
            slicegroups[level] = slice[0]

    # Initialize empty dict to save info about detected compression(s)
    compression_dict = dict()
    # Loop aross discs
    for disc, slice in slicegroups.items():
        if disc in supported_discs:
            # Get quantitative metrics for given disc based on its slice
            # Note: [slice,] is used to convert int to tuple
            cr = metrics_agg_merged[slice,]['CompressionRatio']*100
            csa = metrics_agg_merged[slice,]['MEAN(area)']
            solidity = metrics_agg_merged[slice,]['MEAN(solidity)']*100
            torsion = metrics_agg_merged[slice,]['Torsion']
            # Compute compression probability
            probability = compute_compression_probability(cr, csa, solidity, torsion, disc)
            # TODO - this is probably useful just for a debug --> print this with verbosity == 2
            print(f'Compression probability for disc {disc} (corresponding to slice {slice}) is {probability:.3f}.')
            if probability > CUT_OFF:
                compression_dict[disc] = slice

    print('\nCompression(s) was detected at:')
    for disc, slice in compression_dict.items():
        # Note: [slice,] is used to convert int to tuple
        cr = metrics_agg_merged[slice,]['CompressionRatio']*100
        csa = metrics_agg_merged[slice,]['MEAN(area)']
        print(f'\tdisc {disc} (corresponding to slice {slice}). CSA = {csa:.2f} mm2. CR = {cr:.2f}.')


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
