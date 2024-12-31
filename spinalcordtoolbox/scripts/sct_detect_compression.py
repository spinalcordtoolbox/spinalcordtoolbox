#!/usr/bin/env python
#
# Predict compression probability in a spinal cord MRI image using spinal cord shape metrics.
# IMPORTANT NOTE: The script process *only* axial slices at the level of intervertebral discs C3/C4 (value: 4) to C6/7
# (value: 7). In other words, the script does *not* process all axial slices of the spinal cord segmentation.
#
# The script requires spinal cord segmentation (used to compute the shape metrics) and disc labels.
#
# More details in: https://pubmed.ncbi.nlm.nih.gov/35371944/
#
# Copyright (c) 2024 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import math
from typing import Sequence

import numpy as np
import pandas as pd

from spinalcordtoolbox.process_seg import compute_shape
from spinalcordtoolbox.aggregate_slicewise import aggregate_per_slice_or_level, func_wa, func_std, merge_dict
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.template import get_slices_from_vertebral_levels
from spinalcordtoolbox.image import Image, add_suffix, splitext
from spinalcordtoolbox.utils.fs import get_absolute_path
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel, printv


# Currently, the model works only for discs C3/C4 (4) to C6/7 (7)
SUPPORTED_DISCS = [4, 5, 6, 7]
# Compression cut-offs
#   >0.451 high probability of compression
#   0.345-0.451 moderate probability of compression
#   <0.345 low probability of compression
CUT_OFF_HIGH = 0.451
CUT_OFF_MILD = 0.345


def get_parser():
    parser = SCTArgumentParser(
        description=("""\
Predict compression probability in a spinal cord MRI image using spinal cord shape metrics.
IMPORTANT NOTE: The script process *only* axial slices at the level of intervertebral discs C3/C4 (value: 4) to C6/7 (value: 7).
More details in: https://pubmed.ncbi.nlm.nih.gov/35371944/
        """)
    )

    mandatoryArguments = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-s',
        metavar=Metavar.file,
        required=True,
        help="Segmentation of the spinal cord. Example: t2s_seg.nii.gz"
    )
    mandatoryArguments.add_argument(
        '-discfile',
        metavar=Metavar.file,
        required=True,
        help="File with disc labels. Example: t2s_discs.nii.gz "
             "The convention for disc labels is the following: value=3 -> disc C2/C3, value=4 -> disc C3/C4, etc."
             "Such a label file can be manually created using: sct_label_utils -i IMAGE_REF -create-viewer 4:7 "
             "or obtained automatically using the sct_label_vertebrae function "
             "(the file with the \'labeled_discs.nii.gz\' suffix)."
    )

    optional = parser.add_argument_group('OPTIONAL ARGUMENTS')
    optional.add_argument(
        '-angle-corr',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=1,
        help="Angle correction for computing morphometric measures used within "
             "\'spinalcordtoolbox.process_seg.compute_shape\'. When angle correction is used, the cord within "
             "the slice is stretched/expanded by a factor corresponding to the cosine of the angle between the "
             "centerline and the axial plane. If the cord is already quasi-orthogonal to the slab, you can set "
             "-angle-corr to 0."
    )
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


def predict_compression_probability(cr, csa, solidity, torsion, disc):
    """
    Predict compression probability using the spinal cord shape metrics for currently processed axial slice.
    See Equation 4 and Table 3 for details: https://pubmed.ncbi.nlm.nih.gov/35371944/
    :param cr: compression ratio (CR) in %
    :param csa: cross-sectional area (CSA) in mm2
    :param solidity: solidity in %
    :param torsion: torsion in degrees
    :param disc: intervertebral disc level (e.g., value=4 -> disc C3/C4)
    :return: probability
    """
    coefficients = {
        "constant": 57.501,
        "cr": -0.273,
        "csa": -0.102,
        "solidity": -0.408,
        "torsion": 2.168,
        "level_c6c7": -2.729,
    }
    model = (
        coefficients["constant"]
        + coefficients["cr"] * cr
        + coefficients["csa"] * csa
        + coefficients["solidity"] * solidity
        + (coefficients["torsion"] * torsion if torsion is not None else 0)     # edge-case when torsion is None
        + (coefficients["level_c6c7"] if disc == 7 else 0)
    )
    return math.exp(model) / (1 + math.exp(model))


def compute_shape_metrics(fname_seg, fname_disc, angle_correction, verbose):
    """
    Compute shape metrics per slice.
    Then, compute compression ratio and torsion.
    :param fname_seg: file with spinal cord segmentation
    :param fname_disc: file with disc labels
    :param angle_correction: bool: angle correction for computing morphometric measures
    :param verbose: verbosity
    :return: dictionary with aggregated metrics
    """
    metrics, fit_results = compute_shape(fname_seg,
                                         angle_correction=angle_correction,
                                         param_centerline=ParamCenterline(),
                                         verbose=verbose)

    # Compute the average and standard deviation across axial slices
    group_funcs = (('MEAN', func_wa), ('STD', func_std))
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
    # Compute compression ratio
    metrics_agg_merged = compute_compression_ratio(metrics_agg_merged)
    # Compute torsion
    metrics_agg_merged = compute_torsion(metrics_agg_merged, verbose)

    return metrics_agg_merged


def compute_compression_ratio(metrics_agg_merged):
    """
    Compute compression ratio (CR) as 'diameter_AP' / 'diameter_RL'
    :param metrics_agg_merged: dictionary with aggregated metrics
    :return: dictionary with compression ratio added
    """
    for metrics in metrics_agg_merged.values():  # Loop across slices
        ap = metrics['MEAN(diameter_AP)']
        rl = metrics['MEAN(diameter_RL)']
        metrics['MEAN(compression_ratio)'] = None if ap is None or rl is None else ap / rl

    return metrics_agg_merged


def compute_torsion(metrics_agg_merged, verbose):
    """
    Compute torsion as the average of absolute differences in orientation between the given slice and 3 slices
    above and below. For details see Equation 3 and Supplementary Material in https://pubmed.ncbi.nlm.nih.gov/35371944/.
    NOTE: As the torsion is computed from 3 slices above and below, it cannot be computed for the 3 first and last
    3 slices --> `torsion = None` is used for the 3 first and 3 last slices.
    :param metrics_agg_merged: dictionary with aggregated metrics
    :param verbose: verbosity
    :return: dictionary with torsion added
    """
    slices = list(metrics_agg_merged.keys())[3:-3]

    for key in metrics_agg_merged.keys():  # Loop across slices
        if key in slices:
            try:
                # Note: the key is a tuple (e.g. `1,`), not an int (e.g., 1), thus key[0] is used to convert tuple to int
                # and `,` is used to convert int back to tuple
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
            except Exception as e:
                # TODO: the warning below is raise mainly in cases of no SC segmentation, for example above the C1 level
                #  --> consider processing only slices with SC segmentation
                printv(f"WARNING: Failed to compute torsion for slice {key[0]}. Use `-v 2` flag to see details",
                       verbose, type='warning')
                if verbose == 2:
                    printv(f"\t{e}", verbose)
                metrics_agg_merged[key]['Torsion'] = None
        else:
            metrics_agg_merged[key]['Torsion'] = None

    return metrics_agg_merged


def process_compression(metrics_agg_merged, disc_slices):
    """
    Process shape metrics for axial slices at the level of each disc to predict the compression probability.
    :param metrics_agg_merged: dictionary with aggregated metrics
    :param disc_slices: dictionary with disc labels as keys and corresponding axial slice numbers as values
    :return: dataframe with compression probability
    """
    compression_df = pd.DataFrame()
    # Loop across discs
    for disc, slice in disc_slices.items():
    # for disc, slices in disc_slices.items():
    #     # Add one slice above and below to process multiple slices around the disc to compensate for potential
    #     # disc label shift in the superior-inferior direction
    #     slices = [slices - 1, slices, slices + 1]
    #     for slice in slices:
        if disc in SUPPORTED_DISCS:
            # Note: [slice,] is used to convert int to tuple
            cr = metrics_agg_merged[slice,]['MEAN(compression_ratio)'] * 100    # to convert to %
            csa = metrics_agg_merged[slice,]['MEAN(area)']
            solidity = metrics_agg_merged[slice,]['MEAN(solidity)'] * 100     # to convert to %
            torsion = metrics_agg_merged[slice,]['Torsion']
            # Compute compression probability
            probability = predict_compression_probability(cr, csa, solidity, torsion, disc)
            compression_category = 'high' if probability > CUT_OFF_HIGH else \
                'moderate' if probability > CUT_OFF_MILD else 'low'
            compression_df = pd.concat([compression_df,
                                        pd.DataFrame([{'Disc': disc, 'Axial slice #': slice,
                                                       'Compression probability': probability,
                                                       'Compression probability category': compression_category,
                                                       'Compression ratio (%)': cr, 'CSA (mm2)': csa,
                                                       'Solidity (%)': solidity, 'Torsion (degrees)': torsion}])],
                                       ignore_index=True)

    return compression_df


def get_disc_slices(fname_disc):
    """
    Get axial slice numbers corresponding to the intervertebral discs.
    :param fname_disc: file with disc labels
    :return: dictionary with disc labels as keys and corresponding axial slice numbers as values
    """
    im_disc = Image(fname_disc).change_orientation('RPI')
    discs = [int(disc) for disc in np.trim_zeros(np.unique(im_disc.data))]  # get unique disc labels (e.g., 3, 4, 5 etc)
    # Find axial slices corresponding to the discs
    # Note: [0] is used because get_slices_from_vertebral_levels returns list
    disc_slices = {disc: get_slices_from_vertebral_levels(im_disc, disc)[0] for disc in discs}

    return disc_slices


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    # Parse arguments
    fname_seg = get_absolute_path(arguments.s)
    fname_disc = get_absolute_path(arguments.discfile)
    angle_correction = bool(arguments.angle_corr)

    # Compute shape metrics per slice
    metrics_agg_merged = compute_shape_metrics(fname_seg, fname_disc, angle_correction, verbose)

    # Get discs corresponding to the compression
    disc_slices = get_disc_slices(fname_disc)

    # Process axial slices for each disc to compute compression probability
    compression_df = process_compression(metrics_agg_merged, disc_slices)

    # Save the results to a CSV file
    fname_out = splitext(add_suffix(fname_seg, '_compression_results'))[0] + '.csv'
    compression_df.to_csv(fname_out, index=False)
    printv(f"Results saved to: {fname_out}", verbose)

    # Loop across discs and print results to terminal
    for index, row in compression_df.iterrows():
        printv(f"Disc {int(row['Disc'])} at axial slice {int(row['Axial slice #'])} has a "
               f"{row['Compression probability category']} probability of compression: "
               f"{row['Compression probability'] * 100:.2f}%. ", verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
