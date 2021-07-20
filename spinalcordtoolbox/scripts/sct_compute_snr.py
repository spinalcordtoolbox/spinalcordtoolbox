#!/usr/bin/env python
#########################################################################################
#
# Compute SNR in a given ROI according to different methods presented in Dietrich et al.,
# Measurement of signal-to-noise ratios in MR images: Influence of multichannel coils,
# parallel imaging, and reconstruction filters (2007).
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Simon LEVY
#
# About the license: see the file LICENSE.TXT
########################################################################################

import sys
import os


import numpy as np

from spinalcordtoolbox.image import Image, empty_like, add_suffix
from spinalcordtoolbox.utils import SCTArgumentParser, Metavar, parse_num_list, init_sct, printv, set_loglevel


# PARAMETERS
class Param(object):
    # The constructor
    def __init__(self):
        self.almost_zero = np.finfo(float).eps

# PARSER
# ==========================================================================================


def get_parser():
    parser = SCTArgumentParser(
        description='Compute SNR using methods described in [Dietrich et al., Measurement of'
                    ' signal-to-noise ratios in MR images: Influence of multichannel coils, parallel '
                    'imaging, and reconstruction filters. J Magn Reson Imaging 2007; 26(2): 375-385].'
    )

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-i',
        required=True,
        help='3D or 4D data to compute the SNR on (along the 4th dimension). Example: b0s.nii.gz',
        metavar=Metavar.file)

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        '-m',
        help='Binary (or weighted) mask within which SNR will be averaged. Example: dwi_moco_mean_seg.nii.gz',
        metavar=Metavar.file,
        default='')
    optional.add_argument(
        '-m-noise',
        help="Binary (or weighted) mask within which noise will be calculated. Only valid for '-method single'.",
        metavar=Metavar.file,
        default='')
    optional.add_argument(
        '-method',
        help='R|Method to use to compute the SNR (default: diff):\n'
             "- diff: Substract two volumes (defined by -vol) and estimate noise variance within the ROI "
             "(flag '-m' is required). Requires a 4D volume.\n"
             "- mult: Estimate noise variance over time across volumes specified with '-vol'. Requires a 4D volume.\n"
             "- single: Estimates noise variance in a 5x5 square at the corner of the image, and average the mean "
             "signal inside the ROI specified by flag '-m'. The variance and mean are corrected for Rayleigh "
             "distributions. This corresponds to the cases SNRstd and SNRmean in the Dietrich et al. article. Uses a "
             "3D or a 4D volume. If a 4D volume is input, the volume to compute SNR on is specified by '-vol'.",
        choices=('diff', 'mult', 'single'),
        default='diff')
    optional.add_argument(
        '-vol',
        help="Volumes to compute SNR from. Separate with ',' (Example: '-vol 0,1'), or select range "
             "using ':' (Example: '-vol 2:50'). By default, all volumes in are selected, except if '-method single' "
             "in which case the first volume is selected.",
        metavar=Metavar.str,
        default='')
    optional.add_argument(
        '-r',
        type=int,
        help='Remove temporary files.',
        default=1,
        choices=(0, 1))
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")
    optional.add_argument(
        '-o',
        metavar=Metavar.str,
        type=str,
        default=None,
        help="File name to write the computed SNR to."
    )

    return parser


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    Source: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)
    
    # Default params
    param = Param()

    # Get parser info
    fname_data = arguments.i
    # TODO: fix issue below
    if arguments.m is not None:
        fname_mask = arguments.m
    else:
        fname_mask = ''
    fname_mask_noise = arguments.m_noise
    method = arguments.method
    file_name = arguments.o

    # Check parameters
    if method in ['diff', 'single']:
        if not fname_mask:
            raise SCTArgumentParser.error(parser, f"You need to provide a mask with -method {method}.")

    # Load data
    im_data = Image(fname_data)
    data = im_data.data
    dim = len(data.shape)
    if fname_mask:
        mask = Image(fname_mask).data

    # Check dimensionality
    if method in ['diff', 'mult']:
        if dim is not 4:
            raise ValueError(f"Input data dimension: {dim}. Input dimension for this method should be 4.")
    if method in ['single']:
        if dim not in [3, 4]:
            raise ValueError(f"Input data dimension: {dim}. Input dimension for this method should be 3 or 4.")

    # Check dimensionality of mask
    if fname_mask:
        if len(mask.shape) is not 3:
            raise ValueError(f"Input mask dimension: {dim}. Input dimension for the mask should be 3.")

    # Retrieve selected volumes
    index_vol = parse_num_list(arguments.vol)
    if not index_vol:
        if method in ['diff', 'mult']:
            index_vol = range(data.shape[3])
        elif method in ['single']:
            index_vol = 0

    # Compute SNR
    # NB: "time" is assumed to be the 4th dimension of the variable "data"
    if method == 'mult':
        # Compute mean and STD across time
        data_mean = np.mean(data[:, :, :, index_vol], axis=3)
        data_std = np.std(data[:, :, :, index_vol], axis=3, ddof=1)
        # Generate mask where std is different from 0
        mask_std_nonzero = np.where(data_std > param.almost_zero)
        snr_map = np.zeros_like(data_mean)
        snr_map[mask_std_nonzero] = data_mean[mask_std_nonzero] / data_std[mask_std_nonzero]
        # Output SNR map
        fname_snr = add_suffix(fname_data, '_SNR-' + method)
        im_snr = empty_like(im_data)
        im_snr.data = snr_map
        im_snr.save(fname_snr, dtype=np.float32)
        # Output non-zero mask
        fname_stdnonzero = add_suffix(fname_data, '_mask-STD-nonzero' + method)
        im_stdnonzero = empty_like(im_data)
        data_stdnonzero = np.zeros_like(data_mean)
        data_stdnonzero[mask_std_nonzero] = 1
        im_stdnonzero.data = data_stdnonzero
        im_stdnonzero.save(fname_stdnonzero, dtype=np.float32)
        # Compute SNR in ROI
        if fname_mask:
            snr_roi = np.average(snr_map[mask_std_nonzero], weights=mask[mask_std_nonzero])

    elif method == 'diff':
        # Check user selected exactly 2 volumes for this method.
        if not len(index_vol) == 2:
            raise ValueError(f"Number of selected volumes: {len(index_vol)}. The method 'diff' should be used with "
                             f"exactly 2 volumes. You can specify the number of volumes with the flag '-vol'.")
        data_2vol = np.take(data, index_vol, axis=3)
        # Compute mean in ROI
        data_mean = np.mean(data_2vol, axis=3)
        mean_in_roi = np.average(data_mean, weights=mask)
        data_sub = np.subtract(data_2vol[:, :, :, 1], data_2vol[:, :, :, 0])
        _, std_in_roi = weighted_avg_and_std(data_sub, mask)
        # Compute SNR, correcting for Rayleigh noise (see eq. 7 in Dietrich et al.)
        snr_roi = (2 / np.sqrt(2)) * mean_in_roi / std_in_roi

    elif method == 'single':
        # Check that the input volume is 3D, or if it is 4D, that the user selected exactly 1 volume for this method.
        if dim == 3:
            data3d = data
        elif dim == 4:
            if not len(index_vol) == 1:
                raise ValueError(f"Number of selected volumes: {len(index_vol)}. The method 'diff' should be used with "
                                 f"exactly 1 volume. You can specify the number of volumes with the flag '-vol'.")
            data3d = np.take(data, index_vol, axis=3)
        # Check that input noise mask is provided
        if fname_mask_noise:
            mask_noise = Image(fname_mask_noise).data
        else:
            raise RuntimeError(f"A noise mask is mandatory with '-method single'.")
        # Compute mean in ROI
        mean_in_roi = np.average(data3d, weights=mask)
        # Compute standard deviation in background
        std_in_roi = np.std(data3d[mask_noise])
        # Compute SNR, correcting for Rayleigh noise (see eq. A12 in Dietrich et al.)
        snr_roi = np.sqrt((4 - np.pi) / 2) * mean_in_roi / std_in_roi

    # Display result
    if fname_mask:
        printv('\nSNR_' + method + ' = ' + str(snr_roi) + '\n', type='info')
    
    #Added function for text file
    if file_name is not None:
        with open(file_name, "w") as f:
            f.write(str(snr_roi))
            printv('\nFile saved to ' + file_name)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
