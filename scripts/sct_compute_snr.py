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

from __future__ import division, absolute_import

import sys
import numpy as np
from msct_parser import Parser
from spinalcordtoolbox.image import Image, empty_like
import sct_utils as sct


# PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.almost_zero = 0.000000001

# PARSER
# ==========================================================================================
def get_parser():

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute SNR using methods described in [Dietrich et al., Measurement of'
                                 ' signal-to-noise ratios in MR images: Influence of multichannel coils, parallel '
                                 'imaging, and reconstruction filters. J Magn Reson Imaging 2007; 26(2): 375-385].')
    parser.add_option(name="-i",
                      type_value='image_nifti',
                      description="4D data to compute the SNR on (along the 4th dimension).",
                      mandatory=True,
                      example="b0s.nii.gz")
    parser.add_option(name="-m",
                      type_value='image_nifti',
                      description='Binary (or weighted) mask within which SNR will be averaged.',
                      mandatory=False,
                      default_value='',
                      example='dwi_moco_mean_seg.nii.gz')
    parser.add_option(name="-method",
                      type_value='multiple_choice',
                      description='Method to use to compute the SNR:\n'
                      '- diff (default): Substract two volumes (defined by -vol) and estimate noise variance within the ROI (flag -m is required).\n'
                      '- mult: Estimate noise variance over time across volumes specified with -vol.',
                      mandatory=False,
                      default_value='diff',
                      example=['diff', 'mult'])
    parser.add_option(name='-vol',
                      type_value=[[','], 'int'],
                      description='List of volume numbers to use for computing SNR, separated with ",". Example: 0,31. '
                                  'To select all volumes in series set to -1.',
                      mandatory=False,
                      default_value=[-1])
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic.""",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
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


def main():

    # Default params
    param = Param()

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_data = arguments['-i']
    if '-m' in arguments:
        fname_mask = arguments['-m']
    else:
        fname_mask = ''
    method = arguments["-method"]
    index_vol = arguments['-vol']

    # Check parameters
    if method == 'diff':
        if not fname_mask:
            sct.printv('You need to provide a mask with -method diff. Exit.', 1, type='error')

    # Load data and orient to RPI
    im_data = Image(fname_data).change_orientation('RPI')
    data = im_data.data
    if fname_mask:
        mask = Image(fname_mask).change_orientation('RPI').data

    # if user selected all 3d volumes from the input 4d volume ("-vol -1"), then assign index_vol
    if index_vol[0] == -1:
        index_vol = range(data.shape[3])

    # Make sure user selected 2 volumes with diff method
    if method == 'diff':
        if not len(index_vol) == 2:
            sct.printv('Method "diff" should be used with exactly two volumes (specify with flag "-vol").', 1, 'error')

    # Compute SNR
    # NB: "time" is assumed to be the 4th dimension of the variable "data"
    if method == 'mult':
        # Compute mean and STD across time
        data_mean = np.mean(data[:, :, :, index_vol], axis=3)
        data_std = np.std(data[:, :, :, index_vol], axis=3)
        # Generate mask where std is different from 0
        mask_std_nonzero = np.where(data_std > param.almost_zero)
        snr_map = np.zeros_like(data_mean)
        snr_map[mask_std_nonzero] = data_mean[mask_std_nonzero] / data_std[mask_std_nonzero]
        # Output SNR map
        fname_snr = sct.add_suffix(fname_data, '_SNR-' + method)
        im_snr = empty_like(im_data)
        im_snr.data = snr_map
        im_snr.save(fname_snr, dtype=np.float32)
        # Output non-zero mask
        fname_stdnonzero = sct.add_suffix(fname_data, '_mask-STD-nonzero' + method)
        im_stdnonzero = empty_like(im_data)
        data_stdnonzero = np.zeros_like(data_mean)
        data_stdnonzero[mask_std_nonzero] = 1
        im_stdnonzero.data = data_stdnonzero
        im_stdnonzero.save(fname_stdnonzero, dtype=np.float32)
        # Compute SNR in ROI
        if fname_mask:
            mean_in_roi = np.average(data_mean[mask_std_nonzero], weights=mask[mask_std_nonzero])
            std_in_roi = np.average(data_std[mask_std_nonzero], weights=mask[mask_std_nonzero])
            snr_roi = mean_in_roi / std_in_roi
            # snr_roi = np.average(snr_map[mask_std_nonzero], weights=mask[mask_std_nonzero])

    elif method == 'diff':
        data_2vol = np.take(data, index_vol, axis=3)
        # Compute mean in ROI
        data_mean = np.mean(data_2vol, axis=3)
        mean_in_roi = np.average(data_mean, weights=mask)
        data_sub = np.subtract(data_2vol[:, :, :, 1], data_2vol[:, :, :, 0])
        _, std_in_roi = weighted_avg_and_std(data_sub, mask)
        # Compute SNR, correcting for Rayleigh noise (see eq. 7 in Dietrich et al.)
        snr_roi = (2/np.sqrt(2)) * mean_in_roi / std_in_roi

    # Display result
    if fname_mask:
        sct.printv('\nSNR_' + method + ' = ' + str(snr_roi) + '\n', type='info')


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
