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
#########################################################################################


from __future__ import division, absolute_import

import sys
import operator
import numpy as np
from msct_parser import Parser
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import parse_num_list
from spinalcordtoolbox.template import get_slices_from_vertebral_levels
import sct_utils as sct


# PARSER
# ==========================================================================================
def get_parser():

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute SNR in a given ROI using methods described in [Dietrich et al., Measurement of'
                                 ' signal-to-noise ratios in MR images: Influence of multichannel coils, parallel '
                                 'imaging, and reconstruction filters. J Magn Reson Imaging 2007; 26(2): 375-385].')
    parser.add_option(name="-i",
                      type_value='image_nifti',
                      description="4D data to compute the SNR on (along the 4th dimension).",
                      mandatory=True,
                      example="b0s.nii.gz")
    parser.add_option(name="-m",
                      type_value='image_nifti',
                      description='ROI within which SNR will be averaged.',
                      mandatory=True,
                      example='dwi_moco_mean_seg.nii.gz')
    parser.add_option(name="-method",
                      type_value='multiple_choice',
                      description='Method to use to compute the SNR:\n'
                      '- diff (default): Substract two volumes (defined by -vol) and estimate noise variance over space.\n'
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
    parser.add_option(name="-vertfile",
                      type_value='image_nifti',
                      description='File name of the vertebral labeling registered to the input images.',
                      mandatory=False,
                      default_value='label/template/PAM50_levels.nii.gz')
    parser.add_option(name="-vert",
                      type_value='str',
                      description='Vertebral levels where to compute the SNR.',
                      mandatory=False,
                      example='2:6',
                      default_value='None')
    parser.add_option(name="-z",
                      type_value='str',
                      description='Slices where to compute the SNR.',
                      mandatory=False,
                      example='2:6',
                      default_value='None')
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


# MAIN
# ==========================================================================================
def main():

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_data = arguments['-i']
    fname_mask = arguments['-m']
    vert_label_fname = arguments["-vertfile"]
    vert_levels = arguments["-vert"]
    slices_of_interest = arguments["-z"]
    index_vol = arguments['-vol']
    method = arguments["-method"]

    # Load data and orient to RPI
    data = Image(fname_data).change_orientation('RPI').data
    mask = Image(fname_mask).change_orientation('RPI').data

    # Fetch slices to compute SNR on
    slices_list = []
    if not vert_levels == 'None':
        list_levels = parse_num_list(vert_levels)
        im_vertlevel = Image(vert_label_fname).change_orientation('RPI')
        for level in list_levels:
            slices_list.append(get_slices_from_vertebral_levels(im_vertlevel, level))
        if slices_list == []:
            sct.log.error('The specified vertebral levels are not in the vertebral labeling file.')
        else:
            slices_list = reduce(operator.add, slices_list)  # flatten and sort
            slices_list.sort()
    elif not slices_of_interest == 'None':
        slices_list = parse_num_list(slices_of_interest)
    else:
        slices_list = np.arange(data.shape[2]).tolist()

    # Set to 0 all slices in the mask that are not includes in the slices_list
    nz_to_exclude = [i for i in range(mask.shape[2]) if not i in slices_list]
    mask[:, :, nz_to_exclude] = 0

    # if user selected all 3d volumes from the input 4d volume ("-vol -1"), then assign index_vol
    if index_vol[0] == -1:
        index_vol = range(data.shape[3])

    # Get signal and noise
    indexes_roi = np.where(mask == 1)
    if method == 'mult':
        # get voxels in ROI to obtain a (x*y*z)*t 2D matrix
        data_in_roi = data[indexes_roi]
        # compute signal and STD across by averaging across time
        signal = np.mean(data_in_roi[:, index_vol])
        std_input_temporal = np.std(data_in_roi[:, index_vol], 1)
        noise = np.mean(std_input_temporal)
    elif method == 'diff':
        # if user did not select two volumes, then exit with error
        if not len(index_vol) == 2:
            sct.printv('ERROR: ' + str(len(index_vol)) + ' volumes were specified. Method "diff" should be used with '
                                                         'exactly two volumes (check flag "vol").', 1, 'error')
        data_1 = data[:, :, :, index_vol[0]]
        data_2 = data[:, :, :, index_vol[1]]
        # compute voxel-average of voxelwise sum
        signal = np.mean(np.add(data_1[indexes_roi], data_2[indexes_roi]))
        # compute voxel-STD of voxelwise substraction, multiplied by sqrt(2) as described in equation 7 of Dietrich et al.
        noise = np.std(np.subtract(data_1[indexes_roi], data_2[indexes_roi])) * np.sqrt(2)

    # compute SNR
    SNR = signal / noise

    # Display result
    sct.printv('\nSNR_' + method + ' = ' + str(SNR) + '\n', type='info')


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
