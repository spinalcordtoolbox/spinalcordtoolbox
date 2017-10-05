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

import sys
import numpy as np
from msct_parser import Parser
from msct_image import Image
from sct_image import get_orientation, orientation
import sct_utils as sct
import shutil


# PARSER
# ==========================================================================================
def get_parser():

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute SNR in a given ROI using methods described in [Dietrich et al., Measurement of signal-to-noise ratios in MR images: Influence of multichannel coils, parallel imaging, and reconstruction filters. J Magn Reson Imaging 2007; 26(2): 375-385].')
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
                      '- diff: Substract two volumes (defined by -vol) and estimate noise variance over space.\n'
                      '- mult: Estimate noise variance over time across volumes specified with -vol.',
                      mandatory=False,
                      default_value='diff',
                      example=['diff', 'mult'])
    parser.add_option(name='-vol',
                      type_value=[[','], 'int'],
                      description='List of volume numbers to use for computing SNR, separated with ",". Example: 0,31. To select all volumes in series set to -1.',
                      mandatory=False,
                      default_value=[-1])
    parser.add_option(name="-vertfile",
                      type_value='image_nifti',
                      description='File name of the vertebral labeling registered to the input images.',
                      mandatory=False,
                      default_value='label/template/MNI-Poly-AMU_level.nii.gz')
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

    # initialization
    fname_mask = ''

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
    verbose = int(arguments['-v'])

    # Check if data are in RPI
    input_im = Image(fname_data)
    input_orient = get_orientation(input_im)

    # If orientation is not RPI, change to RPI
    if input_orient != 'RPI':
        sct.printv('\nCreate temporary folder to change the orientation of the NIFTI files into RPI...', verbose)
        path_tmp = sct.tmp_create()
        # change orientation and load data
        sct.printv('\nChange input image orientation and load it...', verbose)
        input_im_rpi = orientation(input_im, ori='RPI', set=True, fname_out=path_tmp + 'input_RPI.nii')
        input_data = input_im_rpi.data
        # Do the same for the mask
        sct.printv('\nChange mask orientation and load it...', verbose)
        mask_im_rpi = orientation(Image(fname_mask), ori='RPI', set=True, fname_out=path_tmp + 'mask_RPI.nii')
        mask_data = mask_im_rpi.data
        # Do the same for vertebral labeling if present
        if vert_levels != 'None':
            sct.printv('\nChange vertebral labeling file orientation and load it...', verbose)
            vert_label_im_rpi = orientation(Image(vert_label_fname), ori='RPI', set=True, fname_out=path_tmp + 'vert_labeling_RPI.nii')
            vert_labeling_data = vert_label_im_rpi.data
        # Remove the temporary folder used to change the NIFTI files orientation into RPI
        sct.printv('\nRemove the temporary folder...', verbose)
        shutil.rmtree(path_tmp, True)
    else:
        # Load data
        sct.printv('\nLoad data...', verbose)
        input_data = input_im.data
        mask_data = Image(fname_mask).data
        if vert_levels != 'None':
            vert_labeling_data = Image(vert_label_fname).data
    sct.printv('\tDone.', verbose)

    # Get slices corresponding to vertebral levels
    if vert_levels != 'None':
        from sct_extract_metric import get_slices_matching_with_vertebral_levels
        slices_of_interest, actual_vert_levels, warning_vert_levels = get_slices_matching_with_vertebral_levels(mask_data, vert_levels, vert_labeling_data, verbose)

    # Remove slices that were not selected
    if slices_of_interest == 'None':
        slices_of_interest = '0:' + str(mask_data.shape[2] - 1)
    slices_boundary = slices_of_interest.split(':')
    slices_of_interest_list = range(int(slices_boundary[0]), int(slices_boundary[1]) + 1)
    # Crop
    input_data = input_data[:, :, slices_of_interest_list, :]
    mask_data = mask_data[:, :, slices_of_interest_list]

    # if user selected all slices (-vol -1), then assign index_vol
    if index_vol[0] == -1:
        index_vol = range(0, input_data.shape[3], 1)

    # Get signal and noise
    indexes_roi = np.where(mask_data == 1)
    if method == 'mult':
        # get voxels in ROI to obtain a (x*y*z)*t 2D matrix
        input_data_in_roi = input_data[indexes_roi]
        # compute signal and STD across by averaging across time
        signal = np.mean(input_data_in_roi[:, index_vol])
        std_input_temporal = np.std(input_data_in_roi[:, index_vol], 1)
        noise = np.mean(std_input_temporal)
    elif method == 'diff':
        # if user did not select two volumes, then exit with error
        if not len(index_vol) == 2:
            sct.printv('ERROR: ' + str(len(index_vol)) + ' volumes were specified. Method "diff" should be used with exactly two volumes.', 1, 'error')
        data_1 = input_data[:, :, :, index_vol[0]]
        data_2 = input_data[:, :, :, index_vol[1]]
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
    sct.start_stream_logger()
    # call main function
    main()
