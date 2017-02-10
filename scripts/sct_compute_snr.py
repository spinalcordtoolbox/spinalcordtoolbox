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

import os
import sys

import numpy as np

import msct_image
import msct_parser
import sct_extract_metric
import sct_image
import sct_utils as sct


# PARSER
# ==========================================================================================
def get_parser():

    # Initialize the parser
    parser = msct_parser.Parser(__file__)
    parser.usage.set_description('Compute SNR in a given ROI using methods described in [Dietrich et al., Measurement of signal-to-noise ratios in MR images: Influence of multichannel coils, parallel imaging, and reconstruction filters. J Magn Reson Imaging 2007; 26(2): 375-385].')
    parser.add_option(name="-i",
                      type_value='image_nifti',
                      description="Input images to compute the SNR on. Must be concatenated in time. Typically, 2 or 3 b0s concatenated in time (depending on the method used).",
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
                      '- mult: Use all volumes (or those defined by -vol) to estimate noise variance over time.',
                      mandatory=False,
                      default_value='diff',
                      example=['diff', 'mult'])
    parser.add_option(name='-vol',
                      type_value=[[','], 'int'],
                      description='List of volume numbers to use for computing SNR, separated with ",". Example: 0,1',
                      mandatory=False,
                      default_value=[0, 1])
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


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    else:
        script_name =os.path.splitext(os.path.basename(__file__))[0]
        sct.printv('{0} {1}'.format(script_name, " ".join(args)))
    # initialization
    fname_mask = ''

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    fname_data = arguments['-i']
    fname_mask = arguments['-m']
    vert_label_fname = arguments["-vertfile"]
    vert_levels = arguments["-vert"]
    slices_of_interest = arguments["-z"]
    index_vol = arguments['-vol']
    method = arguments["-method"]
    verbose = int(arguments['-v'])

    # Check if data are in RPI
    input_im = msct_image.Image(fname_data)
    input_orient = sct_image.get_orientation_3d(input_im)

    # If orientation is not RPI, change to RPI
    if input_orient != 'RPI':
        sct.printv('\nCreate temporary folder to change the orientation of the NIFTI files into RPI...', verbose)
        path_tmp = sct.tmp_create()
        # change orientation and load data
        sct.printv('\nChange input image orientation and load it...', verbose)
        input_im_rpi = sct_image.set_orientation(input_im, 'RPI', fname_out=path_tmp+'input_RPI.nii')
        input_data = input_im_rpi.data
        # Do the same for the mask
        sct.printv('\nChange mask orientation and load it...', verbose)
        mask_im_rpi = sct_image.set_orientation(msct_image.Image(fname_mask), 'RPI', fname_out=path_tmp+'mask_RPI.nii')
        mask_data = mask_im_rpi.data
        # Do the same for vertebral labeling if present
        if vert_levels != 'None':
            sct.printv('\nChange vertebral labeling file orientation and load it...', verbose)
            vert_label_im_rpi = sct_image.set_orientation(msct_image.Image(vert_label_fname), 'RPI', fname_out=path_tmp+'vert_labeling_RPI.nii')
            vert_labeling_data = vert_label_im_rpi.data
        # Remove the temporary folder used to change the NIFTI files orientation into RPI
        sct.printv('\nRemove the temporary folder...', verbose)
        os.rmdir(path_tmp)
    else:
        # Load data
        sct.printv('\nLoad data...', verbose)
        input_data = input_im.data
        mask_data = msct_image.Image(fname_mask).data
        if vert_levels != 'None':
            vert_labeling_data = msct_image.Image(vert_label_fname).data
    sct.printv('\tDone.', verbose)

    # Get slices corresponding to vertebral levels
    if vert_levels != 'None':
        slices_of_interest, actual_vert_levels, warning_vert_levels = \
            sct_extract_metric.get_slices_matching_with_vertebral_levels(mask_data, vert_levels, vert_labeling_data, verbose)

    # Remove slices that were not selected
    if slices_of_interest == 'None':
        slices_of_interest = '0:'+str(mask_data.shape[2]-1)
    slices_boundary = slices_of_interest.split(':')
    slices_of_interest_list = range(int(slices_boundary[0]), int(slices_boundary[1])+1)
    # Crop
    input_data = input_data[:, :, slices_of_interest_list, :]
    mask_data = mask_data[:, :, slices_of_interest_list]

    # Get signal and noise
    indexes_roi = np.where(mask_data == 1)
    if method == 'mult':
        signal = np.mean(input_data[indexes_roi])
        std_input_temporal = np.std(input_data, 3)
        noise = np.mean(std_input_temporal[indexes_roi])
    elif method == 'diff':
        data_1 = input_data[:, :, :, index_vol[0]]
        data_2 = input_data[:, :, :, index_vol[1]]
        signal = np.mean(np.add(data_1[indexes_roi], data_2[indexes_roi]))
        noise = np.std(np.subtract(data_1[indexes_roi], data_2[indexes_roi]))

    # compute SNR
    SNR = signal/noise

    # Display result
    sct.printv('\nSNR_'+method+' = '+str(SNR)+'\n', type='info')



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()
