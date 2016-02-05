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
from sct_image import get_orientation, set_orientation
from sct_utils import printv, tmp_create
from os import rmdir

class Param:
    def __init__(self):
        self.verbose = 1


# PARSER
# ==========================================================================================
def get_parser():

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute SNR in a given ROI according to different methods presented in Dietrich et al., Measurement of signal-to-noise ratios in MR images: Influence of multichannel coils, parallel imaging, and reconstruction filters (2007).')
    parser.add_option(name="-i",
                      type_value='image_nifti',
                      description="Input images to compute the SNR on. Must be concatenated in time. Typically, 2 or 3 b0s concatenated in time (depending on the method used).",
                      mandatory=True,
                      example="b0s.nii.gz")
    parser.add_option(name="-s",
                      type_value='image_nifti',
                      description='Mask of the ROI to compute the SNR in.',
                      mandatory=True,
                      example='dwi_moco_mean_seg.nii.gz')
    parser.add_option(name="-vertfname",
                      type_value='image_nifti',
                      description='File name of the vertebral labeling registered to the input images (flag -i).',
                      mandatory=False,
                      example='label/template/MNI-Poly-AMU_levels.nii.gz',
                      default_value='None')
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
    parser.add_option(name="-method",
                      type_value='str',
                      description='Method to use to compute the SNR:\n- multi: take advantage of more than 2 images to compute SNR (reference method)\n- diff: enable to compute SNR from only two images.',
                      mandatory=False,
                      example='diff',
                      default_value='multi')
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic.""",
                      mandatory=False,
                      default_value=param.verbose,
                      example=['0', '1'])
    return parser


# MAIN
# ==========================================================================================
def main():

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    b0s_fname = arguments["-i"]
    mask_fname = arguments["-s"]
    vert_label_fname = arguments["-vertfname"]
    vert_levels = arguments["-vert"]
    slices_of_interest = arguments["-z"]
    method = arguments["-method"]
    verbose = int(arguments['-v'])

    # Check if data are in RPI
    input_im = Image(b0s_fname)
    input_orient = get_orientation(input_im)

    # If orientation is not RPI, change to RPI
    if input_orient != 'RPI':
        printv('\nCreate temporary folder to change the orientation of the NIFTI files into RPI...', verbose)
        path_tmp = tmp_create()
        # change orientation and load data
        printv('\nChange input image orientation and load it...', verbose)
        input_im_rpi = set_orientation(input_im, 'RPI', fname_out=path_tmp+'input_RPI.nii')
        input_data = input_im_rpi.data
        # Do the same for the mask
        printv('\nChange mask orientation and load it...', verbose)
        mask_im_rpi = set_orientation(Image(mask_fname), 'RPI', fname_out=path_tmp+'mask_RPI.nii')
        mask_data = mask_im_rpi.data
        # Do the same for vertebral labeling if present
        if vert_levels != 'None':
            printv('\nChange vertebral labeling file orientation and load it...', verbose)
            vert_label_im_rpi = set_orientation(Image(vert_label_fname), 'RPI', fname_out=path_tmp+'vert_labeling_RPI.nii')
            vert_labeling_data = vert_label_im_rpi.data
        # Remove the temporary folder used to change the NIFTI files orientation into RPI
        printv('\nRemove the temporary folder...', verbose)
        rmdir(path_tmp)
    else:
        # Load data
        printv('\nLoad data...', verbose)
        input_data = input_im.data
        mask_data = Image(mask_fname).data
        if vert_levels != 'None':
            vert_labeling_data = Image(vert_label_fname).data
    printv('\tDone.', verbose)


    # Get slices corresponding to vertebral levels
    if vert_levels != 'None':
        from sct_extract_metric import get_slices_matching_with_vertebral_levels
        slices_of_interest, actual_vert_levels, warning_vert_levels = get_slices_matching_with_vertebral_levels(mask_data, vert_levels, vert_labeling_data, verbose)

    # Remove slices that were not selected
    if slices_of_interest == 'None':
        slices_of_interest = '0:'+str(mask_data.shape[2]-1)
    slices_boundary = slices_of_interest.split(':')
    slices_of_interest_list = range(int(slices_boundary[0]), int(slices_boundary[1])+1)
    # Crop
    input_data = input_data[:, :, slices_of_interest_list, :]
    mask_data = mask_data[:, :, slices_of_interest_list]

    # Compute SNR in ROI
    indexes_roi = np.where(mask_data == 1)
    if method == 'multi':
        # b0_1 = input_data[:, :, :, 0]
        # b0_2 = input_data[:, :, :, 1]
        # b0_3 = input_data[:, :, :, 2]
        # mean_b0_1 = np.mean(b0_1[indexes_roi])
        # mean_b0_2 = np.mean(b0_2[indexes_roi])
        # mean_b0_3 = np.mean(b0_3[indexes_roi])
        # mean_roi = np.mean([mean_b0_1, mean_b0_2, mean_b0_3])
        mean_roi = np.mean(input_data[indexes_roi])
        std_input_temporal = np.std(input_data, 3)
        mean_std_temporal = np.mean(std_input_temporal[indexes_roi])
        SNR = mean_roi/mean_std_temporal
    elif method == 'diff':
        b0_1 = input_data[:, :, :, 0]
        b0_2 = input_data[:, :, :, 1]
        numerator = np.mean(np.add(b0_1[indexes_roi], b0_2[indexes_roi]))
        denominator = np.sqrt(2)*np.std(np.subtract(b0_1[indexes_roi], b0_2[indexes_roi]))
        SNR = numerator/denominator

    else:
        printv('\nERROR: unknown mehtod', type='error')

    # Display result
    printv('\nSNR = '+str(SNR)+' (used method "'+method+')', type='info')




# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    # call main function
    main()
