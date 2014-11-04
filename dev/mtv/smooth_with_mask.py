#!/usr/bin/env python
__author__ = 'slevy'
#########################################################################################
#
# Smooth only keeping ROIs according to a mask.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Simon LEVY
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import getopt
import sys
import sct_utils as sct
import numpy as np
import nibabel as nib
import math
import os
import scipy
from sct_extract_metric import extract_metric_within_tract
import pylab
import matplotlib.legend_handler as lgd

class param:
    def __init__(self):
        self.debug = 1

#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization of variables


    # Parameters for debug mode
    if param.debug:
        sct.printv('\n*** WARNING: DEBUG MODE ON ***\n', type='warning')
        fname_spgr = '/home/django/slevy/data/boston/hc_sc_003/mtv/spgr10_crop.nii.gz'
        fname_epi = '/home/django/slevy/data/boston/hc_sc_003/mtv/b1/epi60.nii.gz,/home/django/slevy/data/boston/hc_sc_003/mtv/b1/epi120.nii.gz'
        fname_mask = '/home/django/slevy/data/boston/hc_sc_003/mtv/spgr10_crop_csf_mask.nii.gz'
        fname_cord_seg = '/home/django/slevy/data/boston/hc_sc_003/mtv/spgr10_crop_seg.nii.gz'

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:m:') # define flags
    except getopt.GetoptError as err: # check if the arguments are defined
        print str(err) # error
        #usage() # display usage
    for opt, arg in opts: # explore flags
        if opt in '-d':
            fname_spgr = arg  # e.g.: spgr10_crop.nii.gz
        if opt in '-i':
            fname_epi = arg  # e.g.: ep_fa60.nii.gz,ep_fa120.nii.gz (!!!BECAREFUL!!!: first image = flip angle of alpha, second image =flip angle of 2*alpha)
        if opt in '-m':
            fname_mask = arg  # e.g.:
        if opt in '-s':
            fname_cord_seg = arg  # e.g.: spgr10_crop_seg.nii.gz

    # Parse inputs to get the actual data
    fname_epi_list = fname_epi.split(',')

    # Put the EPIs in the right space and compute the ratio of the two EPIs
    fname_ratio = put_epi_in_right_space(fname_epi_list, fname_spgr)

    # Load data and masks
    half_ratio = nib.load(fname_ratio).get_data()
    hdr = nib.load(fname_ratio).get_header()
    rois_to_avoid = nib.load(fname_mask).get_data()
    cord_seg = nib.load(fname_cord_seg).get_data()

    # Smooth the EPI ratio
    half_ratio_smoothed = smooth_according_to_mask(half_ratio, rois_to_avoid, cord_seg)

    # Generate the smoothed B1 profile as a NIFTI file with the right header
    path_epi, file_first_epi, ext_epi = sct.extract_fname(fname_epi_list[0])
    fname_smoothed_B1_profile_output = path_epi + 'epi_half_ratio_smoothed' + ext_epi
    sct.printv('Generate the T1, PD and MTVF maps as a NIFTI file with the right header...')
    # Associate the header to the MTVF and PD maps data as a NIFTI file
    smoothed_B1_profile_img_with_hdr = nib.Nifti1Image(half_ratio_smoothed, None, hdr)
    # Save the T1, PD and MTVF maps file
    nib.save(smoothed_B1_profile_img_with_hdr, fname_smoothed_B1_profile_output)
    sct.printv('\tFile created:\n\t\t\t\t' + fname_smoothed_B1_profile_output)

    # # Estimate flip angle (alpha)
    # measured_alpha = (180 / np.pi) * np.arccos(half_ratio)  # DOES NOT WORK
    #
    # # Divide the measured fip angle by the nominal flip angle to get a kind of B1 scale
    # b1_map_scale = measured_alpha / nominal_alpha


def smooth_according_to_mask(half_ratio, rois_to_avoid, cord_seg):
    """Smooth the image with a Gaussian filter, replacing the ROIs designed in the mask by the mean value in the spinal
    cord."""

    # Extract mean signal per slice in the cord
    (nx, ny, nz) = cord_seg.shape
    mean_signal_in_cord_per_slice = np.empty((nz))
    for z in range(0, nz):
        slice_cord_seg = np.empty((1), dtype=object)
        slice_cord_seg[0] = cord_seg[..., z]

        mean_signal_in_cord_per_slice[z] = extract_metric_within_tract(half_ratio[..., z], slice_cord_seg, 'wa', 1)[0][0]

    # Replace the voxels specified by the mask by the mean signal in cord for the slice
    for k in range(0, nz):
        for i in range(0, nx):
            for j in range(0, ny):

                if rois_to_avoid[i,j,k] == 1:
                    half_ratio[i,j,k] = mean_signal_in_cord_per_slice[z]

    # Smooth the corrected ratio
    half_ratio_smoothed = scipy.ndimage.gaussian_filter(half_ratio, sigma=1.0, order=0)

    return half_ratio_smoothed


def put_epi_in_right_space(fname_epi_list, fname_spgr):
    """Put the EPI in the space of SPGR images and compute the ratio of the double flip angle EPI. Return this ratio."""


    # Check if user indeed gave 2 images to estimate B1 field
    if len(fname_epi_list) != 2:
        sct.printv('ERROR: You didn\'t provide exactly 2 images to estimate B1 field. Exit program.', type='error')
        sys.exit(2)

    # Check if the 2 images have the same dimensions
    b1_nx, b1_ny, b1_nz, b1_nt, b1_px, b1_py, b1_pz, b1_pt = sct.get_dimension(fname_epi_list[0])
    b1_nx2, b1_ny2, b1_nz2, b1_nt2, b1_px2, b1_py2, b1_pz2, b1_pt2 = sct.get_dimension(fname_epi_list[1])
    if (b1_nx, b1_ny, b1_nz) != (b1_nx2, b1_ny2, b1_nz2):
        sct.printv('ERROR: The 2 images to estimate B1 field have not the same dimensions. Exit program.', type='error')
        sys.exit(2)

    # Compute half-ratio of the two images
    sct.printv('\nCompute the half ratio of the 2 images given as input to estimate B1 field...')
    path_b1_map, file_b1_map, ext_b1_map = sct.extract_fname(fname_epi_list[0])
    fname_half_ratio = path_b1_map + 'b1_maps_half_ratio' + ext_b1_map
    sct.run('fslmaths -dt double ' + fname_epi_list[0] + ' -div 2 -div ' + fname_epi_list[1] + ' ' + fname_half_ratio)
    sct.printv('\tDone.--> ' + fname_half_ratio)

    # Check if the dimensions of the images for b1 estimations are the same as the SPGR data
    sct.printv('\nCheck consistency between dimensions of images for B1 estimations and dimensions of SPGR images...')
    spgr_nx, spgr_ny, spgr_nz, spgr_nt, spgr_px, spgr_py, spgr_pz, spgr_pt = sct.get_dimension(fname_spgr)
    if (b1_nx, b1_ny, b1_nz) != (spgr_nx, spgr_ny, spgr_nz):
        sct.printv(
            '\n\tDimensions of images for B1 field estimation are different from dimensions of SPGR data. \n\t--> resample it to the dimensions of SPGR data...')
        path_fname_half_ratio, file_fname_half_ratio, ext_fname_half_ratio = sct.extract_fname(fname_half_ratio)
        path_spgr10, file_spgr10, ext_spgr10 = sct.extract_fname(fname_spgr)

        # fname_output = path_fname_ratio + file_fname_ratio + '_resampled_' + str(spgr_nx) + 'x' + str(spgr_ny) + 'x' + str(spgr_nz) + 'vox' + ext_fname_ratio
        # sct.run('c3d ' + fname_ratio + ' -interpolation Cubic -resample ' + str(spgr_nx) + 'x' + str(spgr_ny) + 'x' + str(spgr_nz) + 'vox -o '+fname_output)

        fname_output = path_fname_half_ratio + file_fname_half_ratio + '_in_' + file_spgr10 + '_space' + ext_fname_half_ratio
        sct.run(
            'sct_register_multimodal -i ' + fname_half_ratio + ' -d ' + fname_spgr + ' -o ' + fname_output + ' -p 0,SyN,0.5,MeanSquares')
        # Delete useless outputs
        sct.delete_nifti(path_fname_half_ratio + '/warp_dest2src.nii.gz')
        sct.delete_nifti(path_fname_half_ratio + '/warp_src2dest.nii.gz')
        sct.delete_nifti(path_fname_half_ratio + '/spgr10_crop_reg.nii.gz')

        fname_half_ratio = fname_output
        sct.printv('\t\tDone.--> ' + fname_output)

    sct.printv('\tDone.')

    return fname_half_ratio

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # Construct object fro class 'param'
    param = param()
    # Call main function
    main()