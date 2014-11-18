#!/usr/bin/env python
__author__ = 'slevyrosetti'
#########################################################################################
#
# Compute a proton density map and a T1 map from SPGR datasets and b1 map.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Simon LEVY
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import getopt
import sys
import numpy as np
import nibabel as nib
import math
import os
import scipy
import pylab
import matplotlib.legend_handler as lgd
import commands
# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct
from sct_extract_metric import extract_metric_within_tract

class param:
    def __init__(self):
        self.debug = 0
        self.verbose = 1
        self.file_output = 'T1_map,PD_map,MTVF_map'  # file name of the PD and MTVF maps to be generated as ouptut
        self.method = 'mean-PD-in-CSF-from-mean-SPGR'  # method used to compute MTVF for the division by the proton density in CSF


# Define a context manager to suppress stdout and stderr, in order to suppress warnings when running the function 'polyfit' in function 'estimate_PD_and_T1'
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization of variables
    alpha_b1 = None
    flip_angles = ''
    fname_csf_mask = ''
    fname_b1_smoothed = ''
    file_output = param.file_output
    method = param.method
    fname_spgr_seg = ''
    tr = ''
    box_mask = None
    verbose = param.verbose

    # Parameters for debug mode
    if param.debug:
        sct.printv('\n*** WARNING: DEBUG MODE ON ***\n', type='warning')
        alpha_b1 = 60
        flip_angles = '4,10,20,30'
        fname_csf_mask = '/home/django/slevy/data/boston/hc_sc_001/mtv/spgr10_crop_csf_mask.nii.gz'
        fname_b1_smoothed = '' #'/home/django/slevy/data/boston/hc_sc_001/mtv/b1/b1_smoothed_in_spgr10_space_crop.nii.gz' #'/home/django/slevy/data/boston/hc_sc_003/mtv/b1/gre60.nii.gz,/home/django/slevy/data/boston/hc_sc_003/mtv/b1/gre120.nii.gz' #'/home/django/slevy/data/handedness_asymmetries/errsm_31/b1_estimation/ep_fa60.nii.gz,/home/django/slevy/data/handedness_asymmetries/errsm_31/b1_estimation/ep_fa120.nii.gz'
        fname_spgr_data = '/home/django/slevy/data/boston/hc_sc_001/mtv/spgr4_crop.nii.gz,/home/django/slevy/data/boston/hc_sc_001/mtv/spgr10_crop.nii.gz,/home/django/slevy/data/boston/hc_sc_001/mtv/spgr20_crop.nii.gz,/home/django/slevy/data/boston/hc_sc_001/mtv/spgr30_crop.nii.gz'
        file_output = 'T1_map,PD_map,MTVF_map'
        method = 'mean-PD-in-CSF-from-mean-SPGR'
        fname_spgr_seg = '/home/django/slevy/data/boston/hc_sc_001/mtv/spgr10_crop_seg.nii.gz'
        tr = float(0.02)

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'a:b:c:f:i:o:p:s:t:') # define flags
    except getopt.GetoptError as err: # check if the arguments are defined
        print str(err) # error
        #usage() # display usage
    for opt, arg in opts: # explore flags
        if opt in '-a':
            alpha_b1 = int(arg)  # e.g.: 60
        if opt in '-b':
            fname_b1_smoothed = arg  # e.g.: b1/b1_smoothed_in_spgr10_space_crop.nii.gz
        if opt in '-c':
            fname_csf_mask = arg  # e.g.: spgr_10_csf_mask.nii.gz
        if opt in '-f':
            flip_angles = arg  # e.g.: 4,10,20,30
        if opt in '-i':
            fname_spgr_data = arg  # e.g.: file_1.nii.gz,file_2.nii.gz,file_3.nii.gz,file_4.nii.gz
        if opt in '-o':
            file_output = arg  # e.g.: PD_map,mtvf_map
        if opt in '-p':
            method = arg  # e.g.: sbs
        if opt in '-s':
            fname_spgr_seg = arg  # e.g.: spgr_10_crop_seg.nii.gz
        if opt in '-t':
            tr = float(arg)  # TR (in s) of the SPGR scans. e.g.: 0.01

    # TODO: assume a b1_map=1 if no input in flag -b

    # Check existence of input mandatory parameters
    if not fname_spgr_data or not flip_angles or not tr or not fname_csf_mask:
        sct.printv('ERROR: mandatory input(s) are missing. See help below.', type='error')
        #usage()
        sys.exit(2)

    # Parse inputs to get the actual data
    flip_angles = np.array([int(x) for x in flip_angles.split(',')])
    #b1_maps = b1_maps.split(',')
    fname_spgr_data = fname_spgr_data.split(',')
    file_output = file_output.split(',')

    # Check data files existence
    sct.printv('\nCheck data files existence...', verbose)
    for fname in fname_spgr_data:
        sct.check_file_exist(fname)
    sct.check_file_exist(fname_csf_mask)
    # if b1_maps != ['']:
    #     for fname in b1_maps:
    #         sct.check_file_exist(fname)
    if fname_b1_smoothed:
        sct.check_file_exist(fname_b1_smoothed)
    if fname_spgr_seg:
        sct.check_file_exist(fname_spgr_seg)
    sct.printv('\tDone.')

    # Check if a flip angle was given as input for each SPGR image
    nb_flip_angles = len(flip_angles)
    sct.printv('\nCheck if a flip angle was associated with each SPGR image according to the input...')
    if nb_flip_angles != len(fname_spgr_data):
        sct.printv('ERROR: the number of flip angles is different from the the number of SPGR images given as input. Exit program.', type='error')
        sys.exit(2)
    sct.printv('\tDone.')

    # Check if dimensions of the SPGR data (T1) are consistent
    sct.printv('\nCheck consistency in dimensions of SPGR data...')
    spgr_nx, spgr_ny, spgr_nz, spgr_nt, spgr_px, spgr_py, spgr_pz, spgr_pt = sct.get_dimension(fname_spgr_data[0])
    for fname in fname_spgr_data[1:]:
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname)
        if (nx, ny, nz) != (spgr_nx, spgr_ny, spgr_nz):
            sct.printv('ERROR: all the SPGR (T1) data don\'t have the same dimensions. Exit program.', type='error')
            sys.exit(2)
    sct.printv('\tDone.')

    # TODO: if seg given, create mask and crop spgr data
    # Load SPGR images
    spgr = np.empty((spgr_nx, spgr_ny, spgr_nz, nb_flip_angles))
    for i_fa in range(0, nb_flip_angles):
        spgr[:, :, :, i_fa] = nib.load(fname_spgr_data[i_fa]).get_data()
    # Record header to generate the final MTV map with the same header
    hdr = nib.load(fname_spgr_data[0]).get_header()

    # Load cord and CSF masks
    csf_mask = np.empty([1], dtype=object)  # initialization to be consistent with the data structure of function 'extract_metric_within_tract'
    csf_mask[0] = nib.load(fname_csf_mask).get_data()
    sc_mask = np.empty([1], dtype=object)
    sc_mask[0] = nib.load(fname_spgr_seg).get_data()

    # Compute B1 scaling map
    if not fname_b1_smoothed or not alpha_b1:
        # If no GRE images are given to estimate B1, then set the B1 scaling map to 1 at every voxel:
        sct.printv('No GRE images or no flip angle were specified to estimate the B1 scaling map so B1 will be assumed homogeneous and the B1 scaling map will be set to 1.', 'warning')
        b1_map_scale = np.ones((spgr_nx, spgr_ny, spgr_nz))
    else:
        # If GRE images are given, estimate the B1 scaling map with the method of double flip angle
        sct.printv('\nB1 scaling map will be estimated using the double flip angle method.\n')
        b1_map_scale = estimate_b1_with_double_flip_angle(fname_b1_smoothed, alpha_b1)  # estimate_b1_with_double_flip_angle(b1_maps, spgr_nx, spgr_ny, spgr_nz, alpha_b1, fname_spgr_data[1])

    # Estimate B1 scale factor
    PD_mean_in_CSF_per_slice = estimate_mean_PD_per_slice_from_mean_in_SPGR_data(spgr, csf_mask, sc_mask, flip_angles, tr)

    # If segmentation of the cord is provided, create a box mask of data surrounding cord and CSF from the segmentation of the cord to reduce the computational time
    if fname_spgr_seg:
        path_spgr, file_first_flip_angle_spgr, ext_spgr = sct.extract_fname(fname_spgr_data[0])
        fname_box_max = path_spgr + 'spgr_cord_box_mask' + ext_spgr
        sct.printv('Create a box mask of data surrounding cord and CSF from the segmentation of the cord to reduce the computational time...')
        sct.run('sct_create_mask -i ' + fname_spgr_data[0] + ' -m centerline,' + fname_spgr_seg + ' -f box -s 40')
        sct.run('mv mask_' + file_first_flip_angle_spgr + ext_spgr + ' ' + fname_box_max)
        # Load this box mask
        box_mask = nib.load(fname_box_max).get_data()
        sct.printv('\tDone.')

    # Estimate PD and T1
    sct.printv('\nCompute PD and T1 maps...')
    with suppress_stdout_stderr():
        PD_map, t1_map = estimate_PD_and_T1(spgr, flip_angles, tr, b1_map_scale, spgr_nx, spgr_ny, spgr_nz, box_mask)
    sct.printv('\tDone.')

    # # Create a CSF mask of data based on the T1 values
    # fname_csf_mask = create_CSF_mask_based_on_t1_map(t1_map, hdr, fname_spgr_seg, fname_spgr_data[0])

    # Estimate proton density mean value in CSF and compute the PD map normalized by PD in CSF
    csf_mask = np.empty([1], dtype=object)  # initialization to be consistent with the data structure of function 'extract_metric_within_tract'
    csf_mask[0] = nib.load(fname_csf_mask).get_data()
    PD_map_normalized_CSF = np.copy(PD_map)  # initialization
    # Normalization by the mean PD in CSF across all slices
    if method == 'estimation-in-whole-CSF':
        PD_mean_in_whole_CSF, PD_std_in_csf = extract_metric_within_tract(PD_map, csf_mask, 'wa', 0)
        PD_map_normalized_CSF = PD_map/PD_mean_in_whole_CSF[0]
    # Normalization by the mean PD in CSF slice-by-slice
    elif method == 'mean-PD-in-CSF-after-estimation-voxel-wize':
        # Estimate the mean +/- std PD in CSF for each slice z
        PD_mean_in_CSF_per_slice_estimate_then_mean = np.empty((spgr_nz))
        for z in range(0, spgr_nz):
            csf_mask_slice = np.empty([1], dtype=object)  # in order to keep compatibility with the function 'extract_metric_within_tract', define a new array for the slice z of the normalizing labels
            csf_mask_slice[0] = csf_mask[0][..., z]
            PD_mean_in_CSF_per_slice_estimate_then_mean[z] = extract_metric_within_tract(PD_map[..., z], csf_mask_slice, 'wa', 0)[0][0]  # estimate the metric mean and std in CSF for the slice z
            # if PD_mean_in_CSF_per_slice[z, 0] > 0:
            #     PD_map_normalized_CSF[..., z] = PD_map[..., z] / PD_mean_in_CSF_per_slice[z, 0]  # divide all the slice z by this value
        # Fit a 2nd order polynomial to the PD value in CSF
        polyfit_bias = np.polyfit(range(0, spgr_nz), PD_mean_in_CSF_per_slice_estimate_then_mean, 2)
        # Compute the corrected value of PD in CSF accroding to the previous fit
        corrected_PD_mean_in_CSF_per_slice = np.array([polyfit_bias[0]*(x**2)+polyfit_bias[1]*x+polyfit_bias[2] for x in range(0, spgr_nz)])
        # Compute the PD map normalized slice-by-slice by the corrected PD means in CSF
        for z in range(0, spgr_nz):
            PD_map_normalized_CSF[..., z] = PD_map[..., z] / corrected_PD_mean_in_CSF_per_slice[z]  # divide all the slice z by the fitted PD in CSF

    elif method == 'mean-PD-in-CSF-from-mean-SPGR':
        # Estimate mean PD per slice in CSF based on the mean signal in SPGR data in CSF
        PD_mean_in_CSF_per_slice = estimate_mean_PD_per_slice_from_mean_in_SPGR_data(spgr, csf_mask, sc_mask, flip_angles, tr)
        polyfit_bias = np.polyfit(range(0, spgr_nz-1), PD_mean_in_CSF_per_slice[:-1], 2)
        corrected_PD_mean_in_CSF_per_slice = np.array([polyfit_bias[0]*(x**2)+polyfit_bias[1]*x+polyfit_bias[2] for x in range(0, spgr_nz)])

        fig = pylab.figure(20)
        pylab.plot(range(0, spgr_nz), PD_mean_in_CSF_per_slice, marker='o', color='b')
        pylab.plot(range(0, spgr_nz), corrected_PD_mean_in_CSF_per_slice, marker='o', color='g')
        #pylab.plot(range(0, spgr_nz), PD_mean_in_CSF_per_slice_estimate_then_mean, marker='o', color='r')
        pylab.legend(['mean then PD estimation', 'corrected PD mean (mean then estimation)'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)
        pylab.grid(True)

        for z in range(0, spgr_nz):
            PD_map_normalized_CSF[..., z] = PD_map[..., z] / corrected_PD_mean_in_CSF_per_slice[z]


    # Compute MTVF map
    MTVF_map = np.ones((spgr_nx, spgr_ny, spgr_nz)) - PD_map_normalized_CSF

    # Generate the T1, PD and MTVF maps as a NIFTI file with the right header
    # done above: path_spgr, file_first_flip_angle_spgr, ext_spgr = sct.extract_fname(fname_spgr_data[0])
    fname_T1_output = path_spgr + file_output[0] + ext_spgr
    fname_PD_output = path_spgr+file_output[1]+ext_spgr
    fname_MTVF_output = path_spgr+file_output[2]+ext_spgr
    sct.printv('Generate the T1, PD and MTVF maps as a NIFTI file with the right header...')
    # Associate the header to the MTVF and PD maps data as a NIFTI file
    T1_map_img_with_hdr = nib.Nifti1Image(t1_map, None, hdr)
    PD_map_img_with_hdr = nib.Nifti1Image(PD_map, None, hdr)
    MTVF_map_img_with_hdr = nib.Nifti1Image(MTVF_map, None, hdr)
    # Save the T1, PD and MTVF maps file
    nib.save(T1_map_img_with_hdr, fname_T1_output)
    nib.save(PD_map_img_with_hdr, fname_PD_output)
    nib.save(MTVF_map_img_with_hdr, fname_MTVF_output)
    sct.printv('\tFile created:\n\t\t\t\t'+fname_T1_output+'\n\t\t\t\t'+fname_PD_output+'\n\t\t\t\t'+fname_MTVF_output)

    # # TODO: TO BE REMOVE (FOR PLOTS)
    # # Plot the mean PD +/- std in CSF slice-by-slice
    # fig1 = pylab.figure(1)
    # fig1.suptitle('Mean PD in CSF by slice', fontsize=20)
    # fig1_ax = fig1.add_subplot(111)
    # fig1_ax.grid(True)
    # fig1_ax.set_xlabel('Slices', fontsize=18)
    # fig1_ax.set_ylabel('Mean PD in CSF', fontsize=18)
    # fig1_ax.errorbar(range(0, spgr_nz), PD_mean_in_CSF_per_slice[:,0], PD_mean_in_CSF_per_slice[:,1], marker='o')
    # fig1_ax.plot(range(0, spgr_nz), corrected_PD_mean_in_CSF_per_slice[:], marker='o', color='g')
    # fig1_ax.axis([-1, spgr_nz + 1, 0, 1200])
    # pylab.show()


#=======================================================================================================================
# Compute B1 field estimation
#=======================================================================================================================
def estimate_b1_with_double_flip_angle(fname_b1_smoothed, nominal_alpha):
    """Compute B1 field estimation."""

    #def estimate_b1_with_double_flip_angle(b1_maps_list, spgr_nx, spgr_ny, spgr_nz, nominal_alpha, fname_spgr10):
    # # Check if user indeed gave 2 images to estimate B1 field
    # if len(b1_maps_list) != 2:
    #     sct.printv('ERROR: You didn\'t provide exactly 2 images to estimate B1 field. Exit program.', type='error')
    #     sys.exit(2)
    #
    # # Check if the 2 images have the same dimensions
    # b1_nx, b1_ny, b1_nz, b1_nt, b1_px, b1_py, b1_pz, b1_pt = sct.get_dimension(b1_maps_list[0])
    # b1_nx2, b1_ny2, b1_nz2, b1_nt2, b1_px2, b1_py2, b1_pz2, b1_pt2 = sct.get_dimension(b1_maps_list[1])
    # if (b1_nx, b1_ny, b1_nz) != (b1_nx2, b1_ny2, b1_nz2):
    #     sct.printv('ERROR: The 2 images to estimate B1 field have not the same dimensions. Exit program.', type='error')
    #     sys.exit(2)
    #
    # # Compute half-ratio of the two images
    # sct.printv('\nCompute the half ratio of the 2 images given as input to estimate B1 field...')
    # path_b1_map, file_b1_map, ext_b1_map = sct.extract_fname(b1_maps_list[0])
    # fname_ratio = path_b1_map + 'b1_maps_half_ratio' + ext_b1_map
    # sct.run('fslmaths -dt double ' + b1_maps_list[0] + ' -div 2 -div ' + b1_maps_list[1] + ' ' + fname_ratio)
    # sct.printv('\tDone.--> ' + fname_ratio)

    # Load image
    half_ratio = nib.load(fname_b1_smoothed).get_data()

    # Smooth B1 profile
    #half_ratio_smoothed = scipy.ndimage.gaussian_filter(half_ratio, sigma=1.0, order=0)

    # Estimate flip angle (alpha)
    measured_alpha = (180/np.pi)*np.arccos(half_ratio)  # some values are out of the definition field of arccos

    # Divide the measured fip angle by the nominal flip angle to get a kind of B1 scale
    b1_map_scale = measured_alpha/nominal_alpha

    return b1_map_scale

#=======================================================================================================================
# Compute the mean PD per slice based on the mean signal in SPGR data per slice in CSF
#=======================================================================================================================
def estimate_mean_PD_per_slice_from_mean_in_SPGR_data(spgr, csf_mask, sc_mask, flip_angles, tr):

    # Record dimensions of data
    (nx, ny, nb_slices, nb_flip_angles) = spgr.shape
    # Initialization of the matrices for CSF and SC that will contain:
    #   - 1st axis: slices
    #   - 2nd axis: flip angles
    mean_signal_in_CSF = np.zeros((nb_slices, nb_flip_angles))
    mean_signal_in_SC = np.zeros((nb_slices, nb_flip_angles))
    # Initialization of matrices for CSF and SC that will contain the slope
    slope_CSF = np.zeros((nb_slices))
    slope_SC = np.zeros((nb_slices))
    intercept_CSF = np.zeros((nb_slices))
    intercept_SC = np.zeros((nb_slices))

    for z in range(0, nb_slices):
        for theta in range(0, nb_flip_angles):

            # Extract the slice in the correct structure
            CSF_mask_slice = np.empty((1), dtype=object)
            CSF_mask_slice[0] = csf_mask[0][:, :, z]
            SC_mask_slice = np.empty((1), dtype=object)
            SC_mask_slice[0] = sc_mask[0][:, :, z]
            # Extract signal mean
            mean_signal_in_CSF[z, theta] = extract_metric_within_tract(spgr[:, :, z, theta], CSF_mask_slice, 'wa', 0)[0][0]
            mean_signal_in_SC[z, theta] = extract_metric_within_tract(spgr[:, :, z, theta], SC_mask_slice, 'wa', 0)[0][0]

        # Plot the signal equation
        fig_eq = pylab.figure(z)

        y_CSF = np.divide(mean_signal_in_CSF[z, :], np.sin(flip_angles*(np.pi/180)))
        y_SC = np.divide(mean_signal_in_SC[z, :], np.sin(flip_angles*(np.pi/180)))
        x_CSF = np.divide(mean_signal_in_CSF[z, :], np.tan(flip_angles*(np.pi/180)))
        x_SC = np.divide(mean_signal_in_SC[z, :], np.tan(flip_angles*(np.pi/180)))

        pylab.plot(x_CSF, y_CSF, marker='o', color='b')
        pylab.plot(x_SC, y_SC, marker='o', color='r')

        pylab.legend(['CSF', 'cord'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)

        # Fit the data
        [slope_CSF[z], intercept_CSF[z]] = np.polyfit(x_CSF, y_CSF, 1)
        [slope_SC[z], intercept_SC[z]] = np.polyfit(x_SC, y_SC, 1)

    # Estimate the PD for each slice starting from this fitting
    PD_mean_in_CSF = np.divide(intercept_CSF,(1 - slope_CSF))
    PD_mean_in_SC = np.divide(intercept_SC,(1 - slope_SC))

    # Plot
    fig_PD = pylab.figure(z+10)
    fig_PD.suptitle('PD mean per slice')
    pylab.grid(True)

    pylab.plot(range(0, nb_slices), PD_mean_in_CSF, marker='o', color='b')
    pylab.plot(range(0, nb_slices), PD_mean_in_SC, marker='o', color='r')

    pylab.legend(['CSF', 'cord'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)

    pylab.show(block=False)


    return PD_mean_in_CSF

#=======================================================================================================================
# Estimate the proton density and T1 by Fram's method (1987)
#=======================================================================================================================
def estimate_PD_and_T1(spgr, flip_angles, tr, b1_map_scale, nx, ny, nz, box_mask=None):
    """Estimate the proton density map and the T1 maps using linear regression presented by Fram (1987)"""

    # Initialization of the maps to be estimated
    PD_map = np.zeros((nx, ny, nz))
    t1_map = np.zeros((nx, ny, nz))

    ## Create mask from the input 'spgr_seg' to reduce the computational time by reducing the ROI to cord

    for i in range(0, nx):
        for j in range(0, ny):
            for k in range(0, nz):
                if (((box_mask is not None) and (box_mask[i, j, k] == 1)) or (box_mask is None)):

                    y = np.divide(spgr[i, j, k, :], np.sin(flip_angles*(np.pi/180)*b1_map_scale[i, j, k]))
                    x = np.divide(spgr[i, j, k, :], np.tan(flip_angles*(np.pi/180)*b1_map_scale[i, j, k]))

                    linear_regression = np.polyfit(x, y, 1)
                    slope = linear_regression[0]
                    intercep = linear_regression[1]

                    if slope > 0:
                        t1 = -tr/math.log(slope)
                        if t1 > 20:
                            t1 = 20
                    else:  # due to noise or bad fitting
                        t1 = 0.000000000000001

                    t1_map[i, j, k] = t1
                    PD_map[i, j, k] = intercep/(1 - math.exp(-tr/t1))

    return PD_map, t1_map


#=======================================================================================================================
# Create a CSF mask of PD map based on the T1 map
#=======================================================================================================================
def create_CSF_mask_based_on_t1_map(t1_map, hdr, fname_spgr_seg, fname_spgr):
    ''' Create CSF mask based on the T1 map, keeping an interval between 3.5 and 5'''

    # First create a box mask surrounding thecord and CSF
    path_spgr, file_first_flip_angle_spgr, ext_spgr = sct.extract_fname(fname_spgr)
    fname_box_max_output = path_spgr + 'spgr_cord_box_mask' + ext_spgr
    sct.run('sct_create_mask -i ' + fname_spgr + ' -m centerline,' + fname_spgr_seg + ' -f box -s 40')
    sct.run('mv mask_' + file_first_flip_angle_spgr + ext_spgr + ' ' + fname_box_max_output)
    # Load this box mask
    box_max = nib.load(fname_box_max_output).get_data()
    # Create the CSF mask based on an interval of T1 values (3.5;5)
    CSF_mask_data = np.zeros(t1_map.shape)
    for i in range(0, t1_map.shape[0]):
        for j in range(0, t1_map.shape[1]):
            for k in range(0, t1_map.shape[2]):
                if box_max[i, j, k] == 1 and 3.5 < t1_map[i, j, k] < 5.0:
                    CSF_mask_data[i, j, k] = 1
    # Generate the CSF mask that has just been created as a NIFTI file
    fname_CSF_mask_output = path_spgr + 't1_based_CSF_mask' + ext_spgr
    sct.printv('Generate a CSF mask based on the T1 map...')
    # Associate the header to the MTVF and PD maps data as a NIFTI file
    CSF_mask_img_with_hdr = nib.Nifti1Image(CSF_mask_data, None, hdr)
    # Save the T1, PD and MTVF maps file
    nib.save(CSF_mask_img_with_hdr, fname_CSF_mask_output)
    sct.printv('\tFile created:\n\t\t\t\t' + fname_CSF_mask_output)

    return fname_CSF_mask_output


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # Construct object fro class 'param'
    param = param()
    # Call main function
    main()
