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
import commands
# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct
from msct_parser import *
from sct_extract_metric import extract_metric_within_tract


class Param:
    def __init__(self):
        self.debug = 0
        self.verbose = 1
        self.file_output = 'T1_map,PD_map,MTVF_map'  # file name of the PD and MTVF maps to be generated as ouptut
        self.method = 'mean-PD-in-CSF-from-mean-SPGR'  # method used to compute MTVF for the division by the proton density in CSF


# # Define a context manager to suppress stdout and stderr, in order to suppress warnings when running the function 'polyfit' in function 'estimate_PD_and_T1'
# class suppress_stdout_stderr(object):
#     '''
#     A context manager for doing a "deep suppression" of stdout and stderr in
#     Python, i.e. will suppress all print, even if the print originates in a
#     compiled C/Fortran sub-function.
#        This will not suppress raised exceptions, since exceptions are printed
#     to stderr just before a script exits, and after the context manager has
#     exited (at least, I think that is why it lets exceptions through).
#
#     '''
#     def __init__(self):
#         # Open a pair of null files
#         self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
#         # Save the actual stdout (1) and stderr (2) file descriptors.
#         self.save_fds = (os.dup(1), os.dup(2))
#
#     def __enter__(self):
#         # Assign the null pointers to stdout and stderr.
#         os.dup2(self.null_fds[0],1)
#         os.dup2(self.null_fds[1],2)
#
#     def __exit__(self, *_):
#         # Re-assign the real stdout/stderr back to (1) and (2)
#         os.dup2(self.save_fds[0],1)
#         os.dup2(self.save_fds[1],2)
#         # Close the null files
#         os.close(self.null_fds[0])
#         os.close(self.null_fds[1])


#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization of variables
    alpha_b1 = None
    flip_angles = ''
    fname_csf_mask = ''
    fname_b1_smoothed = ''
    file_output = Param.file_output
    method = Param.method
    fname_spgr_seg = ''
    tr = ''
    box_mask = None
    verbose = Param.verbose

    # Check input Parameters
    parser = Parser(__file__)
    parser.usage.set_description('compute MTV')
    parser.add_option("-b", "file", "alpha_b1map_smooth", True, "b1/b1_smoothed_in_spgr10_space_crop.nii.gz")
    parser.add_option("-c", "file", "CSF mask", False, "spgr10_crop_csf_mask.nii.gz")
    parser.add_option("-f", "str", "Flip angles", True, "4,10,20,30")
    parser.add_option("-o", "str", "output file name", True, "T1_map,PD_map,MTVF_map")
    parser.add_option("-i", "str", "fname_spgr_data", True, "spgr5to10.nii.gz,spgr10_crop.nii.gz,spgr20to10.nii.gz,spgr30to10.nii.gz")
    parser.add_option("-t", "float", "TR (in s) of the SPGR scans", True, 0.01)
    parser.add_option("-s", "file", "segmentation", True, 0.01)
    usage = parser.usage.generate()

    # Parameters for debug mode
    if Param.debug:
        working_dir = '/Volumes/users_hd2/tanguy/data/Boston/2014-07/Connectome/MS_SC_002/MTV'
        os.chdir(working_dir)
        sct.printv('\n*** WARNING: DEBUG MODE ON ***\n\t\tWorking directory: '+working_dir, type='warning')
        flip_angles = '4,10,30'
        fname_csf_mask = 'spgr10-csf.nii.gz'
        fname_b1_smoothed = 'B1angle_reg.nii'
        fname_spgr_data = 'spgr4.nii.gz,spgr10.nii.gz,spgr30.nii.gz'
        file_output = 'T1_map,PD_map,MTVF_map'
        method = 'mean-PD-in-CSF-from-mean-SPGR'
        fname_spgr_seg = 'spgr10_seg.nii.gz'
        tr = float(0.01)
    else:
        arguments = parser.parse(sys.argv[1:])
        fname_b1_smoothed = arguments["-b"]  # e.g.: b1/b1_smoothed_in_spgr10_space_crop.nii.gz
        fname_csf_mask = arguments["-c"]  # e.g.: spgr_10_csf_mask.nii.gz
        flip_angles = arguments["-f"]  # e.g.: 4,10,20,30
        fname_spgr_data = arguments["-i"]  # e.g.: file_1.nii.gz,file_2.nii.gz,file_3.nii.gz,file_4.nii.gz
        file_output = arguments["-o"]  # e.g.: PD_map,mtvf_map
        method = 'mean-PD-in-CSF-from-mean-SPGR'  # e.g.: sbs
        fname_spgr_seg = arguments["-s"]  # e.g.: spgr_10_crop_seg.nii.gz
        tr = float(arguments["-t"])  # TR (in s) of the SPGR scans. e.g.: 0.01

    # Check existence of input mandatory Parameters
    if not fname_spgr_data or not flip_angles or not tr or not fname_csf_mask:
        sct.printv('ERROR: mandatory input(s) are missing. See help below.', type='error')
        #usage()
        sys.exit(2)

    # Parse inputs to get the actual data
    flip_angles = np.array([int(x) for x in flip_angles.split(',')])
    #b1_maps = b1_maps.split(',')
    fname_spgr_data = fname_spgr_data.split(',')
    file_output = file_output.split(',')


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
    if not fname_b1_smoothed:
        # If no GRE images are given to estimate B1, then set the B1 scaling map to 1 at every voxel:
        sct.printv('No GRE images or no flip angle were specified to estimate the B1 scaling map so B1 will be assumed homogeneous and the B1 scaling map will be set to 1.', 'warning')
        b1_map_scale = np.ones((spgr_nx, spgr_ny, spgr_nz))
    else:
        # If GRE images are given, estimate the B1 scaling map with the method of double flip angle
        sct.printv('\nB1 scaling map will be estimated using the double flip angle method.\n')
        b1_map_scale = nib.load(fname_b1_smoothed).get_data() # estimate_b1_with_double_flip_angle(b1_maps, spgr_nx, spgr_ny, spgr_nz, alpha_b1, fname_spgr_data[1])

    # # If segmentation of the cord is provided, create a box mask of data surrounding cord and CSF from the segmentation of the cord to reduce the computational time
    # if fname_spgr_seg:
    #     path_spgr, file_first_flip_angle_spgr, ext_spgr = sct.extract_fname(fname_spgr_data[0])
    #     fname_box_max = path_spgr + 'spgr_cord_box_mask' + ext_spgr
    #     sct.printv('Create a box mask of data surrounding cord and CSF from the segmentation of the cord to reduce the computational time...')
    #     sct.run('sct_create_mask -i ' + fname_spgr_data[0] + ' -m centerline,' + fname_spgr_seg + ' -f box -s 40')
    #     sct.run('mv mask_' + file_first_flip_angle_spgr + ext_spgr + ' ' + fname_box_max)
    #     # Load this box mask
    #     box_mask = nib.load(fname_box_max).get_data()
    #     sct.printv('\tDone.')









    #===================================================================================================================
    # SCRIPT START HERE
    #===================================================================================================================

    # Estimate PD and T1
    sct.printv('\nCompute PD and T1 maps...')
    #with suppress_stdout_stderr():
    PD_map, t1_map, recorder_unconsistent_voxels = estimate_PD_and_T1(spgr, flip_angles, tr, b1_map_scale, spgr_nx, spgr_ny, spgr_nz) #estimate_PD_and_T1(spgr, flip_angles, tr, b1_map_scale, spgr_nx, spgr_ny, spgr_nz, box_mask)
    sct.printv('\nNumber of voxels with unconsistent values: '+str(recorder_unconsistent_voxels))
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

    # Normalization slice-by-slice by the mean PD in CSF
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

    # Normalization slice-by-slice by the mean PD in CSF estimated based on mean signal in SPGR data in CSF
    elif method == 'mean-PD-in-CSF-from-mean-SPGR':
        # Estimate mean PD per slice in CSF based on the mean signal in SPGR data in CSF
        PD_mean_in_CSF_per_slice, PD_mean_in_SC_per_slice = estimate_mean_PD_per_slice_from_mean_in_SPGR_data(spgr, csf_mask, sc_mask, flip_angles, tr)
        polyfit_bias = np.polyfit(range(1, spgr_nz-1), PD_mean_in_CSF_per_slice[1:-1], 2)
        corrected_PD_mean_in_CSF_per_slice = np.array([polyfit_bias[0]*(x**2)+polyfit_bias[1]*x+polyfit_bias[2] for x in range(0, spgr_nz)])

        fig_mean_PD_correction = plt.figure(2)

        fig_mean_PD_correction.suptitle('Estimation from mean SPGR and correction by fitting')

        ax_mean_PD_CSF_correction = fig_mean_PD_correction.add_subplot(121, title='Mean PD estimation in CSF')
        ax_mean_PD_CSF_correction.plot(range(0, spgr_nz), PD_mean_in_CSF_per_slice, marker='o', color='b')
        ax_mean_PD_CSF_correction.plot(range(0, spgr_nz), corrected_PD_mean_in_CSF_per_slice, marker='o', color='g')
        ax_mean_PD_CSF_correction.legend(['No correction', 'Correction by fitting'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)
        ax_mean_PD_CSF_correction.grid(True)

        ax_mean_MTV_correction = fig_mean_PD_correction.add_subplot(122, title='Mean MTV estimation in cord and CSF')
        MTV_mean_in_CSF_per_slice_corrected = 1 - np.divide(corrected_PD_mean_in_CSF_per_slice, corrected_PD_mean_in_CSF_per_slice)
        MTV_mean_in_SC_per_slice_corrected = 1 - np.divide(PD_mean_in_SC_per_slice, corrected_PD_mean_in_CSF_per_slice)
        ax_mean_MTV_correction.plot(range(0, spgr_nz), MTV_mean_in_CSF_per_slice_corrected, marker='o', color='b')
        ax_mean_MTV_correction.plot(range(0, spgr_nz), MTV_mean_in_SC_per_slice_corrected, marker='o', color='r')
        ax_mean_MTV_correction.legend(['CSF', 'Cord'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)
        ax_mean_MTV_correction.grid(True)



        for z in range(0, spgr_nz):
            PD_map_normalized_CSF[..., z] = PD_map[..., z] / corrected_PD_mean_in_CSF_per_slice[z]


    # Compute MTVF map
    MTVF_map = np.ones((spgr_nx, spgr_ny, spgr_nz)) - PD_map_normalized_CSF

    # Generate the T1, PD and MTVF maps as a NIFTI file with the right header
    path_spgr, file_first_flip_angle_spgr, ext_spgr = sct.extract_fname(fname_spgr_data[0])
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



#=======================================================================================================================
# Compute the mean PD per slice based on the mean signal in SPGR data per slice
#=======================================================================================================================
def estimate_mean_PD_per_slice_from_mean_in_SPGR_data(spgr, csf_mask, sc_mask, flip_angles, tr):
    """Compute the mean PD per slice based on the mean signal in SPGR data per slice."""

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

        # compute the y and x of the equation per slice based on the mean of SPGR data, separately in cord and CSF
        y_CSF = np.divide(mean_signal_in_CSF[z, :], np.sin(flip_angles*(np.pi/180)))
        y_SC = np.divide(mean_signal_in_SC[z, :], np.sin(flip_angles*(np.pi/180)))
        x_CSF = np.divide(mean_signal_in_CSF[z, :], np.tan(flip_angles*(np.pi/180)))
        x_SC = np.divide(mean_signal_in_SC[z, :], np.tan(flip_angles*(np.pi/180)))

        # Fit the data
        [slope_CSF[z], intercept_CSF[z]] = np.polyfit(x_CSF, y_CSF, 1)
        [slope_SC[z], intercept_SC[z]] = np.polyfit(x_SC, y_SC, 1)

    # Estimate the PD for each slice starting from this fitting
    PD_mean_in_CSF = np.divide(intercept_CSF, (1 - slope_CSF))
    PD_mean_in_SC = np.divide(intercept_SC, (1 - slope_SC))

    # Compute MTV
    MTV_mean_in_CSF_per_slice = 1 - np.divide(PD_mean_in_CSF, PD_mean_in_CSF)
    MTV_mean_in_SC_per_slice = 1 - np.divide(PD_mean_in_SC, PD_mean_in_CSF)

    # Plot
    fig_PD_mean_CSF = plt.figure(1)
    fig_PD_mean_CSF.suptitle('Estimation slice-wise from mean SPGR in cord and CSF')

    ax_PD_mean_CSF = fig_PD_mean_CSF.add_subplot(121, title='Mean PD per slice estimated from mean SPGR in cord and CSF per slice')
    ax_PD_mean_CSF.grid(True)
    ax_PD_mean_CSF.plot(range(0, nb_slices), PD_mean_in_CSF, marker='o', color='b')
    ax_PD_mean_CSF.plot(range(0, nb_slices), PD_mean_in_SC, marker='o', color='r')
    ax_PD_mean_CSF.legend(['CSF', 'cord'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)
    ax_PD_mean_CSF.set_xlabel('Slices')

    ax_MTV_from_mean_SPGR = fig_PD_mean_CSF.add_subplot(122, title='Mean MTV per slice estimated from mean SPGR in cord and CSF per slice')
    ax_MTV_from_mean_SPGR.grid(True)
    ax_MTV_from_mean_SPGR.plot(range(0, nb_slices), MTV_mean_in_CSF_per_slice, marker='o', color='b')
    ax_MTV_from_mean_SPGR.plot(range(0, nb_slices), MTV_mean_in_SC_per_slice, marker='o', color='r')
    ax_MTV_from_mean_SPGR.legend(['CSF', 'cord'], loc=2, handler_map={lgd.Line2D: lgd.HandlerLine2D(numpoints=1)}, fontsize=18)
    ax_MTV_from_mean_SPGR.set_xlabel('Slices')


    return PD_mean_in_CSF, PD_mean_in_SC

#=======================================================================================================================
# Estimate the proton density and T1 by Fram's method (1987)
#=======================================================================================================================
def estimate_PD_and_T1(spgr, flip_angles, tr, b1_map_scale, nx, ny, nz, box_mask=None):
    """Estimate the proton density map and the T1 maps using linear regression presented by Fram (1987)"""

    # Initialization of the maps to be estimated
    PD_map = np.zeros((nx, ny, nz))
    t1_map = np.zeros((nx, ny, nz))

    # Compute PD and T1 voxel-wize
    recorder_vox_out = 0  # recorder of the number of voxels with values out of range

    for k in range(1, nz):
        for j in range(1, ny):
            for i in range(1, nx):

                if (((box_mask is not None) and (box_mask[i, j, k] == 1)) or (box_mask is None)) and not (b1_map_scale[i, j, k] == 0) and not (np.max(spgr[i, j, k, :]) == 0):

                    y = np.divide(spgr[i, j, k, :], np.sin(flip_angles*(np.pi/180)*b1_map_scale[i, j, k]))
                    x = np.divide(spgr[i, j, k, :], np.tan(flip_angles*(np.pi/180)*b1_map_scale[i, j, k]))

                    linear_regression = np.polyfit(x, y, 1)
                    slope = linear_regression[0]
                    intercep = linear_regression[1]

                    if slope > 0:
                        t1 = -tr/math.log(slope)
                    else:  # due to noise or bad fitting
                        t1 = 0.000000000000001
                        recorder_vox_out += 1

                    t1_map[i, j, k] = t1
                    PD_map[i, j, k] = intercep/(1 - math.exp(-tr/t1))

    return PD_map, t1_map, recorder_vox_out


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
    # Construct object fro class 'Param'
    Param = Param()
    # Call main function
    main()

    # if os.fork():
    #     # Parent
    #     pass
    # else:
    #     # Child
    #     main()
