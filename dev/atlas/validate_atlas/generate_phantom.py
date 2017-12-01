#!/usr/bin/env python

# Generates a phantom from an atlas of tracts
# Saves the phantom to a niftii image afterwards

## Import common Python libraries
# import os
# import getopt
# import sys
# import time
import glob
# import re
#Modif outside toolobx
import sct_utils as sct
#import sct_utilsc as sct
import numpy as np
# import math
import random
import nibabel as nib
# from numpy import mean, asarray, std, zeros, sum, ones, dot, eye, sqrt, empty, size, linspace, abs, amin, argmin, concatenate, array
# from numpy.linalg import solve,pinv


# Phantom parameters
true_value = 40
std_noise = 0
range_tract = 5
#folder_atlas = "/home/django/cnaaman/data/data_marc/WMtracts_outputs/final_results/"
#folder_atlas = "/home/django/cnaaman/code/stage/cropped_atlas"
#folder_atlas = "/home/django/cnaaman/data/data_marc/WMtracts_outputstest/final_results"
folder_atlas = "/home/django/cnaaman/data/data_marc/WMtracts_outputsc_julien/final_results"
#Generated phantom without noise
WM_phantom = 'coplete_WM_phantom.nii.gz'       
WM_phantom_noise = 'complete_WM_phantom_noise.nii.gz'
tracts_sum_img = 'tracts_sum.nii.gz'
def main():
    
    #Generated phantom with added noise
    # Extract the tracts from the atlas' folder
    tracts = get_tracts(folder_atlas)
    # Generate the phantom
    [synthetic_vol, synthetic_voln, values_synthetic_data, tracts_sum] = phantom_generation(tracts, std_noise, range_tract, true_value)
    # Save the phantom without added noise to niftii image
    save_3D_nparray_nifti(synthetic_vol, WM_phantom)
    # Save the phantom with added noise to niftii image
    save_3D_nparray_nifti(synthetic_voln, WM_phantom_noise)
    # Save the sum of the tracts to nifti image
    save_3D_nparray_nifti(tracts_sum, tracts_sum_img)
    # adjust geometry between saved images and tracts
    fname_tract = glob.glob(os.path.join(folder_atlas, "*.nii.gz"))
    sct.run('fslcpgeom ' + fname_tract[0] + ' ' + WM_phantom)
    sct.run('fslcpgeom ' + fname_tract[0] + ' ' + WM_phantom_noise)
    sct.run('fslcpgeom ' + fname_tract[0] + ' ' + tracts_sum_img)
#=======================================================================================================================
# phantom generation
#=======================================================================================================================

def phantom_generation(tracts, std_noise_perc, range_tract_perc, true_value, folder_out, value_gm, value_csf):
    """
    :param tracts: np array
    :param std_noise: std of noise to generate pseudo-random gaussianly-distributed noise
    :param range_tract: range of value to generate pseudo-random uniformly-distributed tract value
    :param true_value: true value of the tract
    :param folder_out: output folder
    :param zero_last_tract: assign value 0 to last tract (corresponds to CSF)
    :return: synthetic_vol, synthetic_voln, values_synthetic_data, tracts_sum
    """

    fname_phantom = os.path.join(folder_out, 'phantom_values.txt')
    #value_gm = 30
    #value_csf = 0

    # Transform std noise and range tract to a percentage of the true value
    range_tract = float(range_tract_perc) / 100 * true_value
    std_noise = float(std_noise_perc) / 100 * true_value

    # Generate synthetic Volume  
    numtracts = len(tracts)
    nx, ny, nz = tracts[0][0].shape

   # open txt file that includes true values per tract
    fid_file = open(fname_phantom, 'w+')
    print >> fid_file, 'std_noise='+str(std_noise_perc)+'%, range_tract='+str(range_tract_perc)+'%, true_value='+str(true_value)

    # create volume of tracts with randomly-assigned values
    tracts_weighted = np.zeros([numtracts, nx, ny, nz])
    synthetic_vol = np.zeros([nx, ny, nz])
    synthetic_voln = np.zeros([nx, ny, nz])
    values_synthetic_data = np.zeros([numtracts])
    for i in range(0, numtracts):
        if i == numtracts-2:
            values_synthetic_data[i] = value_gm
        elif i == numtracts-1:
            values_synthetic_data[i] = value_csf
        else:
            values_synthetic_data[i] = (true_value - range_tract + random.uniform(0, 2*range_tract))  # true_value - range_tract <= values_synthetic_data <= true_value + range_tract
        print >> fid_file, 'label=' + str(i) + ', value=' + str(values_synthetic_data[i])
        tracts_weighted[i, :, :, :] = values_synthetic_data[i] * tracts[i, 0]
        # add tract to volume
        synthetic_vol = synthetic_vol + tracts_weighted[i, :, :, :]

    # sum all tracts
    tracts_sum = np.zeros([nx, ny, nz])
    for i in range(0, numtracts):
        tracts_sum = tracts_sum + tracts[i, 0]

    # close txt file
    fid_file.close()

    # add gaussian noise
    if not std_noise == 0:
        synthetic_voln = synthetic_vol + np.random.normal(loc=0, scale=std_noise, size=(nx, ny, nz))
    else:
        synthetic_voln = synthetic_vol

    return[synthetic_vol, synthetic_voln, values_synthetic_data, tracts_sum]


#=======================================================================================================================
# Save 3D numpy array to a nifti
#=======================================================================================================================
def save_3D_nparray_nifti(np_matrix_3d, output_image, fname_atlas):
    """
    Save 3d numpy matrix to niftii image
    :param np_matrix_3d: 3d numpy array
    :param output_image:  name of the created nifti file. e.g.; 3d_matrix.nii.gz
    :param fname_atlas: name of the file to copy nifti header from
    :return: nonw
    """
    img = nib.Nifti1Image(np_matrix_3d, np.eye(4))
    affine = img.get_affine()
    np_matrix_3d_nii = nib.Nifti1Image(np_matrix_3d,affine)
    nib.save(np_matrix_3d_nii, output_image)
    # copy geometric information
    sct.run('fslcpgeom '+fname_atlas+' '+output_image, verbose=0)
    return None


#=======================================================================================================================
# Get tracts 
#=======================================================================================================================
def get_tracts(tracts_folder):
    # Loads tracts in an atlas folder and converts them from .nii.gz format to numpy ndarray 
    # Save path of each tracts
    # Only the tract must be in tracts_format in the folder
    fname_tract = glob.glob(tracts_folder + '/*' + '.nii.gz')
    
    #Initialise tracts variable as object because there are 4 dimensions
    tracts = np.empty([len(fname_tract), 1], dtype=object)
    
    #Load each partial volumes of each tracts
    for label in range(0, len(fname_tract)):
       tracts[label, 0] = nib.load(fname_tract[label]).get_data()
    
    #Reshape tracts if it is the 2D image instead of 3D
    for label in range(0, len(fname_tract)):
       if (tracts[label,0]).ndim == 2:
           tracts[label,0] = tracts[label,0].reshape(int(np.size(tracts[label,0],0)), int(np.size(tracts[label,0],1)),1)
    return tracts

if __name__ == "__main__":
    main()
