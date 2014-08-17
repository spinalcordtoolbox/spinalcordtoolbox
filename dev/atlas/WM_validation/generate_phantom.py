#!/usr/bin/env python

# Generates a phantom from an atlas of tracts
# Saves the phantom to a niftii image afterwards

## Import common Python libraries
import os
import getopt
import sys
import time
import glob
import re
#Modif outside toolobx
import sct_utils as sct
#import sct_utilsc as sct
import numpy
import math
import random
import nibabel
#from nifti import *
try:
  #library of nifty format for read data and tracts files
  from nibabel import *
except ImportError:
   print '--- nibabel not installed! Exit program. ---'
   sys.exit(2)
try:
   # library of calculations and processing matrices
   from numpy import mean, asarray, std, zeros, sum, ones, dot, eye, sqrt, empty, size, linspace, abs, amin, argmin, concatenate, array
   from numpy.linalg import solve,pinv
except ImportError:
   print '--- numpy not installed! Exit program. ---'
   sys.exit(2)

# Phantom parameters
true_value = 40
std_noise = 5
range_tract = 0
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
    save_3D_nparray_niftii(synthetic_vol, WM_phantom)
    # Save the phantom with added noise to niftii image
    save_3D_nparray_niftii(synthetic_voln, WM_phantom_noise)
    # Save the sum of the tracts to nifti image
    save_3D_nparray_niftii(tracts_sum, tracts_sum_img)
    # adjust geometry between saved images and tracts
    fname_tract = glob.glob(folder_atlas + '/*' + '.nii.gz')
    sct.run('fslcpgeom ' + fname_tract[0] + ' ' + WM_phantom)
    sct.run('fslcpgeom ' + fname_tract[0] + ' ' + WM_phantom_noise)
    sct.run('fslcpgeom ' + fname_tract[0] + ' ' + tracts_sum_img)
#=======================================================================================================================
# phantom generation
#=======================================================================================================================

def phantom_generation(tracts, std_noise, range_tract, true_value):
   
    # Transform std noise and range tract to a percentage of the true value
    range_tract = float(range_tract) / 100 * true_value
    std_noise = float(std_noise) / 100 * true_value

    # Generate synthetic Volume  
    numtracts = len(tracts)  
    nx = zeros(numtracts)
    ny = zeros(numtracts)
    nz = zeros(numtracts)

    for i in range(0, numtracts):
        [nx[i], ny[i], nz[i]] = (tracts[i, 0]).shape
    nx = nx[0]
    ny = ny[0]
    nz = nz[0]
    
    tracts_weighted = zeros([numtracts,nx, ny, nz])
    synthetic_vol = zeros([nx, ny, nz])
    synthetic_voln = zeros([nx, ny, nz])
    tracts_sum = zeros([nx, ny, nz])
    values_synthetic_data = zeros([numtracts])
    for i in range(0, numtracts):
        tracts_sum = tracts_sum + tracts[i,0]
        values_synthetic_data[i] = (true_value - range_tract + random.uniform(0,2*range_tract))
        # print '\t label \t' + str(i) + '\ttrue value\t' + str(round(values_synthetic_data[i],3))
        tracts_weighted[i,:,:,:] = values_synthetic_data[i] * tracts[i,0]
        synthetic_vol = synthetic_vol + tracts_weighted[i,:,:,:]
    synthetic_voln = synthetic_vol + numpy.random.normal(loc=0.0,scale=std_noise,size=(nx, ny, nz))

    return[synthetic_vol, synthetic_voln, values_synthetic_data, tracts_sum]

#=======================================================================================================================
# Save 3D numpy array to a niftii
#=======================================================================================================================
def save_3D_nparray_niftii(np_matrix_3d, output_image):
    # Save 3d numpy matrix to niftii image
    # np_matrix_3d is a 3D numpy ndarray
    # output_image is the name of the niftii image created, ex: '3D_matrix.nii.gz'
    img = Nifti1Image(np_matrix_3d, numpy.eye(4))
    affine = img.get_affine()
    np_matrix_3d_nii = Nifti1Image(np_matrix_3d,affine)
    nibabel.save(np_matrix_3d_nii, output_image)
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
    tracts = empty([len(fname_tract), 1], dtype=object)
    
    #Load each partial volumes of each tracts
    for label in range(0, len(fname_tract)):
       tracts[label, 0] = load(fname_tract[label]).get_data()
    
    #Reshape tracts if it is the 2D image instead of 3D
    for label in range(0, len(fname_tract)):
       if (tracts[label,0]).ndim == 2:
           tracts[label,0] = tracts[label,0].reshape(int(size(tracts[label,0],0)), int(size(tracts[label,0],1)),1)
    return tracts

if __name__ == "__main__":
    main()
