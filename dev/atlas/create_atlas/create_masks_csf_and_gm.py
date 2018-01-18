#!/usr/bin/env python
# create masks of CSF and gray matter
# Author: jcohen@polymtl.ca
# Created: 2014-12-06

# TODO: get GM
# TODO: add tract corresponding to the undefined values in WM atlas

import sys, io, os, glob

import numpy as np
import nibabel as nib

path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, 'scripts'))
import sct_utils as sct

# parameters
tracts_to_sum_index = 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
folder_atlas = os.path.join("WMtracts_outputs", "final_results")
file_csf = "WMtract__csf.nii.gz"
file_gm = "WMtract__gm.nii.gz"
file_label = 'info_label.txt'


def main():
    # Extract the tracts from the atlas' folder
    tracts = get_tracts(folder_atlas)
    nb_tracts = len(tracts)
    # Get the sum of the tracts
    tracts_sum = add_tracts(tracts, tracts_to_sum_index)
    # Save sum of the tracts to niftii
    save_3D_nparray_nifti(tracts_sum, 'tmp.WM_all.nii.gz', os.path.join(folder_atlas, "WMtract__00.nii.gz"))
    # binarize it
    sct.run('fslmaths tmp.WM_all.nii.gz  -thr 0.5 -bin tmp.WM_all_bin.nii.gz')
    # dilate it
    sct.run('fslmaths tmp.WM_all_bin.nii.gz  -kernel boxv 5x5x1 -dilM tmp.WM_all_bin_dil.nii.gz')
    # subtract WM mask to obtain CSF mask
    sct.run('fslmaths tmp.WM_all_bin_dil -sub tmp.WM_all '+os.path.join(os.path.join(folder_atlas, file_csf)))
    # add line in info_label.txt
    text_label = '\n'+str(nb_tracts)+', CSF, '+file_csf
    io.open(os.path.join(folder_atlas, file_label) 'a+b').write(text_label)

def get_tracts(tracts_folder):
    """Loads tracts in an atlas folder and converts them from .nii.gz format to numpy ndarray 
    Save path of each tracts
    Only the tract must be in tracts_format in the folder"""
    fname_tract = glob.glob(os.path.join(tracts_folder, "*.nii.gz"))
    
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


def save_3D_nparray_nifti(np_matrix_3d, output_image, fname_atlas):
    # Save 3d numpy matrix to niftii image
    # np_matrix_3d is a 3D numpy ndarray
    # output_image is the name of the niftii image created, ex: '3D_matrix.nii.gz'
    img = nib.Nifti1Image(np_matrix_3d, np.eye(4))
    affine = img.get_affine()
    np_matrix_3d_nii = nib.Nifti1Image(np_matrix_3d,affine)
    nib.save(np_matrix_3d_nii, output_image)
    # copy geometric information
    sct.run('fslcpgeom '+fname_atlas+' '+output_image, verbose=0)


def add_tracts(tracts, tracts_to_sum_index):
    tracts_sum = np.empty((tracts[0, 0]).shape)
    for i in tracts_to_sum_index:
        tracts_sum = tracts_sum + tracts[i, 0]
    return tracts_sum


if __name__ == "__main__":
    main()
