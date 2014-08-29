#!/usr/bin/env python

# Add .nii.gz tracts and save their addition to niftii format

from generate_phantom import get_tracts, save_3D_nparray_niftii
from numpy import zeros,empty

# To get the dorsal colum, tracts_to_sum_index = 0,1,15,16
tracts_to_sum_index = 0,1,15,16
folder_atlas = "cropped_atlas"
tracts_sum_img = 'DICE_coefficient/dorsal_column.nii.gz'
def main():
    
    # Extract the tracts from the atlas' folder
    tracts = get_tracts(folder_atlas)
    # Get the sum of the tracts 
    tracts_sum = add_tracts(tracts, tracts_to_sum_index)
    # Save sum of the tracts to niftii
    save_3D_nparray_niftii(tracts_sum, tracts_sum_img)
    
def add_tracts(tracts, tracts_to_sum_index):
    tracts_sum = empty((tracts[0, 0]).shape)
    for i in tracts_to_sum_index:
        tracts_sum = tracts_sum + tracts[i,0]
    return tracts_sum
    
if __name__ == "__main__":
    main()
