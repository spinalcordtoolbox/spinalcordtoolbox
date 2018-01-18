#!/usr/bin/env python

# Compute dice coefficient script
# Two different dice coefficient are computed

# 1 : 
# DICE coefficient between binarized tracts thresholded at 0.5 and masks created from user.
# For each user, the DICE coefficient is calculated for each tract between the user and the atlas
# and averaged over tracts afterwards

# 2 :
# DICE coefficient between users for each tract
# For each tract : dice(user1,user2), dice(user1,user3), dice(user2,user3), etc...
# All of those DICE Coefficients are afterwards averaged in each tract

# Author : Charles Naaman
# Created : 31-07-2014

import sys, io, os, re, glob, math

import numpy as np

path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
import sct_utils as sct


# Tracts for which masks are created
# The enumeration order of the tracts must correspond with the manually created masks
selected_tracts = (2,17)
selected_tracts = list(selected_tracts)

# Folder where the results of dice estimation are outputed
DICE_estimation_folder= 'DICE_coefficient'
if not os.path.isdir(DICE_estimation_folder):
    os.makedirs(DICE_estimation_folder)

# File where results are printed
results_file = os.path.join(DICE_estimation_folder, 'results.txt')

# Folder where the atlas of the tracts are
atlas_tracts_folder = 'cropped_atlas'

# Tract of the dorsal column
sct.run('./add_tracts_dc.py')
dorsal_column_tract = os.path.join(DICE_estimation_folder, 'dorsal_column.nii.gz')

# Name and path of the binarized created tracts 
bin_tracts_folder = os.path.join(DICE_estimation_folder, 'bin_tracts')

# Remove all of the binarized tracts already in the folder
if os.path.isdir(bin_tracts_folder):
    shutil.rmtree(bin_tracts_folder)
os.makedirs(bin_tracts_folder)

bin_tracts_prefix = 'bin_tract_mask'

# Folder where the manually made masks of tracts are located
# In this folder, each subfolder has to contain manually created masks by different users. No other subfolders are allowed.
manual_masks_folder = 'manual_masks'

# Get the names of the folders created by user
user_folders = os.walk(manual_masks_folder).next()[1]

# Determine the number of different users
number_users = len(user_folders)


# Extract the name of the different tracts in the atlas
fname_atlas_tract = glob.glob(os.path.join(atlas_tracts_folder, '*.nii.gz'))
fname_atlas_tract.append(dorsal_column_tract)

# Create binarized tracts 
selected_tracts.append(len(fname_atlas_tract)-1)

# Create the name of the different binarized tracts
fname_bin_tract = [None]*(len(fname_atlas_tract)+1)
for i in selected_tracts:
    if i < 10:
        fname_bin_tract[i] = os.path.join(bin_tracts_folder, bin_tracts_prefix + '0' + str(i) + '.nii.gz')
    else:
        fname_bin_tract[i] = os.path.join(bin_tracts_folder, bin_tracts_prefix + str(i) + '.nii.gz')
        
# Initialize arrays in which DICE estimation results are saved
DICE_atlas_user = np.zeros([number_users, len(selected_tracts)])
DICE_user_user =  np.zeros([math.factorial(number_users-1), len(selected_tracts)])

# Extract a list of the masks for each user
fname_mask = [None]*(len(fname_atlas_tract)+1)


# Put tracts equal to 0 under the threshold    

for i in selected_tracts:
    sct.run('fslmaths ' + fname_atlas_tract[i] + ' -thr 0.5 ' + fname_bin_tract[i])
    sct.run('fslmaths ' + fname_bin_tract[i] + ' -bin ' + fname_bin_tract[i])
    
# Initialize results_file
sct.run('echo "=====================================================================================" > ' + results_file)
sct.run('echo "3D DICE coefficient between atlas and user" >> ' + results_file)
sct.run('echo "=====================================================================================" >> ' + results_file)


for j in range(0, number_users):
    # Get the name and path of the masks created by the user
    fname_mask[j] = glob.glob(os.path.join(manual_masks_folder, user_folders[j] + '*'))
    for i in selected_tracts:
        # Adjust geometry of the binarized tracts and the masks created by user
        sct.run('fslcpgeom ' + fname_mask[j][selected_tracts.index(i)] + ' ' + fname_bin_tract[i])
        # Calculate dice coefficient for each tract
        status, output = sct.run('sct_dice_coefficient ' + fname_mask[j][selected_tracts.index(i)] + ' ' + fname_bin_tract[i])
        # Remove non decimal values from output
        non_decimal = re.compile(r'[^\d.]+')
        output = non_decimal.sub('', output)        
        # Remove the first number since it does not correspond with the dice coefficient
        output = output[1:]
        DICE_atlas_user[j, selected_tracts.index(i)] = float(output)
    sct.run('echo ' +  user_folders[j] + '\t : ' + str(round(np.mean(DICE_atlas_user[j,:]),3)) + ' >> ' + results_file)
    
      
sct.run('echo "=====================================================================================" >> ' + results_file)
sct.run('echo "3D DICE coefficient between users" >> ' + results_file)
sct.run('echo "=====================================================================================" >> ' + results_file)


    # Calculate 3D DICE coefficient between masks created by different users

for i in selected_tracts:
    # Index of the inter-user dice coefficient, the number of coefficients are equal to factorial(number_users)
    index = 0
    for j in range(0, number_users - 1):
        # Get the name and path of the masks created by the user
        for l in range(j+1, number_users):
            # Calculate DICE coefficient for each tract
            status,output = sct.run('sct_dice_coefficient ' + fname_mask[j][selected_tracts.index(i)] + ' ' + fname_mask[l][selected_tracts.index(i)] + ' -o output.txt' )
            # Remove non decimal values from output
            non_decimal = re.compile(r'[^\d.]+')
            output = non_decimal.sub('', output)        
            # Remove the first number since it does not correspond with the dice coefficient
            output = output[1:]
            DICE_user_user[index, selected_tracts.index(i)] = float(output)
            index=index+1
    # Print results in the results file
    sct.run('echo  tract ' + str(i) + ' : \t ' +  str(round(np.mean(DICE_atlas_user[:,selected_tracts.index(i)]),3)) + ' >> ' + results_file )

status, output = sct.run('cat ' + results_file)
print output
   
      
#  OlD STUFF: sum the tracts and calculate DICE on the tracts_sum
# # Calculate 3D DICE coefficient between binarized atlas tracts and masks made by user
# If dice_sum = 0, the DICE coefficient is calculated for each tract individually
# If dice_sum = 1, the DICE coefficient is calculated on the sum of all the tracts
# dice_sum = 1
# fname_sum_man_masks = [None]*(number_users)
# fname_sum_bin_tract = [None]*(number_users)
# if dice_sum == 1:
#     for j in range(0, number_users):
#         add_man_masks = 'fslmaths '
#         add_bin_tracts = 'fslmaths '
#         fname_mask[j] = glob.glob(os.path.join(manual_masks_folder, user_folders[j] + '*'))
#         for i in selected_tracts:
#             # Get the name and path of the masks created by the user
#             add_man_masks = add_man_masks + fname_mask[j][selected_tracts.index(i)] + ' -add '
#             add_bin_tracts = add_bin_tracts + fname_bin_tract[i] + ' -add '
#
#         add_man_masks=add_man_masks[:-len(' -add ')]
#         add_bin_tracts=add_bin_tracts[:-len(' -add ')]
#
#         fname_sum_man_masks[j] = os.path.join(manual_masks_folder, user_folders[j], user_folders[j] + 'sum.nii.gz')
#         fname_sum_bin_tract[j] = os.path.join(bin_tracts_folder, 'sum.nii.gz')
#
#         add_man_masks = add_man_masks + ' ' + fname_sum_man_masks[j]
#         add_bin_tracts = add_bin_tracts + ' ' + fname_sum_bin_tract[j]
#
#         sct.run(add_man_masks)
#         sct.run(add_bin_tracts)
#         # Adjust geometry of the binarized tracts and the masks created by user
#         sct.run('fslcpgeom ' + fname_sum_man_masks[j] + ' ' + fname_bin_tract[i])
#         sct.run('fslcpgeom ' + fname_sum_bin_tract[j] + ' ' + fname_bin_tract[i])
#         sct.run('fslcpgeom ' + fname_sum_bin_tract[j] + ' ' + fname_sum_man_masks[j])
#
#         # Calculate dice coefficient for each user
#         sct.run('sct_dice_coefficient ' + fname_sum_man_masks[j] + ' ' + fname_sum_bin_tract[j] + ' -o output.txt' )
#         # Print results in the results file
#         sct.run('echo ' + user_folders[j] + ' >> ' + results_file )
#         sct.run('cat output.txt >> ' + results_file)
#         sct.run('echo "" >>' + results_file)

# if dice_sum == 1:
#     for j in range(0, number_users - 1):
#         # Get the name and path of the masks created by the user
#         for l in range(j+1, number_users):
#             # Calculate DICE coefficient for each tract
#             sct.run('sct_dice_coefficient ' + fname_sum_man_masks[j] + ' ' + fname_sum_man_masks[l] + ' -o output.txt' )
#             # Print results in the results file
#             sct.run('echo ' + user_folders[j] + '  ' +  user_folders[l] + ' >> ' + results_file )
#             sct.run('cat output.txt >> ' + results_file)
#             sct.run('echo "" >>' + results_file)
        

