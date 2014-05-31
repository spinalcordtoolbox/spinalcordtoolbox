#!/usr/bin/env python

## @package test_sct_estimate_MAP_tracts
#
# - generate synthetic images test for validation of sct_estimate_MAP_tracts
# - test sct_estimate_MAP_tracts using the generated images

#Import library
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import subprocess
import sys
import os

def main():

    print '\nGeneration of files test ...'

    # Extract path of script
    path_script = os.path.dirname(__file__) + '/'

    # Create repertory of images if it does not exist
    if not os.path.exists(path_script+'atlas_test'):
        os.makedirs(path_script+'atlas_test')

    # Create repertory of atlas if it does not exist
    if not os.path.exists(path_script+'images_test'):
        os.makedirs(path_script+'images_test')

    # Initialise tracts variable as object
    tracts = np.empty([4,1], dtype=object)
    for label in range(0, 4):
        tracts[label, 0] = np.zeros([120,120])

    # Generate 4 domains
    tracts[0, 0][0:60, 0:60] = np.ones([60,60])
    tracts[1, 0][60:120, 60:120] = np.ones([60,60])
    tracts[2, 0][0:60, 60:120] = np.ones([60,60])
    tracts[3, 0][60:120, 0:60] = np.ones([60,60])

    # Display tracts
    #plt.imshow(tracts[0, 0])
    #plt.colorbar()
    #plt.show()

    # Save tracts in nifty
    for label in range(0, 4):
        tract = nib.Nifti1Image(tracts[label, 0],np.eye(4))
        fname = 'tract_'+str(label)+'.nii.gz'
        nib.save(tract, path_script + 'atlas_test/'+fname)

    # Generate synthetic image (4 domains which we know values)
    tracts[0, 0][0:60, 0:60] = np.ones([60,60])*10
    tracts[1, 0][60:120, 60:120] = np.ones([60,60])*25
    tracts[2, 0][0:60, 60:120] = np.ones([60,60])*45
    tracts[3, 0][60:120, 0:60] = np.ones([60,60])*100
    data = tracts[0, 0]+tracts[1, 0]+tracts[2, 0]+tracts[3, 0]

    # Display true values
    print '\nTrue values of metrics known in advance'
    print '\t  Label 0 : 10'
    print '\t  Label 1 : 25'
    print '\t  Label 2 : 45'
    print '\t  Label 3 : 100'

    # Define noise range
    sigma_noise = np.array ([0, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 10, 25, 50, 75, 100])

    # Initialise array of file names
    fname_image = [[]] * int(np.size(sigma_noise))

    # Generate data without noise
    data_nii = nib.Nifti1Image(data,np.eye(4))
    nib.save(data_nii,path_script + 'images_test/image_n_0.nii.gz')
    fname_image[0] = path_script + 'images_test/image_n_0.nii.gz'

    # Generate data with noise
    for i in range(1, int(np.size(sigma_noise))):
        w = np.random.normal(0,sigma_noise[i],[120,120])
        data_nii = data + w
        # Display image
        #plt.imshow(data)
        #plt.colorbar()
        #plt.show()
        #plt.close()
        # Save synthetic image
        data_nii = nib.Nifti1Image(data_nii,np.eye(4))
        nib.save(data_nii, path_script + 'images_test/image_n_'+ str(sigma_noise[i]).replace('.','_')+'.nii.gz')
        fname_image[i] = path_script + 'images_test/image_n_'+ str(sigma_noise[i]).replace('.','_')+'.nii.gz'

    # Test program for all images test
    for i in range(0, int(np.size(fname_image))):
        print '\n _____________________________Test for image with STD noise = ' + str(sigma_noise[i]) +'_____________________________'
        subprocess.call([sys.executable, path_script + '../../scripts/sct_estimate_MAP_tracts.py', '-i'+fname_image[i],'-t'+ path_script + 'atlas_test'])

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()
