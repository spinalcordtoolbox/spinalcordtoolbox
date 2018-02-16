#!/usr/bin/env python
#########################################################################################
# 
# Extract spinal levels
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Karun Raju, Julien Touati
# Modified: 2016-11-28 by jcohenadad
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys, io, os, shutil, time

import matplotlib
import nibabel
import numpy as np

#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # initialization
    folder_out = 'spinal_levels/'
    file_infolabel = 'info_label.txt'
    name_level = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5']

    # Use agg to redirect figure display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # get path of the toolbox
    path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # Create output folder (and delete previous folder if exist)
    try:
        shutil.rmtree(folder_out, ignore_errors=True)
    except Exception as e:
        print e
    os.mkdir(folder_out)

    # Load files
    fname_template_vertebral_level = os.path.join(path_sct, 'data', 'PAM50', 'template', 'PAM50_levels.nii.gz')
    fname_template_vertebral_cord = os.path.join(path_sct, 'data', 'PAM50', 'template', 'PAM50_cord.nii.gz')
    
    vertebral_level_file = nibabel.load(fname_template_vertebral_level)
    vertebral_cord_file = nibabel.load(fname_template_vertebral_cord)
    
    vertebral_level_image = vertebral_level_file.get_data()
    vertebral_cord_image = vertebral_cord_file.get_data()
    
    hdr_vertebral_level = vertebral_level_file.get_header()
    hdr_vertebral_cord = vertebral_cord_file.get_header()

    # get dimensions
    px, py, pz = hdr_vertebral_level.get_zooms()

    # z_size_vertebral_level = hdr_vertebral_level['pixdim'][3]
    # z_size_vertebral_cord = hdr_vertebral_cord['pixdim'][3]
    
    # Find the centers of the vertebral levels
    vertebral_levels = find_mid_point_vertebral_level(vertebral_level_image)
    nb_vert = len(vertebral_levels)
    
    # Mean and Variance of Vertebral centers with respect to the PMJ
    # Note: these were obtained from Figure 3 in https://www.ncbi.nlm.nih.gov/pubmed/25523587
    vertebral_Mu = np.zeros(nb_vert)
    vertebral_Sigma = np.zeros(nb_vert)
    vertebral_Mu[2], vertebral_Sigma[2] = 69.99, 4.58  # C3
    vertebral_Mu[3], vertebral_Sigma[3] = 86.11, 5.35  # C4
    vertebral_Mu[4], vertebral_Sigma[4] = 101.16, 6.33  # C5
    vertebral_Mu[5], vertebral_Sigma[5] = 115.54, 7.15  # C6
    vertebral_Mu[6], vertebral_Sigma[6] = 130.08, 8.25   # C7

    # Mean and Variance of Spinal centers with respect to the PMJ
    spinal_Mu = np.zeros(nb_vert)
    spinal_Sigma = np.zeros(nb_vert)
    spinal_Mu[2], spinal_Sigma[2] = 51.46, 3.15  # C3
    spinal_Mu[3], spinal_Sigma[3] = 65.68, 3.96  # C4
    spinal_Mu[4], spinal_Sigma[4] = 81.08, 5.15  # C5
    spinal_Mu[5], spinal_Sigma[5] = 95.39, 5.68  # C6
    spinal_Mu[6], spinal_Sigma[6] = 109.33, 6.05  # C7
    spinal_Mu[7], spinal_Sigma[7] = 122.77, 7.03  # C8

    #==================================================================================
    # Extrapolating unknown Means and Variances from the given data
    
    # Finding the distance between the known Spinal centers and Vertebral centers from the given data
    # here the hard-coded range(5) and "+2" in the index corresponds to the known values from the graph: i=2 to i=7
    spinal_vertebral_dist = np.zeros(5)
    for i in range(5):
        spinal_vertebral_dist[i] = vertebral_Mu[i+2] - spinal_Mu[i+2]
    
    # Linear fit of the distances between Vertebral and Spinal centers
    popt_spinal_vertebral = np.polyfit(np.arange(3, 8), spinal_vertebral_dist, 1)
    P_fit_spinal_vertebral_dist = np.poly1d(popt_spinal_vertebral)

    # display
    plt.figure()
    plt.plot(np.arange(0, nb_vert), P_fit_spinal_vertebral_dist(np.arange(0, nb_vert)), marker='+')
    plt.plot(np.arange(3, 8), spinal_vertebral_dist, marker='o', linestyle='None')
    plt.title('Spinal-Vertebral distances')
    plt.savefig('fig_spinal-to-VertebralDistance.png')

    #
    # # Compute the conversion factor between delta(vertlevel_PAM50) and delta(vertlevel_toronto), based on the known levels from Toronto.
    # conversion_factor = np.zeros(4)
    # for i in range(2, 6, 1):
    #     conversion_factor[i-2] = (vertebral_levels[i] - vertebral_levels[i+1]) / (vertebral_Mu[i+1] - vertebral_Mu[i])
    # average_conversion_factor = np.mean(conversion_factor)
    #
    # # Apply the conversion factor to scale the delta(vertlevel_PAM50) for all levels.
    # delta_vertebral_levels = np.zeros(len(vertebral_levels)-1)
    # for i in range(len(vertebral_levels)-1):
    #     delta_vertebral_levels[i] = (vertebral_levels[i] - vertebral_levels[i+1]) / average_conversion_factor
    #
    # # Compute vertebral distance from the PMJ
    # for i in range(0, len(vertebral_levels)):
    #     vertebral_Mu[0] = vertebral_Mu[1] - delta_vertebral_levels[0]
    #
    # vertebral_Mu[0] = vertebral_Mu[1] - delta_vertebral_levels[0]
    # vertebral_Mu[1] = vertebral_Mu[2] - delta_vertebral_levels[1]
    # vertebral_Mu[7] = vertebral_Mu[6] + delta_vertebral_levels[6]
    # vertebral_Mu[8] = vertebral_Mu[7] + delta_vertebral_levels[7]
    # vertebral_Mu[9] = vertebral_Mu[8] + delta_vertebral_levels[8]
    # vertebral_Mu[10] = vertebral_Mu[9] + delta_vertebral_levels[9]
    # vertebral_Mu[11] = vertebral_Mu[10] + delta_vertebral_levels[10]
    # vertebral_Mu[12] = vertebral_Mu[11] + delta_vertebral_levels[11]
    #
    # spinal_Mu[0] = vertebral_Mu[0] - P_fit_spinal_vertebral_dist(1)
    # spinal_Mu[1] = vertebral_Mu[1] - P_fit_spinal_vertebral_dist(2)
    # spinal_Mu[8] = vertebral_Mu[8] - P_fit_spinal_vertebral_dist(9)
    # spinal_Mu[9] = vertebral_Mu[9] - P_fit_spinal_vertebral_dist(10)
    # spinal_Mu[10] = vertebral_Mu[10] - P_fit_spinal_vertebral_dist(11)
    # spinal_Mu[11] = vertebral_Mu[11] - P_fit_spinal_vertebral_dist(12)
    # spinal_Mu[12] = vertebral_Mu[12] - P_fit_spinal_vertebral_dist(13)

    # Compute spinal distance from each vertebral level using fitted distance
    # Note: distance is divided by pz to account for voxel size (because variance in Cadotte et al. is given in mm).
    spinal_levels = np.zeros(nb_vert)
    for i in range(0, nb_vert):
        spinal_levels[i] = vertebral_levels[i] + P_fit_spinal_vertebral_dist(i) / pz
    #
    # # Linear Fit of the known Vertebral variances to find the unkown variances
    # popt_vertebral_sigma = np.polyfit(np.arange(3,8), vertebral_Sigma[2:7],1)
    # P_fit_vertebral_sigma = np.poly1d(popt_vertebral_sigma)
    # plt.plot(np.arange(0,14),P_fit_vertebral_sigma(np.arange(0,14)),marker='+')
    # plt.title('Vertebral_sigma')
    # plt.plot(np.arange(3, 8), vertebral_Sigma[2:7],marker='o',linestyle='None')
    # plt.show()

    # Linear Fit of the known Spinal variances
    popt_spinal_sigma = np.polyfit(np.arange(3, 9), spinal_Sigma[2: 8], 1)
    P_fit_spinal_sigma = np.poly1d(popt_spinal_sigma)
    # display
    plt.figure()
    plt.plot(np.arange(0, nb_vert), P_fit_spinal_sigma(np.arange(0, nb_vert)), marker='+')
    plt.title('Spinal_sigma')
    plt.plot(np.arange(3, 9), spinal_Sigma[2:8], marker='o', linestyle='None')
    plt.savefig('fig_spinalVariance.png')

    # Compute spinal variance using fitted variance
    # Note: variance is divided by pz to account for voxel size (because variance in Cadotte et al. is given in mm).
    for i in range(0, nb_vert):
        spinal_Sigma[i] = P_fit_spinal_sigma(i) / pz

    # vertebral_Sigma[1] = P_fit_vertebral_sigma(0)
    # vertebral_Sigma[0] = P_fit_vertebral_sigma(1)
    # vertebral_Sigma[7] = P_fit_vertebral_sigma(8)
    # vertebral_Sigma[8] = P_fit_vertebral_sigma(9)
    # vertebral_Sigma[9] = P_fit_vertebral_sigma(10)
    # vertebral_Sigma[10] = P_fit_vertebral_sigma(11)
    # vertebral_Sigma[11] = P_fit_vertebral_sigma(12)
    # vertebral_Sigma[12] = P_fit_vertebral_sigma(13)
    #
    # spinal_Sigma[1] = P_fit_spinal_sigma(2)
    # spinal_Sigma[0] = P_fit_spinal_sigma(1)
    # spinal_Sigma[8] = P_fit_spinal_sigma(9)
    # spinal_Sigma[9] = P_fit_spinal_sigma(10)
    # spinal_Sigma[10] = P_fit_spinal_sigma(11)
    # spinal_Sigma[11] = P_fit_spinal_sigma(12)
    # spinal_Sigma[12] = P_fit_spinal_sigma(13)


    # for i in range(len(vertebral_levels)):
    #     y = gaussian(np.arange(0, 500, 0.1), spinal_Mu[i], spinal_Sigma[i])
    #     plt.plot(np.arange(0, 500, 0.1), y)
    # plt.title('Mean Distance from the PMJ to the centre of each Nerve Rootlet Segment')
    # plt.show()
    #
    # # Finding the distance between the Spinal and Vertebral centers in the image
    # spinal_vertebral_distances = np.zeros(len(vertebral_levels))
    # for i in range(len(vertebral_levels)):
    #     spinal_vertebral_distances[i] = round((vertebral_Mu[i] - spinal_Mu[i])*average_conversion_factor)
    #
    # spinal_levels = np.zeros(len(vertebral_levels))
    # for i in range(len(vertebral_levels)):
    #     spinal_levels[i] = int(vertebral_levels[i] + spinal_vertebral_distances[i])

    # Display spinal levels
    plt.figure()
    for i in range(nb_vert):
        y = gaussian(np.arange(0, vertebral_level_image.shape[2], 1), spinal_levels[i], spinal_Sigma[i])
        plt.plot(np.arange(0, vertebral_level_image.shape[2], 1), y)
    plt.title('Probabilistic distribution of spinal level')
    plt.savefig('fig_spinalLevels.png')

    # Creating an image for each Spinal level
    k = 0
    X, Y, Z = np.where(vertebral_cord_image > 0)
    file_name = [None] * nb_vert
    for i in range(nb_vert):
        template_spinal_image = vertebral_cord_image.copy()
        count = 0
        # loop across z
        for iz in range(np.amax(Z), np.amin(Z)-1, -1):
            x, y = np.where(vertebral_cord_image[:, :, iz] > 0)
            template_spinal_image[x, y, iz] = gaussian(iz, spinal_levels[i], spinal_Sigma[i])
            # diff = (i - j)/average_conversion_factor
            # value = gaussian((spinal_Mu[k] + diff), spinal_Mu[k], spinal_Sigma[k])
            # template_spinal_image[x, y, j] = value
        # k = k+1

        # Write NIFTI volumes
        hdr_vertebral_cord.set_data_dtype('uint8') # set imagetype to uint8
        print '\nWrite NIFTI volumes...'
        img = nibabel.Nifti1Image(template_spinal_image, None, hdr_vertebral_cord)
        file_name[i] = 'spinal_level_'+str(i+1).zfill(2)+'.nii.gz'
        nibabel.save(img, os.path.join(folder_out, file_name[i]))
        print '  File created:' + file_name[i]

    # create info_label.txt file
    fid_infolabel = io.open(os.path.join(folder_out, file_infolabel,) 'w')
    # Write date and time
    fid_infolabel.write('# Spinal levels labels - generated on ' + time.strftime('%Y-%m-%d') + '\n')
    fid_infolabel.write('# Keyword=IndivLabels (Please DO NOT change this line)\n')
    fid_infolabel.write('# ID, name, file\n')
    for i in range(nb_vert):
        fid_infolabel.write('%i, %s, %s\n' % (i, 'Spinal level ' + name_level[i], file_name[i]))
    fid_infolabel.close()


#==================================================================================
def find_mid_point_vertebral_level(data):

    vertebral_levels = np.zeros(int(np.amax(data)))
    for i in range((int(np.amin(data))+1),(int(np.amax(data))+1)):
    
        #finding the co-ordinates of voxels in each level 
        x,y,z = np.where(data==i)
        z = np.sort(z)
        vertebral_levels[i-1] = np.amin(z) + round((np.amax(z)-np.amin(z))/2)
    return vertebral_levels

#==================================================================================
def gaussian(x, mu, sig):
    return (np.exp(-((x - mu)*(x - mu))/(2*sig*sig)))/(np.sqrt(2*(np.pi)*sig*sig))

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()
