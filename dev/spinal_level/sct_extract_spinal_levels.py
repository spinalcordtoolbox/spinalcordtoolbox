#!/usr/bin/env python
#########################################################################################
# 
# Extract spinal levels
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Karun Raju, Julien Touati
# Modified: 2014-06-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################


# check if needed Python libraries are already installed or not
import os
import getopt
import commands
import sys
import matplotlib.pyplot as plt
try:
    import nibabel
except ImportError:
    print '--- nibabel not installed! Exit program. ---'
    sys.exit(2)
try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! Exit program. ---'
    sys.exit(2)

#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    
    #loading files    
    fname_template_vertebral_level  = path_sct+'/data/template/MNI-Poly-AMU_level.nii.gz'
    fname_template_vertebral_cord = path_sct+'/data/template/MNI-Poly-AMU_cord.nii.gz'
    
    vertebral_level_file = nibabel.load(fname_template_vertebral_level)
    vertebral_cord_file = nibabel.load(fname_template_vertebral_cord)
    
    vertebral_level_image = vertebral_level_file.get_data()
    vertebral_cord_image = vertebral_cord_file.get_data()
    
    hdr_vertebral_level=vertebral_level_file.get_header()
    hdr_vertebral_cord=vertebral_cord_file.get_header()
    
    z_size_vertebral_level=hdr_vertebral_level['pixdim'][3]
    z_size_vertebral_cord=hdr_vertebral_cord['pixdim'][3]
    
    #finding the centers of the vertebral levels
    vertebral_levels = find_mid_point_vertebral_level(vertebral_level_image)    
    
    # Mean and Variance of Vertebral centers (Data from the graph)
    vertebral_Mu = np.zeros(len(vertebral_levels))
    vertebral_Sigma = np.zeros(len(vertebral_levels))

    vertebral_Mu[0], vertebral_Sigma[0] = 0, 0
    vertebral_Mu[1], vertebral_Sigma[1] = 0, 0
    vertebral_Mu[2], vertebral_Sigma[2] =  69.99, 4.58
    vertebral_Mu[3], vertebral_Sigma[3] =  86.11, 5.35
    vertebral_Mu[4], vertebral_Sigma[4] = 101.16, 6.33
    vertebral_Mu[5], vertebral_Sigma[5] = 115.54, 7.15
    vertebral_Mu[6], vertebral_Sigma[6] = 130.08, 8.25 
    vertebral_Mu[7], vertebral_Sigma[7] = 0, 0
    vertebral_Mu[8], vertebral_Sigma[8] = 0, 0
    vertebral_Mu[9], vertebral_Sigma[9] = 0, 0
    vertebral_Mu[10], vertebral_Sigma[10] = 0, 0
    vertebral_Mu[11], vertebral_Sigma[11] = 0, 0
    vertebral_Mu[12], vertebral_Sigma[12] = 0, 0

    # Mean and Variance of Spinal centers (Data from the graph)
    spinal_Mu = np.zeros(len(vertebral_levels))
    spinal_Sigma = np.zeros(len(vertebral_levels))
    
    spinal_Mu[0], spinal_Sigma[0] = 0, 0
    spinal_Mu[1], spinal_Sigma[1] = 0, 0
    spinal_Mu[2], spinal_Sigma[2] =  51.46, 3.15
    spinal_Mu[3], spinal_Sigma[3] =  65.68, 3.96
    spinal_Mu[4], spinal_Sigma[4] =  81.08, 5.15
    spinal_Mu[5], spinal_Sigma[5] =  95.39, 5.68
    spinal_Mu[6], spinal_Sigma[6] = 109.33, 6.05
    spinal_Mu[7], spinal_Sigma[7] = 122.77, 7.03
    spinal_Mu[8], spinal_Sigma[8] = 0, 0
    spinal_Mu[9], spinal_Sigma[9] = 0, 0
    spinal_Mu[10], spinal_Sigma[10] = 0, 0
    spinal_Mu[11], spinal_Sigma[11] = 0, 0
    spinal_Mu[12], spinal_Sigma[12] = 0, 0
    
    #==================================================================================
    #Extrapolating unknown Means and Variances from the given data
    
    #Finding the distance between the known Spinal centers and Vertebral centers from the given data
    spinal_vertebral_dist=np.zeros(5)
    for i in range(5):
        spinal_vertebral_dist[i]=(vertebral_Mu[i+2] - spinal_Mu[i+2])
    
    #Linear fit of the distances between Vertebral and Spinal centers
    popt_spinal_vertebral = np.polyfit(np.arange(3,8),spinal_vertebral_dist,1)
    P_fit_spinal_vertebral_dist = np.poly1d(popt_spinal_vertebral)
    
    plt.plot(np.arange(0,15),P_fit_spinal_vertebral_dist(np.arange(0,15)),marker='+')
    plt.plot(np.arange(3,8),spinal_vertebral_dist,marker='o',linestyle='None')
    plt.title('Spinal-Vertebral distances')
    plt.show()

    conversion_factor=np.zeros(4)
    for i in range(2,6,1):
        conversion_factor[i-2]=(vertebral_levels[i]-vertebral_levels[i+1])/(vertebral_Mu[i+1]-vertebral_Mu[i])
    average_conversion_factor=np.mean(conversion_factor)

    delta_vertebral_levels = np.zeros(len(vertebral_levels)-1)
    for i in range(len(vertebral_levels)-1):
        delta_vertebral_levels[i] = (vertebral_levels[i] - vertebral_levels[i+1])/average_conversion_factor

    vertebral_Mu[0] = vertebral_Mu[1] - delta_vertebral_levels[0]
    vertebral_Mu[1] = vertebral_Mu[2] - delta_vertebral_levels[1]
    vertebral_Mu[7] = vertebral_Mu[6] + delta_vertebral_levels[6]
    vertebral_Mu[8] = vertebral_Mu[7] + delta_vertebral_levels[7]
    vertebral_Mu[9] = vertebral_Mu[8] + delta_vertebral_levels[8]
    vertebral_Mu[10] = vertebral_Mu[9] + delta_vertebral_levels[9]
    vertebral_Mu[11] = vertebral_Mu[10] + delta_vertebral_levels[10]
    vertebral_Mu[12] = vertebral_Mu[11] + delta_vertebral_levels[11]

    spinal_Mu[0] = vertebral_Mu[0] - P_fit_spinal_vertebral_dist(1)
    spinal_Mu[1] = vertebral_Mu[1] - P_fit_spinal_vertebral_dist(2)
    spinal_Mu[8] = vertebral_Mu[8] - P_fit_spinal_vertebral_dist(9)
    spinal_Mu[9] = vertebral_Mu[9] - P_fit_spinal_vertebral_dist(10)
    spinal_Mu[10] = vertebral_Mu[10] - P_fit_spinal_vertebral_dist(11)
    spinal_Mu[11] = vertebral_Mu[11] - P_fit_spinal_vertebral_dist(12)
    spinal_Mu[12] = vertebral_Mu[12] - P_fit_spinal_vertebral_dist(13)

    #Linear Fit of the known Vertebral variances to find the unkown variances
    popt_vertebral_sigma = np.polyfit(np.arange(3,8),vertebral_Sigma[2:7],1)
    P_fit_vertebral_sigma = np.poly1d(popt_vertebral_sigma)
    plt.plot(np.arange(0,14),P_fit_vertebral_sigma(np.arange(0,14)),marker='+')
    plt.title('Vertebral_sigma')
    plt.plot(np.arange(3,8),vertebral_Sigma[2:7],marker='o',linestyle='None')
    plt.show()

    #Linear Fit of the known Spinal variances to find the unknown variances
    popt_spinal_sigma = np.polyfit(np.arange(3,9),spinal_Sigma[2:8],1)
    P_fit_spinal_sigma = np.poly1d(popt_spinal_sigma)
    plt.plot(np.arange(0,14),P_fit_spinal_sigma(np.arange(0,14)),marker='+')
    plt.title('Spinal_sigma')
    plt.plot(np.arange(3,9),spinal_Sigma[2:8],marker='o',linestyle='None')
    plt.show()

    vertebral_Sigma[1]=P_fit_vertebral_sigma(0)
    vertebral_Sigma[0]=P_fit_vertebral_sigma(1)
    vertebral_Sigma[7]=P_fit_vertebral_sigma(8)
    vertebral_Sigma[8]=P_fit_vertebral_sigma(9)
    vertebral_Sigma[9]=P_fit_vertebral_sigma(10)
    vertebral_Sigma[10]=P_fit_vertebral_sigma(11)
    vertebral_Sigma[11]=P_fit_vertebral_sigma(12)
    vertebral_Sigma[12]=P_fit_vertebral_sigma(13)

    spinal_Sigma[1]=P_fit_spinal_sigma(2)
    spinal_Sigma[0]=P_fit_spinal_sigma(1)
    spinal_Sigma[8]=P_fit_spinal_sigma(9)
    spinal_Sigma[9]=P_fit_spinal_sigma(10)
    spinal_Sigma[10]=P_fit_spinal_sigma(11)
    spinal_Sigma[11]=P_fit_spinal_sigma(12)
    spinal_Sigma[12]=P_fit_spinal_sigma(13)

    for i in range(len(vertebral_levels)):
        y = gaussian(np.arange(0,500,0.1),spinal_Mu[i],spinal_Sigma[i])
        plt.plot(np.arange(0,500,0.1),y)
    plt.title('Mean Distance from the PMJ to the centre of each Nerve Rootlet Segment')
    plt.show()

    #Finding the distance between the Spinal and Vertebral centers in the image
    spinal_vertebral_distances=np.zeros(len(vertebral_levels))
    for i in range(len(vertebral_levels)):
        spinal_vertebral_distances[i]=round((vertebral_Mu[i] - spinal_Mu[i])*average_conversion_factor)

    spinal_levels = np.zeros(len(vertebral_levels))
    for i in range(len(vertebral_levels)):
        spinal_levels[i] = int(vertebral_levels[i] + spinal_vertebral_distances[i])

    #Creating Templates for each Spinal level
    k = 0
    X,Y,Z = np.where(vertebral_cord_image>0)
    for i in spinal_levels[:]:
        template_spinal_image = vertebral_cord_image.copy()
        count = 0
        
        for j in range(np.amax(Z),np.amin(Z)-1,-1):
            x,y = np.where(vertebral_cord_image[:,:,j]>0)
            diff = (i - j)/average_conversion_factor
            value = gaussian((spinal_Mu[k]+diff),spinal_Mu[k],spinal_Sigma[k])
            template_spinal_image[x,y,j] = value
        k = k+1

        # Write NIFTI volumes
        hdr_vertebral_cord.set_data_dtype('uint8') # set imagetype to uint8
        print '\nWrite NIFTI volumes...'
        img = nibabel.Nifti1Image(template_spinal_image, None, hdr_vertebral_cord)
        if k<=8:
            file_name = 'spinal_level_C'+str(k)+'.nii.gz'
        else:
            file_name = 'spinal_level_T'+str(k-8)+'.nii.gz'
        nibabel.save(img,file_name)
        print '.. File created:' + file_name    


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