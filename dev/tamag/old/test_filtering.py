#!/usr/bin/env python


#Test programm for the 2D gaussian filter



# check if needed Python libraries are already installed or not
import os

import sys
import sct_utils as sct
import nibabel as nib
import numpy

import matplotlib.pyplot as plt
import sct_create_mask
from msct_image import Image

def filter_2Dgaussian(input_padded_file, size_filter, output_file_name='Result'):
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(input_padded_file)
        print nx, ny, nz, nt, px, py, pz, pt
        gaussian_filter=sct_create_mask.create_mask2d(center=((int(size_filter/px)-1.0)/2.0,(int(size_filter/py)-1.0)/2.0), shape='gaussian', size=size_filter, nx=int(size_filter/px),ny=int(size_filter/py))    #pb center
#on a oublie le facteur multiplicatif du filtre gaussien classique (create_mask2D ne le prend pas en compte)
        print (int(size_filter/px)-1.0)/2.0,(int(size_filter/py)-1.0)/2.0
        print int(size_filter/px), int(size_filter/py)

        plt.plot(gaussian_filter)
        plt.grid()
        plt.show()
#center=(int(size_filter/px)/2.0,int(size_filter/py)/2.0)

        #Pad
        #gaussian_filter_pad = 'pad_' + gaussian_filter
        #sct.run('sct_c2d ' + gaussian_filter + ' -pad ' + pad + 'x0vox ' + pad + 'x' + pad + 'x0vox 0 -o ' + gaussian_filter_pad)   #+ pad+ 'x'

        image_input_padded_file=Image(input_padded_file)

        print('1: numpy.sum(image_input_padded_file.data[:,:,:])', numpy.sum(image_input_padded_file.data[:,:,:]))

        #Create the output file
        #im_output=image_input_padded_file
        im_output = image_input_padded_file.data * 0     #ici, image_input_padded_file.data est lui aussi mis a zero
        #im_output_freq=image_input_padded_file
        im_output_freq = image_input_padded_file.data * 0

        #Create padded filter in frequency domain
        gaussian_filter_freq=numpy.fft.fft2(gaussian_filter, s=(image_input_padded_file.data.shape[0], image_input_padded_file.data.shape[1]))

        plt.plot(gaussian_filter_freq)
        plt.grid()
        plt.show()

        hauteur_image=image_input_padded_file.data.shape[2]

        print('2: numpy.sum(image_input_padded_file.data[:,:,:])', numpy.sum(image_input_padded_file.data[:,:,:]))

        #Apply 2D filter to every slice of the image
        for i in range(hauteur_image):
            image_input_padded_file_frequentiel=numpy.fft.fft2(image_input_padded_file.data[:,:,i], axes=(0,1))
            im_output_freq[:,:,i]=gaussian_filter_freq*image_input_padded_file_frequentiel
            im_output[:,:,i]=numpy.fft.ifft2(im_output_freq[:,:,i], axes=(0,1))

        print('numpy.sum(im_output[:,:,:])', numpy.sum(im_output[:,:,:]))

        #Save the file
        #im_output.setFileName(output_file_name)
        #im_output.save('minimize')

        # Generate the T1, PD and MTVF maps as a NIFTI file with the right header
        path_spgr, file_name, ext_spgr = sct.extract_fname(input_padded_file)
        fname_output = path_spgr + output_file_name + ext_spgr
        sct.printv('Generate the NIFTI file with the right header...')
        # Associate the header to the MTVF and PD maps data as a NIFTI file
        hdr = nib.load(input_padded_file).get_header()
        img_with_hdr = nib.Nifti1Image(im_output, None, hdr)
        # Save the T1, PD and MTVF maps file
        nib.save(img_with_hdr, fname_output)      #PB: enregistre le fichier Result dans tmp.150317111945 lors des tests

        return img_with_hdr


#def filter_2Dmean(input_padded_file, size_kernel, output_file_name='Result'):
    #mean_filter=(1/9)*numpy.ones((int(9/px),int(9/py)))  #filter of 9mm

os.chdir('/home/tamag/data/test_straightening/20150316_allan/tmp.150317111945/')

filter_2Dgaussian('tmp.centerline_pad.nii.gz', 15)


