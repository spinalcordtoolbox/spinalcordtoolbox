#!/usr/bin/env python
#########################################################################################
#
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Touati
# Created: 2014-08-11
#
# About the license: see the file LICENSE.TXT
#########################################################################################


#DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 0
        self.mean_intensity = 1000  # value to assign to the spinal cord
        self.padding = 3 # vox
        self.window_length = 80 # size of the smoothing window

# check if needed Python libraries are already installed or not
import sys
import commands
import os
import getopt

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))

import sct_utils as sct
import nibabel
import numpy as np

from time import strftime
import matplotlib.pyplot as plt
from scipy.interpolate import splrep,splev
from scipy import ndimage
from msct_smooth import smoothing_window

from sct_image import Image

def main():

   #Initialization
   fname = ''
   fname_centerline = ''
   mean_intensity = param.mean_intensity
   verbose = param.verbose
   padding = param.padding
   window_length = param.window_length

   try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:c:v:p:')
   except getopt.GetoptError:
       usage()
   for opt, arg in opts :
       if opt == '-h':
           usage()
       elif opt in ("-i"):
           fname = arg
       elif opt in ("-c"):
           fname_centerline = arg
       elif opt in ("-p"):
           window_length = int(arg)
       elif opt in ('-v'):
           verbose = int(arg)

   # display usage if a mandatory argument is not provided
   #if fname == '' or fname_centerline == '':
   if fname == '':
       usage()


   # check existence of input files
   print'\nCheck if file exists ...'
   sct.check_file_exist(fname)
   #sct.check_file_exist(fname_centerline)

   # Display arguments
   print'\nCheck input arguments...'
   print'  Input volume ...................... '+fname
   print'  Centerline ...................... '+fname_centerline
   print'  Verbose ........................... '+str(verbose)

   # Extract path, file and extension
   path_input, file_input, ext_input = sct.extract_fname(fname)


   sct.printv('\nOpen volume...',verbose)
   file = nibabel.load(fname)
   data = file.get_data()
   hdr = file.get_header()

   if fname_centerline != '':
   ## [Process 1] Command for extracting center of mass for each slice of the centerline file if provided
       sct.printv('\nOpen centerline...',verbose)
       print '\nGet dimensions of input centerline...'
       nx, ny, nz, nt, px, py, pz, pt = Image(fname_centerline).dim
       print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
       print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'
       file_c = nibabel.load(fname_centerline)
       data_c = file_c.get_data()


       #X,Y,Z = (data_c>0).nonzero()

       #min_z_index, max_z_index = min(Z), max(Z)


       z_centerline = [iz for iz in range(0, nz, 1) if data_c[:,:,iz].any() ]
       nz_nonz = len(z_centerline)
       if nz_nonz==0 :
           print '\nERROR: Centerline is empty'
           sys.exit()
       x_centerline = [0 for iz in range(0, nz_nonz, 1)]
       y_centerline = [0 for iz in range(0, nz_nonz, 1)]
       #print("z_centerline", z_centerline,nz_nonz,len(x_centerline))
       print '\nGet center of mass of the centerline ...'
       for iz in xrange(len(z_centerline)):
           x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(np.array(data_c[:,:,z_centerline[iz]]))
   # end of Process 1

   ## [Process 2] Process for defining the middle vertical line as reference for normalizing the intensity of the image
   if fname_centerline == '':
       print '\nGet dimensions of input image...'
       nx, ny, nz, nt, px, py, pz, pt = Image(fname).dim
       print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
       print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'
       z_centerline = [iz for iz in range(0, nz, 1)]
       nz_nonz = len(z_centerline)
       x_middle = int(round(nx/2))
       y_middle = int(round(ny/2))
       x_centerline = [x_middle for iz in range(0, nz, 1)]
       y_centerline = [y_middle for iz in range(0, nz, 1)]
       # end of Process 2

   means = [0 for iz in range(0, nz_nonz, 1)]

   print '\nGet mean intensity along the centerline ...'
   for iz in xrange(len(z_centerline)):
       means[iz] =  np.mean(data[(int(round(x_centerline[iz]))-padding):(int(round(x_centerline[iz]))+padding),(int(round(y_centerline[iz]))-padding):(int(round(y_centerline[iz]))+padding),z_centerline[iz]])


   # print('\nSmoothing results with spline...')
   # # Smoothing with scipy library (Julien Touati's code)
   # m =np.mean(means)
   # sigma = np.std(means)
   # smoothing_param = (((m + np.sqrt(2*m))*(sigma**2))+((m - np.sqrt(2*m))*(sigma**2)))/2
   # #Equivalent to : m*sigma**2
   # tck = splrep(z_centerline, means, s=smoothing_param)
   # means_smooth = splev(z_centerline, tck)

   # Smoothing with low-pass filter
   print '\nSmoothing with lowpass filter: butterworth order 5...'
   from msct_smooth import lowpass
   means_smooth = lowpass(means)



   # #Smoothing with nurbs
   #points = [[means[n],0, z_centerline[n]] for n in range(len(z_centerline))]
   #nurbs = NURBS(3,1000,points)
   #P = nurbs.getCourbe3D()
   #means_smooth=P[0]  #size of means_smooth? should be bigger than len(z_centerline)

   # #Smoothing with hanning
   # print('\nSmoothing results with hanning windowing...')
   # means = np.asarray(means)
   # means_smooth = smoothing_window(means, window_len=window_length)
   # print means.shape[0], means_smooth.shape[0]

   if verbose :
       plt.figure()
       #plt.subplot(2,1,1)
       plt.plot(z_centerline,means, "ro")
       #plt.subplot(2,1,2)
       plt.plot(means_smooth)
       plt.title("Mean intensity: Type of window: hanning     Window_length= %d mm" % window_length)
       plt.show()
   print('\nNormalizing intensity along centerline...')


   #Define extended meaned intensity for all the spinal cord
   means_smooth_extended = [0 for i in range(0, data.shape[2], 1)]
   for iz in range(len(z_centerline)):
       means_smooth_extended[z_centerline[iz]] = means_smooth[iz]


   X_means_smooth_extended = np.nonzero(means_smooth_extended)
   X_means_smooth_extended = np.transpose(X_means_smooth_extended)

   if len(X_means_smooth_extended) != 0:
        means_smooth_extended[0] = means_smooth_extended[X_means_smooth_extended[0]]
        means_smooth_extended[-1] = means_smooth_extended[X_means_smooth_extended[-1]]
        #Add two rows to the vector X_mask_completed:
        # one before as mask_completed[0] is now diff from 0
        # one after as mask_completed[-1] is now diff from 0
        X_means_smooth_extended = np.append(X_means_smooth_extended, len(means_smooth_extended)-1)
        X_means_smooth_extended = np.insert(X_means_smooth_extended, 0, 0)
        #linear interpolation
        count_zeros=0
        for i in range(1,len(means_smooth_extended)-1):
            if means_smooth_extended[i]==0:
                means_smooth_extended[i] = 0.5 * (means_smooth_extended[X_means_smooth_extended[i-1-count_zeros]] + means_smooth_extended[X_means_smooth_extended[i-count_zeros]]) # linear interpolation with closest non zero points
                #redefine X_mask_completed
                X_means_smooth_extended = np.nonzero(means_smooth_extended)
                X_means_smooth_extended = np.transpose(X_means_smooth_extended)


   #recurrence
   # count_zeros=0
   # for i in range(1,len(means_smooth_extended)-1):
   #     if means_smooth_extended[i]==0:
   #          means_smooth_extended[i] = 0.5*(means_smooth_extended[X_means_smooth_extended[i-1-count_zeros]] + means_smooth_extended[X_means_smooth_extended[i-count_zeros]])
   #          # redefine X_mask_extended
   #          X_mask_completed = np.nonzero(means_smooth_extended)
   #          X_mask_completed = np.transpose(X_mask_completed)
   #          #count_zeros += 1
   if verbose :
       plt.figure()

       plt.subplot(2,1,1)
       plt.plot(z_centerline,means)
       plt.plot(z_centerline,means_smooth)
       plt.title("Mean intensity")

       plt.subplot(2,1,2)
       plt.plot(z_centerline,means)
       plt.plot(means_smooth_extended)
       plt.title("Extended mean intensity")

       plt.show()

   for i in range(data.shape[2]):
       data[:,:,i] = data[:,:,i] * (mean_intensity/means_smooth_extended[i])

   hdr.set_data_dtype('uint16') # set imagetype to uint16
   # save volume
   sct.printv('\nWrite NIFTI volumes...',verbose)
   data = data.astype(np.float32, copy =False)
   img = nibabel.Nifti1Image(data, None, hdr)
   output_name = file_input+'_normalized'+ext_input
   nibabel.save(img,output_name)
   sct.printv('\n.. File created:' + output_name,verbose)

   print('\nNormalizing overall intensity...')
   # sct.run('fslmaths ' + output_name + ' -inm ' + str(mean_intensity) + ' ' + output_name)

   # to view results
   print '\nDone !'
   print '\nTo view results, type:'
   print 'fslview '+output_name+' &\n'



def usage():
   print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
Normalize intensity of the input image by smoothing the average intensity of the middle of the image by slice. If a centerline file is provided, the average intensity for each slice is calculated in the near region of the centerline.

USAGE
 """+os.path.basename(__file__)+"""  -i <input_volume>

MANDATORY ARGUMENTS
 -i <input_volume>         input volume to be processed. No Default value

OPTIONAL ARGUMENTS
 -c <centerline>            centerline file. No Default Value
 -n <mean_intensity>        mean intensity.
                            Default="""+str(param.mean_intensity)+"""
 -v {0,1}                   verbose. Default="""+str(param.verbose)+"""
 -h                         help. Show this message

EXAMPLE
 """+os.path.basename(__file__)+""" -i input_t2.nii.gz\n"""

   # exit program
   sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
   # initialize parameters
   param = param()
   # call main function
   main()

