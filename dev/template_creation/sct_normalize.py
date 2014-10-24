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
        
# check if needed Python libraries are already installed or not
import sys
import os
import getopt
import sct_utils as sct
import nibabel
import numpy as np
from time import strftime
import matplotlib.pyplot as plt
from scipy.interpolate import splrep,splev
from scipy import ndimage

def main():
    
    #Initialization
    fname = ''
    fname_centerline = ''
    mean_intensity = param.mean_intensity
    verbose = param.verbose
    padding = param.padding
    
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:c:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname = arg
        elif opt in ("-c"):
            fname_centerline = arg    
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname == '' or fname_centerline == '':
        usage()
    
    
    # check existence of input files
    print'\nCheck if file exists ...'
    sct.check_file_exist(fname)
    sct.check_file_exist(fname_centerline)
    
    # Display arguments
    print'\nCheck input arguments...'
    print'  Input volume ...................... '+fname
    print'  Centerline ...................... '+fname 
    print'  Verbose ........................... '+str(verbose)
    
    # Extract path, file and extension
    path_input, file_input, ext_input = sct.extract_fname(fname)
    
    
    sct.printv('\nOpen volume...',verbose)
    file = nibabel.load(fname)
    data = file.get_data()
    hdr = file.get_header()
    
    
    sct.printv('\nOpen centerline...',verbose)
    print '\nGet dimensions of input centerline...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_centerline)
    print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
    print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'
    file_c = nibabel.load(fname_centerline)
    data_c = file_c.get_data()
    
    
    X,Y,Z = (data_c>0).nonzero()
    
    min_z_index, max_z_index = min(Z), max(Z)
    
    
    z_centerline = [iz for iz in range(0, nz, 1) if data_c[:,:,iz].any() ]
    nz_nonz = len(z_centerline)
    x_centerline = [0 for iz in range(0, nz_nonz, 1)]
    y_centerline = [0 for iz in range(0, nz_nonz, 1)]
    
    print '\nGet center of mass of the centerline ...'
    for iz in xrange(len(z_centerline)):
        x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(np.array(data_c[:,:,z_centerline[iz]]))
    
    means = [0 for i in xrange(len(z_centerline))]
        
    print '\nGet mean intensity along the centerline ...'        
    for iz in xrange(len(z_centerline)):

        means[iz] =  np.mean(data[(int(round(x_centerline[iz]))-padding):(int(round(x_centerline[iz]))+padding),(int(round(y_centerline[iz]))-padding):(int(round(y_centerline[iz]))+padding),iz])
    
    
    print('\nSmoothing results with spline...')    
    m =np.mean(means)
    sigma = np.std(means)
    smoothing_param = (((m + np.sqrt(2*m))*(sigma**2))+((m - np.sqrt(2*m))*(sigma**2)))/2
    tck = splrep(z_centerline, means, s=smoothing_param)
    means_smooth = splev(z_centerline, tck)
    if verbose :
        plt.figure()
        plt.plot(z_centerline,means)
        plt.plot(z_centerline,means_smooth)
        plt.show()
    
    print('\nNormalizing intensity along centerline...')    
    for iz in xrange(len(z_centerline)):
        
        data[:,:,iz] = data[:,:,iz]*(mean_intensity/means_smooth[iz])
       
    hdr.set_data_dtype('uint8') # set imagetype to uint8
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


USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> 

MANDATORY ARGUMENTS
  -i <input_volume>         input volume to be processed. No Default value
  -c <centerline>           centerline. No Default Value             
OPTIONAL ARGUMENTS
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
