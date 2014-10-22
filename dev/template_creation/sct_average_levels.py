#!/usr/bin/env python
#########################################################################################
#
#input directory with all the landmark images. output the mask containing the labels in the template space (average)
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
        self.verbose = 1
        self.output_name = ''
        
# check if needed Python libraries are already installed or not
import sys
import getopt
import sct_utils as sct
import nibabel
import os
from numpy import mean,sort,zeros,array

def main():
    
    #Initialization
    directory = ""
    fname_template = ''
    n_l = 0 
    verbose = param.verbose
         
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:t:n:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            directory = arg 
        elif opt in ("-t"):
            fname_template = arg  
        elif opt in ('-n'):
            n_l = int(arg)                
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname_template == '' or directory == '':
        usage()
        
    # check existence of input files
    print'\nCheck if file exists ...\n'
    sct.check_file_exist(fname_template)
    
    if os.path.isdir(directory) == False :
        print 'Directory does not exists\n '
        usage()
   
    os.chdir(directory)
   
    n_i = len([name for name in os.listdir('.') if os.path.isfile(name)])  # number of landmark images

    average = zeros((n_i,n_l))
    compteur = 0
    
    for file in os.listdir(directory):
        if file.endswith(".nii.gz"):
            print file
            img = nibabel.load(file)
            data = img.get_data()
            X,Y,Z = (data>0).nonzero()
            Z = sort(Z)
            
            for i in xrange(n_l):
                average[compteur][i] = Z[i]
            
            compteur = compteur + 1
            
    average = array([int(round(mean([average[:,i]]))) for i in xrange(n_l)]) 
      
    print average     
    
    
    print '\nGet dimensions of template...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_template)
    print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
    print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'
    
    img = nibabel.load(fname_template)
    data = img.get_data()
    hdr = img.get_header()
    compteur = 0
    for i in average :
        data[int(round(nx/2.0)),int(round(ny/2.0)),i] = n_l - compteur
        compteur = compteur + 1
        
    
    print '\nSave volume ...'
    hdr.set_data_dtype('float32') # set imagetype to uint8
    # save volume
    #data = data.astype(float32, copy =False)
    img = nibabel.Nifti1Image(data, None, hdr)
    file_name = 'template_landmarks.nii.gz'
    nibabel.save(img,file_name)
    print '\nFile created : ' + file_name
    

    
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION


USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> 

MANDATORY ARGUMENTS
  -i <directory>         Directory containing ONLY the landmarks images. No default value
  -t <template_space>    Space you want to create the landmarks into. No default value
  -n <landmark_number>   Number of landmarks
  
  
OPTIONAL ARGUMENTS
  -v {0,1}                verbose.Verbose 2 for plotting. Default="""+str(param.verbose)+"""
  -h                        help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i volume.nii.gz\n"""

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






