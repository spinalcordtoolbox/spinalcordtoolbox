#!/usr/bin/env python
#########################################################################################
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
        self.verbose = 1
# check if needed Python libraries are already installed or not
import sys
import getopt
import sct_utils as sct
import nibabel
import os

def main():
    
    #Initialization
    fname = ''
    verbose = param.verbose
    start = ''
    end = ''
        
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:e:s:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname = arg
        elif opt in ('-s'):
            start = int(arg)
        elif opt in ('-e'):
            end = int(arg)                          
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname == '' :
        usage()
    
    # check existence of input files
    print'\nCheck if file exists ...'
    
    sct.check_file_exist(fname)
    
 
    # Display arguments
    print'\nCheck input arguments...'
    print'  Input volume ...................... '+fname
    print'  Verbose ........................... '+str(verbose)
    
    
    file = nibabel.load(fname)
    data = file.get_data()
    hdr = file.get_header()
    
    for i in range(start,end+1):
        data[:,:,i] = 0
    
        
  
    print '\nSave volume ...'
    hdr.set_data_dtype('float32') # set imagetype to uint8
    # save volume
    #data = data.astype(float32, copy =False)
    img = nibabel.Nifti1Image(data, None, hdr)
    file_name = 'centerline_erased.nii.gz'
    nibabel.save(img,file_name)
    


    
    
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION

Erase everythiong between to specific slices. To be used on a binary centerline or segmentation
USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> -s <start> -end <end>

MANDATORY ARGUMENTS
  -i <input_volume>         input straight cropped volume. No Default value
  -s <start>                bottom slice. No Default
  -e <end>                  higher slice. No Default
OPTIONAL ARGUMENTS
  -v {0,1}                   verbose. Default="""+str(param.verbose)+"""
 
  -h                         help. Show this message
"""

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



