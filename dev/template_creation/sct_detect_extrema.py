#!/usr/bin/env python
#########################################################################################
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
        
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname = arg
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
    
    X,Y,Z = (data>0).nonzero()

    x_max,y_max = (data[:,:,max(Z)]).nonzero()
    x_max = x_max[0]
    y_max = y_max[0]
    z_max = max(Z)
    
    x_min,y_min = (data[:,:,min(Z)]).nonzero()
    x_min = x_min[0]
    y_min = y_min[0]
    z_min = min(Z)
    
    del data
    
    print 'Coords extrema : min [ ' + str(x_min) + ' ,' + str(y_min) + ' ,' + str(z_min) +' ] max [ ' + str(x_max) + ' ,' + str(y_max) + ' ,' + str(z_max) + ' ]' 

    return z_min,z_max
    
    
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION

Detect coordinates of minimum and maximum nonzero voxels when inputing a straight centerline.

USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> 

MANDATORY ARGUMENTS
  -i <input_volume>         straight centerline. No Default value
                            
OPTIONAL ARGUMENTS
  -v {0,1}                   verbose. Default="""+str(param.verbose)+"""
  -h                         help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i straight_centerline.nii.gz\n"""

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



