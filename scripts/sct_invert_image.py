#!/usr/bin/env python
#########################################################################################
#
# Register anatomical image to the template using the spinal cord centerline/segmentation.
# we assume here that we have a RPI orientation, where Z axis is inferior-superior direction
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2014-06-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: currently it seems like cross_radius is given in pixel instead of mm

import os, sys
import getopt
import commands
import sys
import sct_utils as sct
import nibabel
import numpy as np

# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug               = 0


#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization
    fname_input = ''
    fname_output = ''

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    
    # Check input param
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:o:c:r:t:l:d')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            fname_input = arg
        elif opt in ('-o'):
            fname_output = arg

    # display usage if a mandatory argument is not provided
    if fname_output == '':
        usage()
        
    # check existence of input files
    sct.check_file_exist(fname_input)
    
    # extract path/file/extension
    path_input, file_input, ext_input = sct.extract_fname(fname_input)
    path_output, file_output, ext_output = sct.extract_fname(fname_output)

    # read nifti input file
    img = nibabel.load(fname_input)
    # 3d array for each x y z voxel values for the input nifti image
    data = img.get_data()
    hdr = img.get_header()

    print data.max()
    data = data.max()-data

    hdr.set_data_dtype('int32') # set imagetype to uint8, previous: int32.
    print '\nWrite NIFTI volumes...'
    data.astype('int')
    img = nibabel.Nifti1Image(data, None, hdr)
    nibabel.save(img, fname_output)

#=======================================================================================================================
def display_voxel(data):
    # the Z image is assume to be in second dimension
    X, Y, Z = (data > 0).nonzero()
    for k in range(0,len(X)):
        print 'Position=('+str(X[k])+','+str(Y[k])+','+str(Z[k])+') -- Value= '+str(data[X[k],Y[k],Z[k]])

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print 'USAGE: \n' \
        '  sct_label_utils -i <inputdata> -o <outputdata> -c <crossradius>\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i           input volume.\n' \
        '  -o           output volume.\n' \
        '  -t           process: cross, remove, display-voxel\n' \
        '  -c           cross radius in mm (default=5mm).\n' \
        '  -r           reference image for label removing' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -h           help. Show this message.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  sct_label_utils -i t2.nii.gz -c 5\n'
    sys.exit(2)
    
    
#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
