#!/usr/bin/env python
#########################################################################################
#
# Concatenate transformations. This function is a wrapper for ComposeMultiTransform
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-07-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: also enable to concatenate reversed transfo


import sys
import os
import getopt
import sct_utils as sct
import nibabel as nib
from scipy.io import netcdf


# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1


# main
#=======================================================================================================================
def main():

    # Initialization
    fname_data = ''
    fname_out = ''
    verbose = param.verbose

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_data = '/Users/julien/mri/temp/average305_t1_tal_lin.mnc'
        fname_out = '/Users/julien/mri/temp/average305_t1_tal_lin.nii'

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:o:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            fname_data = arg
        elif opt in ('-o'):
            fname_out = arg
        elif opt in ('-v'):
            verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_data == '':
        usage()

    # Check file existence
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_data)

    # extract names
    fname_data = os.path.abspath(fname_data)
    path_in, file_in, ext_in = sct.extract_fname(fname_data)
    if fname_out == '':
        path_out, file_out, ext_out = '', file_in, '.nii'
        fname_out = path_out+file_out+ext_out
    else:
        fname_out = os.path.abspath(fname_out)
        path_out, file_out, ext_out = sct.extract_fname(fname_out)

    # open minc
    mnc = nib.minc.MincFile(netcdf.netcdf_file(fname_data, 'r'))
    img = mnc.get_scaled_data()
    affine = mnc.get_affine()

    # hdr = nib.MincImage()

    # save as nifti
    nii = nib.Nifti1Image(img, affine)
    nib.save(nii, fname_out)

    # Generate output files
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(fname_out, path_out, file_out, ext_out)

    print ''


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Convert minc to nifti.

USAGE
  """+os.path.basename(__file__)+""" -i <data>

MANDATORY ARGUMENTS
  -i <data>             input volume

OPTIONAL ARGUMENTS
  -o <output>           output volume. Add extension. Default="data".nii
  -v {0,1}              verbose. Default="""+str(param.verbose)+"""
  -h                    help. Show this message
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
