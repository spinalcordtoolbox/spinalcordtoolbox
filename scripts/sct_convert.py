#!/usr/bin/env python
#########################################################################################
#
# Module converting image files
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Created: 2014-09-22
#
# Dependences:
#   minc-toolkit - http://www.bic.mni.mcgill.ca/ServicesSoftware/ServicesSoftwareMincToolKit
#
# TO DO:
# - check if minc-toolkit is installed. If not, convert files using nibabel
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys
import commands
import getopt
import sct_utils as sct
import nibabel as nib


# DEFAULT PARAMETERS
class Param:
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
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:o:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
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
    
    cmd = 'which mnc2nii'
    status, output = commands.getstatusoutput(cmd)
    if not output:
        print 'ERROR: minc-toolkit not installed...'
        sys.exit(2)

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
        path_in, file_in, ext_in = sct.extract_fname(fname_data)
        path_out, file_out, ext_out = sct.extract_fname(fname_out)

    if ext_in=='.nii.gz':
        print "Uncompressing input file..."
        sct.run("gunzip -c "+fname_data+" >"+path_in+file_in+".nii")
        ext_in='.nii'
        fname_data=path_in+file_in+ext_in

    if ext_in=='.nii' and ext_out=='.mnc':
        nii2mnc(fname_data,fname_out)
    elif ext_in=='.mnc' and ext_out=='.nii':
        mnc2nii(fname_data,fname_out)
    elif ext_in=='.nii' and ext_out=='.header':
        nii2volviewer(fname_data,fname_out)
    elif ext_in=='.mnc' and ext_out=='.header':
        mnc2volviewer(fname_data,fname_out)


# Convert file from nifti to minc
# ==========================================================================================
def nii2mnc(fname_data,fname_out):
    print "Converting from nifti to minc"
    sct.run("nii2mnc "+fname_data+" "+fname_out)
    

# Convert file from minc to nifti
# ==========================================================================================
def mnc2nii(fname_data,fname_out):
    print "Converting from minc to nifti"
    sct.run("mnc2nii "+fname_data+" "+fname_out)

# Convert file from nifti to volumeviewer
# ==========================================================================================
def nii2volviewer(fname_data,fname_out):
    print "Converting from nifti to volume viewer"
    path_in, file_in, ext_in = sct.extract_fname(fname_data)
    path_out, file_out, ext_out = sct.extract_fname(fname_out)
    fname_data_nii = path_out+file_out+'.mnc'
    nii2mnc(fname_data,fname_data_nii)
    mnc2volviewer(fname_data_nii,path_out+file_out)

# Convert file from minc to volumeviewer
# ==========================================================================================
def mnc2volviewer(fname_data,fname_out):
    print "Converting from minc to volume viewer"
    sct.run("isct_minc2volume-viewer.py "+fname_data+" -o "+fname_out)


# Print usage
# ==========================================================================================
def usage():
    print """
        """+os.path.basename(__file__)+"""
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
            
            DESCRIPTION
            Convert files from nifti to minc, minc to nifti or nifti to volume viewer.
            
            USAGE
            """+os.path.basename(__file__)+""" -i <data>
                
                MANDATORY ARGUMENTS
                -i <data>             input volume
                
                OPTIONAL ARGUMENTS
                -o <output>           output volume. Add extension. Default="data".nii
                -v {0,1}              verbose. Default="""+str(param_default.verbose)+"""
                    -h                    help. Show this message
                    """
    # exit program
    sys.exit(2)



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()