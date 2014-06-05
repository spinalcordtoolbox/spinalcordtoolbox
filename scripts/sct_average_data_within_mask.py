#!/usr/bin/env python
# ==========================================================================================
# Average data within mask. Compute a weighted average if mask is non-binary (values distributed between 0 and 1).
#
#
# USAGE
# ---------------------------------------------------------------------------------------
#   sct_average_data_within_mask.py -i <inputvol> -m <mask> [options]
#
#
# INPUT
# ---------------------------------------------------------------------------------------
# -i inputvol          volume. Can be nii or nii.gz
# -m mask              mask with values between 0 and 1. Can be nii or nii.gz
# -t numbvol           number of volume if mask is 4D.
# -z slice             number of z-slice to compute average on (other slices will not be considered).
#
#
# OUTPUT
# ---------------------------------------------------------------------------------------
# meanvalue         average of values in inputvol within mask.
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# - nibabel <http://nipy.sourceforge.net/nibabel/>
#
# EXTERNAL SOFTWARE
# none
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Author: Julien Cohen-Adad
# Modified: 2013-11-10
#
# TODO: do a zmin zmax
#
# About the license: see the file LICENSE.TXT
# ==========================================================================================

import sys
import getopt
import os
try:
    import nibabel
except ImportError, e:
    print str(e)+"\nUse the following command line for installation: sudo easy_install nibabel"
    sys.exit()


# PARAMETERS
debugging           = 0 # automatic file names for debugging



# MAIN
# ==========================================================================================
def main():

    # Initialization
    fname_src = ''
    fname_mask = ''
    tmask = ''
    zmask = ''
    verbose = 1

    # Check input parameters
    if debugging:
        fname_src  = '/Users/julien/MRI/multisite_DTI/20131011_becky/2013-11-09/input/b0_mean_reg2template.nii'
        fname_mask = '/Users/julien/matlab/toolbox/spinalcordtoolbox_dev/data/atlas/WMtracts.nii'
        tmask = '0'
        zmask = '445'
    else:
        try:
            opts, args = getopt.getopt(sys.argv[1:],'hi:m:t:z:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
            # no option supplied
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ("-i"):
                fname_src = arg
                exist_image(fname_src)
            elif opt in ("-m"):
                fname_mask = arg
                exist_image(fname_mask)
            elif opt in ("-t"):
                tmask = arg
            elif opt in ("-z"):
                zmask = arg
            elif opt in ("-v"):
                verbose = int(arg)


    # check mandatory arguments
    if fname_src == '' or fname_mask == '':
        usage()

    # print arguments
    if verbose:
        print 'Check input parameters...'
        print '.. Image:    '+fname_src
        print '.. Mask:     '+fname_mask
        print '.. tmask:    '+tmask
        print '.. zmask:    '+zmask

    # Extract path, file and extension
    #path_src, file_src, ext_src = extract_fname(fname_src)
    #path_mask, file_mask, ext_mask = extract_fname(fname_mask)

    # Quantify image within mask
    header_src = nibabel.load(fname_src)
    header_mask = nibabel.load(fname_mask)

    data_src = header_src.get_data()
    #data_mask = header_mask.get_data()[:,:,:,int(tmask)]

    # check if mask is 4D
    if tmask == '':
        data_mask = header_mask.get_data()
    else:
        data_mask = header_mask.get_data()[:,:,:,int(tmask)]

    # if user specified zmin and zmax, put rest of slices to 0
    if zmask != '':
        data_mask[:,:,:int(zmask)] = 0
        data_mask[:,:,int(zmask)+1:] = 0

    # find indices of non-zero elements the mask
    ind_nonzero = data_mask.nonzero()

    # perform a weighted average for all nonzero indices from the mask
    weighted_val = 0
    weighted_mask = 0
    for i in range(0, len(ind_nonzero[0][:])):
        # retrieve coordinates from mask
        x, y, z = ind_nonzero[0][i], ind_nonzero[1][i], ind_nonzero[2][i]
        # get values in mask
        val_mask = data_mask[x,y,z]
        # get value in the image
        val_image = data_src[x,y,z]
        # multiply value image by value mask and add iteratively
        weighted_val += val_image * val_mask
        # calculate weight iteratively
        weighted_mask += val_mask
    # divide cumulative sum value by sum of weights
    weighted_average = weighted_val / weighted_mask

    # print result
    print str(weighted_average)

    return weighted_average

    # Display created files


# Extracts path, file and extension
# ==========================================================================================
def extract_fname(fname):
    # extract path
    path_fname = os.path.dirname(fname)+'/'
    # check if only single file was entered (without path)
    if path_fname == '/':
        path_fname = ''
    # extract file and extension
    file_fname = fname
    file_fname = file_fname.replace(path_fname,'')
    file_fname, ext_fname = os.path.splitext(file_fname)
    # check if .nii.gz file
    if ext_fname == '.gz':
        file_fname = file_fname[0:len(file_fname)-4]
        ext_fname = ".nii.gz"
    return path_fname, file_fname, ext_fname


# Print usage
# ==========================================================================================
def usage():
    path_func, file_func, ext_func = extract_fname(sys.argv[0])
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Average data within mask. Compute a weighted average if mask is non-binary (values distributed between 0 and 1).\n' \
        '\n' \
        'USAGE\n' \
        '   '+file_func+ext_func+' -i <inputvol> -m <mask>\n\n' \
        'Options:' \
        '  -i inputvol          image to extract values from\n' \
        '  -m mask              mask with values between 0 and 1. If non-binary, a weighted average will be done\n' \
        '  -t numbvol           number of volume if mask is 4D.\n' \
        '  -z slice             number of z-slice to compute average on (other slices will not be considered)\n' \
        '  -v verbose           verbose. 0 or 1. (default=1).\n' \
        '\n' \

    sys.exit(2)

# Check existence of a file
# ==========================================================================================
def exist_image(fname):
    if os.path.isfile(fname) or os.path.isfile(fname+'.nii') or os.path.isfile(fname+'.nii.gz'):
        pass
    else:
        print('\nERROR: '+fname+' does not exist. Exit program.\n')
        sys.exit(2)



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    main()