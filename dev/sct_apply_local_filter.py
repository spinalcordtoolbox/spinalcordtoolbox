#!/usr/bin/env python
# ----------------------------------------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# - nibabel : <http://nipy.sourceforge.net/nibabel/>
# - numpy   : <http://www.numpy.org>
# - scipy
#
# ----------------------------------------------------------------------------------------------------------------------
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Authors: Simon LEVY
#
# License: see the LICENSE.TXT
# ======================================================================================================================

# BECAREFUL, we assume here that we have a RPI orientation, where Z axis is inferior-superior direction

import os
import getopt
import sys
import sct_utils as sct
import scipy.ndimage


try:
    import nibabel
except ImportError:
    print '--- nibabel not installed! Exit program. ---'
    sys.exit(2)
try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! Exit program. ---'
    sys.exit(2)

#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Variable initialization
    strategy = ""
    fname_centerline = ""
    fname_input_image = ""
    fname_output_image = ""
    fname_mask = ""


    # extract path of the script
    path_script = os.path.dirname(__file__) + '/'

    # Check input param
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:o:m:f:s:c:')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            fname_input_image = arg
        elif opt in ('-o'):
            fname_output_image = arg
        elif opt in ('-m'):
            fname_mask = arg
        elif opt in ('-f'):
            filter_type = str(arg)
        elif opt in ('-s'):
            strategy = str(arg)
        elif opt in ('-c'):
            fname_centerline = arg

    # display usage if a mandatory argument is not provided
    if fname_input_image == '' or fname_mask == '' or (strategy=="along_centerline" and fname_centerline==""):
        print("\n \n \n All mandatory arguments are not provided \n \n \n")
        usage()

    # check existence of input files
    sct.check_file_exist(fname_input_image)
    sct.check_file_exist(fname_mask)
    if strategy == "along_centerline":
        sct.check_file_exist(fname_centerline)

    # extract path/file/extension
    path_input_image, file_input_image, ext_input_image = sct.extract_fname(fname_input_image)
    path_output_image, file_output_image, ext_output_image = sct.extract_fname(fname_output_image)

    # read nifti input file
    img = nibabel.load(fname_input_image)
    # 3d array for each x y z voxel values for the input nifti image
    data = img.get_data()
    hdr = img.get_header()

    # read nifti mask file
    mask = nibabel.load(fname_mask)
    # 3d array for each x y z voxel values for the input nifti mask
    mask_data = mask.get_data()
    mask_hdr = mask.get_header()

    # Compute the image to extract the smoothed spinal cord data from according to the chosen strategy
    if strategy == "mean_per_slice":
        print("\n \nThe smoothing strategy is to apply the smoothing to an image of the spinal cord completed"
              " with the mean value of the spinal cord for each z-slice...\n \n")
        data = smooth_mean_per_slice(data, mask_data)
    elif strategy == "along_centerline":
        print("\n \nThe smoothing strategy is to apply the smoothing to the original data along the spinal cord "
              "in the direction of the centerline...\n \n")
        data = smooth_along_centerline(data, fname_input_image, file_input_image, ext_input_image, mask_data,
                                       fname_centerline)
    elif strategy == "total_image" or "":
        print("\n \nThe smoothing strategy is to apply the smoothing to the entire original image...\n \n")
        data = smooth_total(data,mask_data)
    else:
        print("\n \nThe smoothing strategy is not correct\n \n")
        usage()



    # Return the nifti corrected data
    hdr.set_data_dtype('uint8') # set imagetype to uint8
    print '\nWrite NIFTI volumes...'
    img = nibabel.Nifti1Image(data, None, hdr)
    nibabel.save(img, 'tmp.' + file_output_image + '.nii')
    sct.generate_output_file('tmp.' + file_output_image + '.nii', './', file_output_image, ext_output_image)


#=======================================================================================================================
# Functions used in the main
#=======================================================================================================================
def apply_filter(data, filter_type='gaussian'):
    """Apply the chosen filter to the image"""

    if filter_type == 'gaussian':
        print '\nApply a Gaussian filter...'
        sigma = 1 # standard deviation for Gaussian kernel
        data_filtered = scipy.ndimage.filters.gaussian_filter(data, sigma)
    return data_filtered


def smooth_mean_per_slice(data, mask_data):
    """Apply the smoothing to an image of the spinal cord completed with the mean value of
    the spinal cord for each z-slice and return the original data with the smoothed spinal cord"""

    # Create a new image keeping only the spinal cord and assigning to the other voxels the mean value across the spinal cord

    # Find the voxels that belong to the spinal cord
    X, Y, Z = (mask_data > 0).nonzero()

    # Define useful variables
    N = len(X) # number of voxels in the spinal cord segmentation
    Z_min = min(Z) # min z of the segmentation
    Z_max = max(Z) # max z of the segmentation
    Z_nb = Z_max - Z_min + 1 # number of different z-slice of the segmentation
    x = len(data) # number of voxels in the X direction
    y = len(data[1]) # number of voxels in the Y direction
    z = len(data[1][1]) # number of voxels in the Z direction

    ## Count the number of voxels belonging to the spinal cord for each slice according to the segmentation
    #nb_vox_per_slice = [0 for i in range(0,Z_nb)] # initialization
    #z_index = 0
    #while z_index < Z_nb:
    #    nb_vox_per_slice[z_index]+= 1
    #    z_index+= 1


    # Sort by z-slice the values of the voxels belonging to the spinal cord
    sc_values = [[] for Z_index in range(0, Z_nb)] # initialization
    for vox_index in range(0, N):
        sc_values[Z[vox_index] - Z_min].append(data[X[vox_index]][Y[vox_index]][Z[vox_index]])

    # Compute the mean value for each slice of the spinal cord
    print '\nCompute the mean value for each slice of the spinal cord...'
    sc_mean_per_slice = [0 for Z_index in range(0, Z_nb)] # initialization
    for Z_index in range(0, Z_nb):
        sc_mean_per_slice[Z_index] = sum(sc_values[Z_index]) / len(sc_values[Z_index])

    # Define a new image assigning to all the voxels that don't belong to the mean value across their slice
    print '\nCreate a new image to smooth keeping only the spinal cord and completing with the previously computed mean values...'
    sc_data = [[[0 for k in range(0, z)] for j in range(0, y)] for i in
               range(0, x)] # initialization by the size of the original data
    for k in range(0, z):
        for j in range(0, y):
            for i in range(0, x):

                if k < Z_min:
                    sc_data[i][j][k] = sc_mean_per_slice[0]
                elif Z_min <= k <= Z_max:
                    sc_data[i][j][k] = sc_mean_per_slice[k - Z_min]
                elif k > Z_max:
                    sc_data[i][j][k] = sc_mean_per_slice[len(sc_mean_per_slice) - 1]

    # Assign the right value to the voxels that belong to the spinal cord
    for i in range(0, N):
        sc_data[X[i]][Y[i]][Z[i]] = data[X[i]][Y[i]][Z[i]]

    # Apply the filter to these new image
    smoothed_sc_data = apply_filter(sc_data)

    # Replace the original spinal cord data by the smoothed one in the original data
    for i in range(0, N):
        data[X[i]][Y[i]][Z[i]] = smoothed_sc_data[X[i]][Y[i]][Z[i]]

    # Return the corrected data
    return data


def smooth_total(data, mask_data):
    """Apply the smoothing to the original data and return the original data where the original
    spinal cord was replaced by the smoothed spinal cord"""

    # Find the voxels that belong to the spinal cord
    X, Y, Z = (mask_data > 0).nonzero()

    # Define useful variable
    N = len(X) # number of voxels in the spinal cord segmentation

    # Apply the filter to the original data
    smoothed_sc_data = apply_filter(data)

    # Replace the original spinal cord data by the smoothed one in the original data
    for i in range(0, N):
        data[X[i]][Y[i]][Z[i]] = smoothed_sc_data[X[i]][Y[i]][Z[i]]

    # Return the corrected data
    return data


def smooth_along_centerline(data, fname_input_image, file_input_image, ext_input_image, mask_data, fname_centerline):
    """Apply the smoothing to the original data along the spinal cord in the direction of the centerline and return
    the original data where the original spinal cord was replaced by the smoothed spinal cord"""

    # Find the voxels that belong to the spinal cord
    X, Y, Z = (mask_data > 0).nonzero()

    # Define useful variable
    N = len(X) # number of voxels in the spinal cord segmentation

    # Apply the script "sct_smooth_spinal_cord.py" to the original image
    print("\n \n \n Apply the script \"sct_smooth_spinal_cord.py\" to the original image\n \n ")
    os.system("python sct_smooth_spinal_cord.py -i " + str(fname_input_image) + " -c " + str(fname_centerline))

    # Read the nifti output file resulting from the previously run script
    print("\n \n Loading"+"./" + str(file_input_image) + "_smoothed" + str(ext_input_image)+"\n \n")
    smoothed_img = nibabel.load("./" + str(file_input_image) + "_smoothed" + str(ext_input_image))
    # 3d array for each x y z voxel values for the nifti image
    smoothed_sc_data = smoothed_img.get_data()

    # Replace the original spinal cord data by the smoothed one in the original data
    for i in range(0, N):
        data[X[i]][Y[i]][Z[i]] = smoothed_sc_data[X[i]][Y[i]][Z[i]]

    # Return the corrected data
    return data


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print 'USAGE: \n' \
          '  sct_apply_local_filter.py -i <inputimage> -o <outputimage> -m <mask> [options]\n' \
          '\n' \
          'MANDATORY ARGUMENTS\n' \
          '  -i           input volume.\n' \
          '  -o           output volume.\n' \
          '  -m           binary mask refering to zone where to apply the filter.\n' \
          '\n' \
          'OPTIONAL ARGUMENTS\n' \
          '  -h           help. Show this message.\n' \
          '  -f           type of filter to apply (default=\"gaussian\")\n' \
          '  -s          smoothing strategy: either \"total_image\", \"mean_per_slice\" or \"along_centerline\"' \
          ' (default=\"total image\")\n' \
          '  -c           centerline of the input spinal cord image if \"along_centerline\" is given as -s argument' \
          '\n' \
          '\n' \
          'EXAMPLE:\n' \
          '  sct_apply_local_filter.py -i t2.nii.gz -o t2_filtered_WM.nii.gz -m t2_seg.nii.gz -f median\n'
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()
