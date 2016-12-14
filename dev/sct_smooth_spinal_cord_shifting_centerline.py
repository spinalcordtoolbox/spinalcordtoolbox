#!/usr/bin/env python

## @package sct_smooth_spinal_cord_shifting_centerline.py
#
# - from spinal cord MRI volume 3D (as nifti format) and centerline of the spinal cord (given by sct_get_centerline.py)
#
#
# Description about how the function works:
#
#
#
# USAGE
# ---------------------------------------------------------------------------------------
# sct_smooth_spinal_cord_shifting_centerline.py -i <input_image> -c <centerline>
#
# MANDATORY ARGUMENTS
# ---------------------------------------------------------------------------------------
#   -i       input volume.
#   -c       centerline.
#
# OPTIONAL ARGUMENTS
# ---------------------------------------------------------------------------------------
#   -f       option to choose the centerline fitting method: 'splines' to fit the
#            centerline with splines, 'polynome' to fit the centerline with a polynome
#            (default='splines')
#
#
# EXAMPLES
# ---------------------------------------------------------------------------------------
#   sct_smooth_spinal_cord_shifting_centerline.py -i t2.nii.gz -c centerline.nii.gz
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# - nibabel: <http://nipy.sourceforge.net/nibabel/>
# - numpy: <http://www.numpy.org>
# - scipy: <http://www.scipy.org>
#
# EXTERNAL SOFTWARE
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymt.ca>
# Authors: Simon LEVY
# Modified: 2014-07-05
#
# License: see the LICENSE.TXT
#=======================================================================================================================

import scipy
import getopt
import sys
import time
import math
from scipy.interpolate import splprep, splev
from sct_nurbs import NURBS
import sct_utils as sct
import numpy

# check if needed Python libraries are already installed or not
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
try:
    from scipy import ndimage
except ImportError:
    print '--- scipy not installed! Exit program. ---'
    sys.exit(2)




#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization
    fname_anat = ''
    fname_centerline = ''
    centerline_fitting = ''
    start_time = time.time()

    # Parameters initialization
    # Width and length of xy zone to smooth in terms of voxels
    x_width_sc = 15
    y_width_sc = 17
    # Length of the vector to smooth in terms of number of voxels
    smooth_extend = 81
    # Standard deviation for Gaussian kernel (in terms of number of voxels) of the filter to apply
    sigma = 16

    # Check input param
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:c:f:')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            fname_anat = arg
        elif opt in ('-c'):
            fname_centerline = arg
        elif opt in ('-f'):
            centerline_fitting = str(arg)


    # Display usage if a mandatory argument is not provided
    if fname_anat == '' or fname_centerline == '':
        print '\n \n All mandatory arguments are not provided \n \n'
        usage()

    # Display usage if optional arguments are not correctly provided
    if centerline_fitting == '':
        centerline_fitting = 'splines'
    elif not centerline_fitting == '' and centerline_fitting == 'splines' and centerline_fitting == 'polynome':
        print '\n \n -f argument is not valid \n \n'
        usage()

    # Check existence of input files
    sct.check_file_exist(fname_anat)
    sct.check_file_exist(fname_centerline)

    # Extract path/file/extension of the original image file
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    # Extract path/file/extension of the centerline
    path_centerline, file_centerline, ext_centerline = sct.extract_fname(fname_centerline)

    # Get input image orientation
    status, output = sct.run('sct_orientation -i ' + fname_anat + ' -get')
    input_image_orientation = output[-3:]
    # Get centerline orientation
    status, output = sct.run('sct_orientation -i ' + fname_centerline + ' -get')
    centerline_orientation = output[-3:]

    # Display arguments
    print '\nCheck input arguments...'
    print '.. Anatomical image:           ' + fname_anat
    print '.... orientation:              ' + input_image_orientation
    print '.. Centerline:                 ' + fname_centerline
    print '.... orientation:              ' + centerline_orientation
    print '.... Centerline fitting option:' + centerline_fitting


    # Change orientation of the input image into RPI
    print '\nOrient input volume to RPI orientation...'
    fname_anat_orient = path_anat+ file_anat+'_rpi'+ ext_anat
    sct.run('sct_orientation -i ' + fname_anat + ' -o ' + fname_anat_orient + ' -orientation RPI')
    # Change orientation of the input image into RPI
    print '\nOrient centerline to RPI orientation...'
    fname_centerline_orient = path_centerline + file_centerline +'_rpi' + ext_centerline
    sct.run('sct_orientation -i ' + fname_centerline + ' -o ' + fname_centerline_orient + ' -orientation RPI')

    # Read nifti anat file
    img = nibabel.load(fname_anat_orient)
    # 3d array for each x y z voxel values for the input nifti image
    data_anat = img.get_data()
    hdr_anat = img.get_header()

    # Read nifti centerline file
    img = nibabel.load(fname_centerline_orient)
    # 3d array for each x y z voxel values for the input nifti image
    data_centerline = img.get_data()
    hdr_centerline = img.get_header()

    # Get dimensions of input centerline
    print '\nGet dimensions of input centerline...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(str(fname_centerline_orient))

    # Make a copy of the anatomic image data to be smoothed
    data_anat_smoothed=numpy.copy(data_anat)
    data_anat_smoothed = data_anat_smoothed.astype(float)

    #Loop across z and associate x,y coordinate with the point having maximum intensity
    x_centerline = [0 for iz in range(0, nz, 1)]
    y_centerline = [0 for iz in range(0, nz, 1)]
    z_centerline = [iz for iz in range(0, nz, 1)]
    for iz in range(0, nz, 1):
        x_centerline[iz], y_centerline[iz] = np.unravel_index(data_centerline[:, :, iz].argmax(),
                                                              data_centerline[:, :, iz].shape)
    del data_centerline


    # Fit the centerline points with the kind of curve given as argument of the script and return the new smoothed coordinates
    if centerline_fitting == 'splines':
        x_centerline_fit, y_centerline_fit = b_spline_centerline(x_centerline,y_centerline,z_centerline)
    elif centerline_fitting == 'polynome':
        x_centerline_fit, y_centerline_fit = polynome_centerline(x_centerline,y_centerline,z_centerline)



    # Loop accross the z-slices of the spinal cord
    for z_cl in range(0,nz):

        # Find the nearest coordinates of the centerline point at this z-slice in the anatomic image
        x_cl_anat = round(x_centerline_fit[z_cl])
        y_cl_anat = round(y_centerline_fit[z_cl])



        # Loop accross the points of defined zone of the plane z=z_cl in the anatomic image plane
        for x in range(-int(math.floor(x_width_sc/2)+1),int(math.floor(x_width_sc/2)+1),1):
            for y in range(-int(math.floor(y_width_sc/2)+1),int(math.floor(y_width_sc/2)+1),1):

                # Initialization of the vector to smooth
                vector_to_smooth = numpy.zeros((1,smooth_extend))
                # Filling of the vector to smooth
                for i in range(0,smooth_extend):

                    # Calculate the z coordinate of the slice below and above the considered anatomic z-slice
                    zi = -int(math.floor(smooth_extend/2)) + i + z_cl
                    # CASE: slices on the edges of the image are not taken into account
                    if zi<0 or zi>=nz:
                        zi=0

                    # Calculate the (x,y)
                    x_point_to_interpolate = x_cl_anat + x + (x_centerline_fit[zi]-x_centerline_fit[z_cl])
                    y_point_to_interpolate = y_cl_anat + y + (y_centerline_fit[zi]-y_centerline_fit[z_cl])


                    vector_to_smooth[0][i] = bilin_interpol(data_anat,x_point_to_interpolate,y_point_to_interpolate,zi)

                # Smooth the vector
                vector_smoothed = scipy.ndimage.filters.gaussian_filter1d(vector_to_smooth[0],sigma)

                # Replace the original value by the smoothed value
                data_anat_smoothed[x_cl_anat+x,y_cl_anat+y,z_cl] = vector_smoothed[int(math.floor(smooth_extend/2))]


    # Return the nifti corrected data
    hdr_anat.set_data_dtype('uint8') # set imagetype to uint8
    print '\nWrite NIFTI volumes...'
    img = nibabel.Nifti1Image(data_anat_smoothed, None, hdr_anat)
    nibabel.save(img, 'tmp.anat_smoothed.nii')
    fname_output_image = sct.generate_output_file('tmp.anat_smoothed.nii', './', file_anat+'_centerline_shift_smoothed', ext_anat)

    # Reorient the output image into its initial orientation
    print '\nReorient the output image into its initial orientation...'
    sct.run('sct_orientation -i '+fname_output_image +' -o ' +fname_output_image+' -orientation '+input_image_orientation)

    # Delete temporary files
    print '\nDelete temporary files...'
    sct.run('rm '+fname_anat_orient)
    sct.run('rm '+fname_centerline_orient)



    #Display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished!'
    print '.. '+str(int(round(elapsed_time)))+'s\n'




#=======================================================================================================================
# Functions used in the main
#=======================================================================================================================

def find_index_of_nearest_xy(x_array,y_array,x_point,y_point):

    distance2 = (x_array[0]-x_point)**2 + (y_array[0]-y_point)**2
    idx_near = idy_near = 0

    for idx in range(0,len(x_array)):
        for idy in range(0,len(y_array)):
            distance2_old = np.copy(distance2)
            distance2 = (x_array[idx]-x_point)**2 + (y_array[idy]-y_point)**2
            if distance2<distance2_old:
                idx_near = np.copy(idx)
                idy_near = np.copy(idy)

    return idx_near,idy_near



def bilin_interpol(data_interpolating,x_point_to_interpolate,y_point_to_interpolate,z_plane):

    # Lattice point below the point to interpolate in the two dimensions
    x0=math.floor(x_point_to_interpolate)
    y0=math.floor(y_point_to_interpolate)
    # Lattice point above the point to interpolate in the three dimensions
    x1=math.ceil(x_point_to_interpolate)
    y1=math.ceil(y_point_to_interpolate)


    # Let xd ad yd be the differences between the point to interpolate and the smaller coordinate related
    # in the x and y dimensions

    # Avoid the division by 0 if x0=x1 or y0=y1
    if x1 == x0:
        xd = 0
    else:
        xd = (x_point_to_interpolate - x0)/(x1 - x0)

    if y1 == y0:
            yd = 0
    else:
        yd = (y_point_to_interpolate - y0)/(y1 - y0)


    # First we interpolate along x
    v0 = data_interpolating[x0,y0,z_plane]*(1 - xd) + data_interpolating[x1,y0,z_plane]*xd
    v1 = data_interpolating[x0,y1,z_plane]*(1 - xd) + data_interpolating[x1,y1,z_plane]*xd
    # Then we interpolate along y
    v = v0*(1 - yd) + v1*yd

    return v



def spline_centerline(x_centerline,y_centerline,z_centerline):
    """Fit splines to the centerline points given as argument and return the new coordinates
    """

    s=30 # Smoothness parameter
    k=3 # Spline order. k must be in [1,5]
    nest=-1 # Estimate of number of knots needed (-1 = maximal)

    # Find the knot points
    tckp,u = splprep([x_centerline,y_centerline,z_centerline],s=s,k=k,nest=nest)

    # Evaluate spline, including interpolated points
    x_centerline_fit,y_centerline_fit,z_centerline_fit = splev(u,tckp)

    # Find the (x,y) coordinates of the fitted curved corresponding to the original z-slices
    corresponding_id = numpy.zeros(len(z_centerline))
    for idz in z_centerline:
        distance2_z_zfit = (z_centerline_fit-z_centerline[idz])**2
        corresponding_id[idz] = numpy.where(distance2_z_zfit==distance2_z_zfit.min())[0][0]

    x_centerline_fit_match = numpy.zeros(len(z_centerline))
    y_centerline_fit_match = numpy.zeros(len(z_centerline))

    for k in range(0,len(corresponding_id)):
        x_centerline_fit_match[k]=x_centerline_fit[k]
        y_centerline_fit_match[k]=y_centerline_fit[k]

    return x_centerline_fit_match,y_centerline_fit_match



def b_spline_centerline(x_centerline,y_centerline,z_centerline):
    """Give a better fitting of the centerline than the method 'spline_centerline' using b-splines"""


    points = [[x_centerline[n],y_centerline[n],z_centerline[n]] for n in range(len(x_centerline))]

    nurbs = NURBS(3,1000,points) # for the third argument (number of points), give at least len(z_centerline)
    # (len(z_centerline)+500 or 1000 is ok)
    P = nurbs.getCourbe3D()
    x_centerline_fit=P[0]
    y_centerline_fit=P[1]

    return x_centerline_fit, y_centerline_fit



def polynome_centerline(x_centerline,y_centerline,z_centerline):
    """Fit polynomial function through centerline"""

    # Fit centerline in the Z-X plane using polynomial function
    print '\nFit centerline in the Z-X plane using polynomial function...'
    coeffsx = np.polyfit(z_centerline, x_centerline, deg=5)
    polyx = np.poly1d(coeffsx)
    x_centerline_fit = np.polyval(polyx, z_centerline)

    #Fit centerline in the Z-Y plane using polynomial function
    print '\nFit centerline in the Z-Y plane using polynomial function...'
    coeffsy = np.polyfit(z_centerline, y_centerline, deg=5)
    polyy = np.poly1d(coeffsy)
    y_centerline_fit = np.polyval(polyy, z_centerline)


    return x_centerline_fit,y_centerline_fit


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
          'sct_smooth_spinal_cord_shifting_centerline.py\n' \
          '-------------------------------------------------------------------------------------------------------------\n' \
          'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
          '\n' \
          'DESCRIPTION\n' \
          '  This program smooth the input anatomical image along the spinal cord. It works by applying a convolution ' \
          'of a 1D kernel with the anatomical MRI image along the spinal cord. The output is the smoothed anatomical ' \
          'image and RPI oriented.' \
          '\n' \
          'USAGE\n' \
          '  sct_smooth_spinal_cord_shifting_centerline.py -i <input_image> -c <centerine>\n' \
          '\n' \
          'MANDATORY ARGUMENTS\n' \
          '  -i <anat>         anatomic nifti file. Image to smooth.\n' \
          '  -c <centerline>   centerline.\n' \
          '\n' \
          'OPTIONAL ARGUMENTS\n' \
          '  -h                help. Show this message.\n' \
          '  -f                option to choose the centerline fitting method: \'splines\' to fit the centerline with' \
                               'splines, \'polynome\' to fit the centerline with a polynome (default=\'splines\').\n'

    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()
