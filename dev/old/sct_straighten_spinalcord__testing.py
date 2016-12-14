#!/usr/bin/env python

## @package sct_straighten_spinalcord
#
# - run a gaussian weighted slice by slice registration over a machine to straighten the spinal cord.
# - use the centerline previously calculated during the registration but fitted with spline interpolation to improve the spinal cord straightening
# - find the corresponding warping field (non-linear transformation) to straighten the spine relatively to each orthogonal plane of the centerline fitted
#
#
# Description about how the function works:
#
# 1. slice-wise realignement
# ------------------------------------------------
# input: center
# the algorithm iterates in the upwards and then downward direction (along Z). For each direction, it registers the slice i+1 on the slice i. It does that using a gaussian mask, centers on the spinal cord, and is applied on both the i and i+1 slices (inweight and refweight from flirt).
# NB: the code works on APRLIS
# output:
# - centerline
# - image straightened (but badly straightened because only applied translation plane-wise, therefore there is distortion of structure for high curvature)
# - gaussian mask
# - transformation matrices Tx,Ty
#
# 2. smoothing of the centerline
# ------------------------------------------------
# input: centerline, transfo matrices
# it fits in 3d (using an independent decomposition of the planes XZ and YZ) a spline function of order 1.
# apply the smoothed transformation matrices to the image --> gives a "smoothly" straighted spinal cord (but stil with errors due to deformation of strucutre related to curvatyre)
# output:
# - centerline fitted
# - transformation matrices Tx,Ty smoothed
# - straighted spinal cord
#
# 3. estimation of warping
# ------------------------------------------------
# input: centerline smooth
# parametrize the centerline, i.e., finds its equation
# create two landmark images:
# - source landmarks: landmarks that represents a cross (i.e., 5 landmarks to define the cross), the cross being centered at the centerline, and oriented along a plane orthogonal to the centerline (calculated using the above equation).
# - destination landmarks: landmarks that represents a cross, the cross being centered at a vertical centerline (the one we want to register the input centerline to), and oriented along AP and RL.
# N.B: a "landmark" corresponds to a voxel with a given intereger value. : e.g.,  that are positioned at equal distance
#
# one the two sets of crosses are created, a volume representing the straightened spinal cord is generated. The way this volume is generated is by:
# - sampling the smooth centerline at a specific step. This step correponds to the size of the pixel in the z direction. E.g., for a 1x1x1mm acquisition, the step is 1 mm.
# - for each sample, the plane orthogonal to the centerline will be used to created a new slice in a destination volume. The output of this procedure is a stack of slices orthogonal to the centerline. Note that the destination (straight) centerline is positioned at the XY center of the destination volume.
# NB: The advantage of this approach is that, converserly to the previously straightened spinal cord, which had unwanted curvature-dependent deformations, this reconstruction will correctly XXX reconstruct the spinal cord by straightening it along its curviline absissa. Moreover, as opposed to the previous attempts for straightening the spianl cord [ref: horsfield], the step along the centerline will NOT correpond to the z of the input image, but to a fixed step based on the native z-voxel resoltution. This is importnat, as in case of large curvature, a non-fixed step (i.e., based on the z of the input image) will result in compression/extension of the structure. Which is not desirable, if the end goal is to register the spinal cord to an anatomical template.
#
# Once the destination volume with the straightened spinal cord is created, the volumes are padded by 50 voxels on the top and bottom of the volume. The reason for padding, is that, from our experience, when ANTS generates a deformation field, this deformation field is constrainted at the edges of the images, i.e., without padding, there would be no effective deformation at Z=0 and Z=end. The padded volumes are the following:
# - src
# - src landmarks ("orthogonal_landmarks" in the code)
# - dest (straightened spinal cord-- using orthogonal interpolation)
# - dest landmarks ("horizontal_landmarks" in the code)
#
# When the above things are done, ANTS is applied to estimate a deformation field. The method employed are: PSE (landmark-based) and CC (using the src and dest images).
# The warping field is then applied to the src image in order to give the user the straightened spinal cord.
# output:
# - warping field (and inverted) + affine
# - straighnted spinal cord
#
#
# USAGE
# ---------------------------------------------------------------------------------------
# sct_straighten_spinalcord.py -i <data> -p <binary>
#
#  - -h       help. Show this message.
#
# MANDATORY ARGUMENTS
# ---------------------------------------------------------------------------------------
#  - -i       anatomic nifti file. Image to straighten.
#  - -p       binary nifti file. Image used as initialization for the straightening process.
#
# OPTIONAL ARGUMENTS
# ---------------------------------------------------------------------------------------
#  - -o       nifti file. Spinal cord straightened using slice-by-slice gaussian weighted registration.
#  - -m       nifti file. Gaussian mask centered along the spinal cord.
#  - -g [gap] integer. Gap between slices used for registration. [Default: gap=1].
#  - -d [deform 0|1]  deformation field. Determine a non-linear transformation (Warping field + Affine transform) to straighten the spinal cord based on an orthogonal centerline plane resampling approach. [Default: deform=0].
#
# EXAMPLES
# ---------------------------------------------------------------------------------------
# - Straighten the spine using only a slice-by-slice gaussian-weighted registration. For example:
#     sct_straighten_spinalcord.py -i t2.nii.gz -p binary_t2.nii.gz
#
# - Straighten the spine using a slice-by-slice gaussian-weighted registration and a fitted centerline (spline interpolation). For example:
#     sct_straighten_spinalcord.py -i t2.nii.gz -p binary_t2.nii.gz -r 1
#
# - Find the warping transformation (warping filed + affine transform) to straighten the spine using an orthogonal centerline planes resampling approach. For example:
#     sct_straighten_spinalcord.py -i t2.nii.gz -p binary_t2.nii.gz -d 1
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# - nibabel: <http://nipy.sourceforge.net/nibabel/>
# - scipy: <http://www.scipy.org>
# - numpy: <http://www.numpy.org>
# - sympy: <http://sympy.org/en/index.html>
# - PIL: <http://www.pythonware.com/products/pil/>
#
# EXTERNAL SOFTWARE
# - FSL: <http://fsl.fmrib.ox.ac.uk/fsl/>
# - ANTs: <http://stnava.github.io/ANTs/>
# - sct_orientation: get the spatial orientation of an input image
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Author: Geoffrey LEVEQUE
# Modified: 2013-12-06
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# TODO: output png of sagittal slice centered at hte middle of the destination straightedned spinal cord -> done -- functionning
# TODO: for the orthogonal resampling, reduce the XY size of the destination image. -> not functionning 2013-12-19
# TODO: at the end, apply the deformation field on the UN-PADDED image (i.e., the source) -> done -- functionning
# TODO: use .nii throughout the whole code to be user-environment-variable-independent : fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI -> later
# TODO: things to test with batch
# - landmarks with different values -> test on /scripts -> done -- functionning
# - 3d smooth of the input image (e.g., 3mm gaussian kernel) -> later

# TODO: crop the warping field to match the size of the input volume -> not functionning 2013-12-19


# NOTE FOR THE DEVELOPER:
# ---------------------------------------------------------------------------------------
# for the module that creates the landmarks, if you want to generate landmarks with different/or similar values, go to the line of code that says: "landmarks_orthogonal"...
# there is 5 lines of code to change!!!!!



## Create a structure to pass important user parameters to the main function
class parameters:
    ## The constructor
    def __init__(self):
        self.schedule_file = 'schedule_TxTy_8mm_4mm.sch'
        ## @var schedule_file
        self.debug = 1
        ## @var debug
        self.order_interp_ZX = 1
        ## @var spline interpolation order ZX plane (with gap != 1)
        self.order_interp_ZY = 1
        ## @var spline interpolation order ZY plane (with gap != 1)
        self.order_interp_ZX_ZY = 1
        ## @var spline interpolation order ZX_ZY plane (with gap = 1)
        self.step = 25
        ## @var gap between landmarks
        self.landmarks_cross_size = 5
        ## @var distance between landmarks within the cross structure



# check if needed Python libraries are already installed or not
print 'Verify if needed Python libraries are already installed ...'

try:
    from nibabel import load, save, Nifti1Image
    print '--- nibabel already installed ---'
except ImportError:
    print '--- nibabel not already installed ---'
    exit(2)

try:
    from numpy import array, loadtxt, cross
    print '--- numpy already installed ---'
except ImportError:
    print '--- numpy not already installed ---'
    exit(2)

try:
    from scipy.integrate import quad
    from scipy import interpolate
    print '--- scipy already installed ---'
except ImportError:
    print '--- scipy not already installed ---'
    exit(2)

try:
    from sympy import Symbol, nsolve
    print '--- sympy already installed ---'
except ImportError:
    print '--- sympy not already installed ---'
    exit(2)

from fnmatch import filter
from csv import reader
from math import sqrt
from os import path, walk
from getopt import getopt, GetoptError
from commands import getstatusoutput
from sys import exit, argv
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import numpy as np
import operator


## Extracts path, file and extension
def extract_fname(fname):

    # extract path
    path_fname = path.dirname(fname)+'/'
    # check if only single file was entered (without path)
    if path_fname == '/':
        path_fname = ''
    # extract file and extension
    file_fname = fname
    file_fname = file_fname.replace(path_fname,'')
    file_fname, ext_fname = path.splitext(file_fname)
    # check if .nii.gz file
    if ext_fname == '.gz':
        file_fname = file_fname[0:len(file_fname)-4]
        ext_fname = ".nii.gz"

    return path_fname, file_fname, ext_fname


## Check existence of a file
def exist_image(fname):

    if path.isfile(fname) or path.isfile(fname + '.nii') or path.isfile(fname + '.nii.gz'):
        pass
    else:
        print('\nERROR: ' + fname + ' does not exist. Exit program.\n')
        exit(2)


## Find the coordinate in the sagittal plane of the binary point needed for the centerline initilization
def find_initial_mask_z_coordinate(fname):

    # read file
    initial_file_name = fname
    file = load(initial_file_name)

    # get the image data
    data = file.get_data()
    X, Y, Z = (data > 0).nonzero()
    Z = Z[0]
    reference = Z

    return reference


## Find the dimension of the binary image needed for the centerline initilization in the coronal plane
def find_initial_mask_lenght(fname):

    #read binary file
    initial_file_name = fname
    file = load(initial_file_name)
    dimX, dimY, dimZ = file.get_header().get_data_shape()
    return dimX


## Find the dimension of the binary image needed for the centerline initilization in the sagital plane
def find_initial_mask_width(fname):

    #read binary file
    initial_file_name = fname
    file = load(initial_file_name)
    dimX, dimY, dimZ = file.get_header().get_data_shape()
    return dimY


## Find the centerline coordinates points used for the centerline initialization
def find_centerline_coordinates(volume, variable, reference, distance):

    Z = 0

    if variable == reference:

        # create txt centerline files only one time (up move)
        if distance > 0:

            fileID = open('tmp.centerX.txt','w')
            fileID.close()
            fileID = open('tmp.centerY.txt','w')
            fileID.close()
            fileID = open('tmp.centerZ.txt','w')
            fileID.close()
            fileID = open('tmp.center.txt','w')
            fileID.close()

        reference = int(reference)
        Z=int(variable)

        # read centerline file
        initial_mask_name = volume + '_splitZ' + str(reference).zfill(4) + '-mask.nii.gz'
        print "mask name : " + initial_mask_name
        centerline = load(initial_mask_name)
        #dimX, dimY = centerline.get_header().get_data_shape()
        #print "dimX : " + str(dimX)
        #print "dimY : " + str(dimY)
        #Z = variable
        #print "Z : " + str(Z)

        # get the image data
        data = centerline.get_data()

        X, Y = (data > 0).nonzero()
        X = X[0]
        Y = Y[0]
        print 'reference point coordinates: ' + str(X) + ' ; ' + str(Y) + ' ; ' + str(Z)

        centerX = X
        centerY = Y
        centerZ = Z

        # write reference centerline point only one time (down move)
        if distance < 0:

            fileID = open('tmp.centerX.txt','a')
            fileID.write("%f \n" %centerX)
            fileID.close()
            fileID = open('tmp.centerY.txt','a')
            fileID.write("%f \n" %centerY)
            fileID.close()
            fileID = open('tmp.centerZ.txt','a')
            fileID.write("%f \n" %centerZ)
            fileID.close()
            fileID = open('tmp.center.txt','a')
            fileID.write('%f\t%f\t%f\n' %(centerX, centerY, centerZ))
            fileID.close()

        Z = Z + distance

        # import .mat transformation matrix
        omat_inv = loadtxt('tmp.omat_cumul_inv.mat')

        X = X + omat_inv[0][3]
        Y = Y + omat_inv[1][3]

        print 'centerline coordinates: ' + str(X) + ' ; ' + str(Y) + ' ; ' + str(Z)

        centerX = X
        centerY = Y
        centerZ = Z

        fileID = open('tmp.centerX.txt','a')
        fileID.write("%f \n" %centerX)
        fileID.close()
        fileID = open('tmp.centerY.txt','a')
        fileID.write("%f \n" %centerY)
        fileID.close()
        fileID = open('tmp.centerZ.txt','a')
        fileID.write("%f \n" %centerZ)
        fileID.close()
        fileID = open('tmp.center.txt','a')
        fileID.write('%f\t%f\t%f\n' %(centerX, centerY, centerZ))
        fileID.close()

    else:

        Z1 = int(variable) + distance

        reference = int(reference)

        # read centerline file
        initial_mask_name = volume + '_splitZ' + str(reference).zfill(4) + '-mask.nii.gz'

        print "mask name : " + initial_mask_name
        centerline = load(initial_mask_name)
        #dimX, dimY = centerline.get_header().get_data_shape()
        #print "dimX : " + str(dimX)
        #print "dimY : " + str(dimY)
        #Z = variable
        #print "dimZ : " + str(Z)

        # get the image data
        data = centerline.get_data()
        X, Y = (data > 0).nonzero()
        X = X[0]
        Y = Y[0]
        print 'reference point coordinates: ' + str(X) + ' ; ' + str(Y) + ' ; ' + str(Z)

        # import .mat matrix
        omat_cumul_inv = loadtxt('tmp.omat_cumul_inv.mat')

        X = X + omat_cumul_inv[0][3]
        Y = Y + omat_cumul_inv[1][3]

        print 'centerline coordinates: ' + str(X) + ' ; ' + str(Y) + ' ; ' + str(Z1)

        centerX = X
        centerY = Y
        centerZ = Z1

        fileID = open('tmp.centerX.txt','a')
        fileID.write("%f \n" %centerX)
        fileID.close()
        fileID = open('tmp.centerY.txt','a')
        fileID.write("%f \n" %centerY)
        fileID.close()
        fileID = open('tmp.centerZ.txt','a')
        fileID.write("%f \n" %centerZ)
        fileID.close()
        fileID = open('tmp.center.txt','a')
        fileID.write('%f\t%f\t%f\n' %(centerX, centerY, centerZ))
        fileID.close()


## Create slice by slice transformation matrices from fitted centerline
def apply_fitted_transfo_to_matrices( file_volume, binary, fname, reference, slice, distance ):

    # recover the centerline points coordinates stored in center.txt file
    orderedcenter = 'tmp.center.txt'

    file = open(orderedcenter, 'rb')
    data_ordered_center = reader(file, delimiter='\t')
    table_ordered_center = [row for row in data_ordered_center]

    lenght = len(table_ordered_center)
    for i in range(0,lenght):
        table_ordered_center[i][2] = float(table_ordered_center[i][2])

    # sort the list in z expanded way
    table_ordered_center = sorted(table_ordered_center, key=operator.itemgetter(2))

    # count all the lines not empty in the txt file to determine the size of the M matrix defined below
    lines_counter = 0

    with open(orderedcenter) as f:
        for line in f:
            if line != '\n':
                lines_counter += 1

    lenght = lines_counter
    print "Number of centerline points:"
    print lenght

    X_init = [0 for x in xrange(0, lenght)]
    Y_init = [0 for x in xrange(0, lenght)]
    Z_init = [0 for x in xrange(0, lenght)]

    i = 0
    while (i < lenght):
        X_init[i]=float(table_ordered_center[i][0])
        Y_init[i]=float(table_ordered_center[i][1])
        Z_init[i]=float(table_ordered_center[i][2])
        i = i + 1

    X = array(X_init)
    Y = array(Y_init)
    Z = array(Z_init)

    if distance != 1:

        # centerline fitting using InterpolatedUnivariateSpline
        tck_X = interpolate.splrep(Z,X,s=parameters.order_interp_ZX)
        Xnew = interpolate.splev(Z,tck_X,der=0)

        tck_X_order_2 = interpolate.splrep(Z,X,s=2)
        Xnew_order_2 = interpolate.splev(Z,tck_X_order_2,der=0)

        tck_X_order_10 = interpolate.splrep(Z,X,s=10)
        Xnew_order_10 = interpolate.splev(Z,tck_X_order_10,der=0)

        #plt.figure()
        #plt.plot(Z,X,'.-',label='Linear')
        #plt.plot(Z,Xnew,'r',label='Spline interpolation: order=' + parameters.order_interp_ZX)
        #plt.plot(Z,Xnew_order_2,'g',label='Spline interpolation: order=2')
        #plt.plot(Z,Xnew_order_10,'c',label='Spline interpolation: order=10')
        #plt.legend(loc='upper right')
        #plt.title('Z-X plane polynomial interpolation')
        #plt.show()

        tck_Y = interpolate.splrep(Z,Y,s=parameters.order_interp_ZY)
        Ynew = interpolate.splev(Z,tck_Y,der=0)

        tck_Y_order_2 = interpolate.splrep(Z,Y,s=2)
        Ynew_order_2 = interpolate.splev(Z,tck_Y_order_2,der=0)

        tck_Y_order_10 = interpolate.splrep(Z,Y,s=10)
        Ynew_order_10 = interpolate.splev(Z,tck_Y_order_10,der=0)

        #plt.figure()
        #plt.plot(Z,Y,'.-',label='Linear')
        #plt.plot(Z,Ynew,'r',label='Spline interpolation: order=' + parameters.order_interp_ZY)
        #plt.plot(Z,Ynew_order_2,'g',label='Spline interpolation: order=2')
        #plt.plot(Z,Ynew_order_10,'c',label='Spline interpolation: order=10')
        #plt.legend(loc='upper right')
        #plt.title('Z-Y plane polynomial interpolation')
        #plt.show()

        # calculate the missing centerline point due to the slice gap with the fitted centerline curve equation

        xf = [0 for x in xrange(0, slice+2)]
        yf = [0 for x in xrange(0, slice+2)]
        zf = [0 for x in xrange(0, slice+2)]

        z = 0

        while z < slice+2:
            x_gap = interpolate.splev(z,tck_X,der=0)
            xf[z] = x_gap
            y_gap = interpolate.splev(z,tck_Y,der=0)
            yf[z] = y_gap
            zf[z] = z
            # next iteration
            z = z + 1

        Xf = array(xf)
        Yf = array(yf)
        Zf = array(zf)

        #plt.figure()
        #plt.plot(Zf,Xf,'.-')
        #plt.legend(['spline interpolation: order=' + str(parameters.order_interp_ZX)])
        #plt.title('Z-X plane polynomial interpolation extended to all centerline points')
        #plt.show()

        #plt.figure()
        #plt.plot(Zf,Yf,'.-')
        #plt.legend(['spline interpolation: order=' + str(parameters.order_interp_ZY)])
        #plt.title('Z-Y plane polynomial interpolation extended to all centerline points')
        #plt.show()

        print '******************************************************************************'
        print 'Write txt files for a slice gap: ' + str(distance)

        fileID = open('tmp.centerline_fitted.txt', 'w')
        a = len(xf)
        for i in range(0, a):
            fileID.write('%f\t%f\t%f\n' %(float(xf[i]), float(yf[i]), float(zf[i])))
        fileID.close()
        print 'Write: ' + 'tmp.centerline_fitted.txt'

        if parameters.debug == 1:
            fileID = open('txt_files/centerline_fitted.txt', 'w')
            a = len(xf)
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\n' %(float(xf[i]), float(yf[i]), float(zf[i])))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline_fitted.txt'

            fileID = open('txt_files/centerline_fitted_pse.txt', 'w')
            a = len(xf)
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\t%f\n' %(float(xf[i]), float(yf[i]), float(zf[i]), 1))
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline_fitted_pse.txt'

            fileID = open('txt_files/centerline_fitted_pse_pad50.txt', 'w')
            a = len(xf)
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\t%f\n' %(float(xf[i]), float(yf[i]), float(zf[i])+50, 1))
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline_fitted_pse_pad50.txt'

            print '******************************************************************************'

    else:

        print str(lenght)
        print str(slice+1)

        x = [0 for x in xrange(slice+2)]
        y = [0 for x in xrange(slice+2)]
        z = [0 for x in xrange(slice+2)]

        x = X
        y = Y
        z = Z

        print '******************************************************************************'
        print 'Write txt files for slice gap: ' + str(distance)

        if parameters.debug ==1:
            fileID = open('txt_files/centerline.txt', 'w')
            a = len(x)
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\n' %(float(x[i]), float(y[i]), float(z[i])))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline.txt'

            fileID = open('txt_files/centerline_pse.txt', 'w')
            a = len(x)
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\t%f\n' %(float(x[i]), float(y[i]), float(z[i]), 1))
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline_pse.txt'

            fileID = open('txt_files/centerline_pse_pad50.txt', 'w')
            a = len(x)
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\t%f\n' %(float(x[i]), float(y[i]), float(z[i])+50, 1))
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline_pse_pad50.txt'

            reference=int(reference)
            fileID = open('txt_files/centerline_straightened_pse.txt', 'w')
            a = len(x)
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\t%f\n' %(float(x[reference]), float(y[reference]), float(z[i]), 1))
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline_straightened_pse.txt'

            reference=int(reference)
            fileID = open('txt_files/centerline_straightened_pse_pad50.txt', 'w')
            a = len(x)
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\t%f\n' %(float(x[reference]), float(y[reference]), float(z[i])+50, 1))
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline_straightened_pse_pad50.txt'

        print '******************************************************************************'

        # centerline fitting using InterpolatedUnivariateSpline
        tck_X = interpolate.splrep(Z, X, s=parameters.order_interp_ZX_ZY)
        Xnew = interpolate.splev(Z,tck_X,der=0)

        #plt.figure()
        #plt.plot(Z,X,'.-',Z,Xnew,'r')
        #plt.legend(['Linear','spline interpolation: order=' + str(parameters.order_interp_ZX_ZY)])
        #plt.title('Z-X plane polynomial interpolation')
        #plt.show()

        tck_Y = interpolate.splrep(Z, Y, s=parameters.order_interp_ZX_ZY)
        Ynew = interpolate.splev(Z,tck_Y,der=0)

        #plt.figure()
        #plt.plot(Z,Y,'.-',Z,Ynew,'r')
        #plt.legend(['Linear','spline interpolation: order=' + str(parameters.order_interp_ZX_ZY)])
        #plt.title('Z-Y plane polynomial interpolation')
        #plt.show()

        x_final = [0 for x in xrange(slice+2)]
        y_final = [0 for x in xrange(slice+2)]
        z_final = [0 for x in xrange(slice+2)]

        x_final = Xnew
        y_final = Ynew
        z_final = Z

        print '******************************************************************************'
        print 'Write txt files for slice gap: ' + str(distance)

        fileID = open('tmp.centerline_fitted.txt', 'w')
        a = len(x_final)
        for i in range(0, a):
            fileID.write('%f\t%f\t%f\n' %(float(x_final[i]), float(y_final[i]), float(z_final[i])))
        fileID.close()
        print 'Write: ' + 'tmp.centerline_fitted.txt'

        if parameters.debug == 1:
            fileID = open('txt_files/centerline_fitted.txt', 'w')
            a = len(x_final)
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\n' %(float(x_final[i]), float(y_final[i]), float(z_final[i])))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline_fitted.txt'

            fileID = open('txt_files/centerline_fitted_pse.txt', 'w')
            a = len(x_final)
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\t%f\n' %(float(x_final[i]), float(y_final[i]), float(z_final[i]), 1))
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline_fitted_pse.txt'

            fileID = open('txt_files/centerline_fitted_pse_pad50.txt', 'w')
            a = len(x_final)
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            for i in range(0, a):
                fileID.write('%f\t%f\t%f\t%f\n' %(float(x_final[i]), float(y_final[i]), float(z_final[i])+50, 1))
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            print 'Write: ' + 'txt_files/centerline_fitted_pse_pad50.txt'

        print '******************************************************************************'

    fileID = 'tmp.centerline_fitted.txt'
    # read file
    fid = open(fileID,'rb')
    data = reader(fid, delimiter='\t')
    table = [row for row in data]
    fid.close()
    table_numpy = array(table)
    x_fitted = table_numpy[:, 0]
    y_fitted = table_numpy[:, 1]
    z_fitted = table_numpy[:, 2]

    # use nibabel to read the binary volume
    centerline = load(binary + '.nii.gz')
    img = centerline.get_data()
    shape = img.shape
    print "Input volume size:"
    print shape

    # copy header of input volume
    hdr_binary = centerline.get_header()
    hdr_binary_copy = hdr_binary.copy()

    reference = int(reference)

    for i in range(0,a):
        if i != reference:
            img[int(float(x_fitted[i]))][int(float(y_fitted[i]))][int(float(z_fitted[i]))]=1

    fcenterline = fname + '_APRLIS_centerline_fitted.nii.gz'
    # save the new fitted centerline volume
    data_numpy = array(img)
    img = Nifti1Image(data_numpy, None, hdr_binary_copy)
    save(img, fcenterline)

    centerline = load(binary + '.nii.gz')
    img = centerline.get_data()

    for i in range(0,a):
        if i != reference:
            img[int(float(x_fitted[reference]))][int(float(y_fitted[reference]))][int(float(z_fitted[i]))]=1

    fcenterline_straightened = fname + '_APRLIS_centerline_straightened.nii.gz'
    # save the new straightened centerline volume
    data_numpy = array(img)
    img = Nifti1Image(data_numpy, None, hdr_binary_copy)
    save(img, fcenterline_straightened)

    # create all slice by slice cumulative transformation matrices
    a = len(x_fitted)

    for ref in range(0, a):
        x_tansform = float(x_fitted[ref]) - float(x_fitted[0])
        y_tansform = float(y_fitted[ref]) - float(y_fitted[0])

        initial_mat_name = file_volume + '_splitZ' + str(ref).zfill(4) + '-omat_cumul.txt'
        print '>> ' + initial_mat_name + ' created'

        fid = open(initial_mat_name,'w')
        fid.write('%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f' %(1, 0, 0, -x_tansform, 0, 1, 0, -y_tansform, 0, 0, 1, 0, 0, 0, 0, 1))
        fid.close()


## Realize a slice by slice gaussian weighted registration over a machine to straighten the spinal cord using fitted centerline and corresponding transformation matrices previously calculated
def apply_fitted_transfo_to_image(volume, reference, volume_fitted_straightened, distance):

    VOLUME=volume
    REFERENCE=reference
    OUTPUT=volume_fitted_straightened
    DISTANCE=distance

    print '*************************************************************************************************************************'
    print '* Image to straighten: ' + str(VOLUME)
    print '* Reference slice: ' + str(REFERENCE)
    print '* Spinal cord straightened using fitted centerline (spline interpolation): ' + str(OUTPUT)
    print '* Gap between slices used for registration. [Default=1]: ' + str(DISTANCE)
    print '*************************************************************************************************************************'

    FILE_VOLUME = VOLUME

    # extract slices from nii volumes
    cmd = 'fslsplit ' + str(FILE_VOLUME) + ' ' + str(FILE_VOLUME) + '_splitZ -z'
    print('>> '+ cmd)
    status, output = getstatusoutput(cmd)

    #height of the entire input volume
    FILE_VOLUME_SLICE = load(FILE_VOLUME + '.nii.gz')
    FILE_VOLUME_DATA = FILE_VOLUME_SLICE.get_data()
    FILE_VOLUME_SHAPE = FILE_VOLUME_DATA.shape
    HEIGHT = FILE_VOLUME_SHAPE[2] - 2

    SLICE = HEIGHT
    #print 'Slice number of interest: ' + str(SLICE+1)

    REFERENCE_MOINS = REFERENCE - 1

    FILE_REF_FIT = FILE_VOLUME + '_splitZ' + str(REFERENCE).zfill(4)

    cmd = 'convert_xfm -omat ' + FILE_REF_FIT + '-omat_cumul_inv.txt -inverse ' + FILE_REF_FIT + '-omat_cumul.txt'
    print('>> '+ cmd)
    status, PWD = getstatusoutput(cmd)

    VARIABLE=0

    if VARIABLE <= SLICE:

        while VARIABLE <= SLICE:

            # iterative variable
            VARIABLE_PLUS = VARIABLE + 1

            # slice variables
            FILE_DEST = FILE_VOLUME + '_splitZ' + str(VARIABLE).zfill(4)
            FILE_SRC = FILE_VOLUME + '_splitZ' + str(VARIABLE_PLUS).zfill(4)

            #print 'slice by slice registration'

            if VARIABLE == 0:
                cmd = 'flirt -in ' + FILE_DEST + ' -ref ' + FILE_DEST + ' -applyxfm -init ' + FILE_REF_FIT + '-omat_cumul_inv.txt -out ' + FILE_DEST
                print('>> '+ cmd)
                status, PWD = getstatusoutput(cmd)
                cmd = 'flirt -in ' + FILE_SRC + ' -ref ' + FILE_SRC + ' -applyxfm -init ' + FILE_REF_FIT + '-omat_cumul_inv.txt -out ' + FILE_SRC
                print('>> '+ cmd)
                status, PWD = getstatusoutput(cmd)
            if VARIABLE == REFERENCE_MOINS:
                cmd = 'cp ' + FILE_SRC + '.nii.gz ' + FILE_SRC + '_reg_mask.nii.gz'
                print('>> '+ cmd)
                status, PWD = getstatusoutput(cmd)
            if VARIABLE != REFERENCE_MOINS and VARIABLE != 0:
                cmd = 'flirt -in ' + FILE_SRC + ' -ref ' + FILE_SRC + ' -applyxfm -init ' + FILE_REF_FIT + '-omat_cumul_inv.txt -out ' + FILE_SRC
                print('>> '+ cmd)
                status, PWD = getstatusoutput(cmd)

            if VARIABLE != REFERENCE_MOINS:
                cmd = 'flirt -in ' + FILE_SRC + ' -ref ' + FILE_DEST + ' -applyxfm -init ' + FILE_SRC + '-omat_cumul.txt -out ' + FILE_SRC + '_reg_mask'
                print('>> '+ cmd)
                status, PWD = getstatusoutput(cmd)

            VARIABLE = VARIABLE + 1


    VARIABLE=0

    #merging of registered spinal cord slices
    while VARIABLE <= SLICE:

        # iterative variable
        VARIABLE_PLUS=VARIABLE + 1

        # input volume slice variables
        FILE_DEST = FILE_VOLUME + '_splitZ' + str(VARIABLE).zfill(4)
        FILE_SRC = FILE_VOLUME + '_splitZ' + str(VARIABLE_PLUS).zfill(4)

        # merge each slice file into a pseudo list of image registered files
        if VARIABLE == 0:
            FILE_MASK_REG_LIST = FILE_DEST
        elif VARIABLE == SLICE:
            FILE_MASK_REG_LIST = FILE_MASK_REG_LIST + ' ' + FILE_DEST + '_reg_mask' + ' ' + FILE_SRC + '_reg_mask'
        else:
            FILE_MASK_REG_LIST = FILE_MASK_REG_LIST + ' ' + FILE_DEST + '_reg_mask'

        VARIABLE = VARIABLE + 1


    # merge the new registered images with -z axis [concatenate]
    cmd = 'fslmerge -z ' + OUTPUT + '.nii.gz ' + FILE_MASK_REG_LIST
    print('>> ' + cmd)
    status, PWD = getstatusoutput(cmd)


## Print usage
def usage():
    print 'USAGE: \n' \
        'Spinal cord straightening:\n' \
        '    sct_straighten_spinalcord.py -i <data> -p <binary>\n' \
        '\n'\
        '  -h       help. Show this message.\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i       anatomic nifti file. Image to straighten.\n' \
        '  -p       binary nifti file. Image used as initialization for the straightening process.\n' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -o       nifti file. Spinal cord straightened using slice-by-slice gaussian weighted registration.\n' \
        '  -m       nifti file. Gaussian mask centered along the spinal cord.\n' \
        '  -g       integer. Gap between slices used for registration. [Default=1].\n' \
        '  -d       deformation field. Determine a transformation (Warping field + Affine transform) to straighten the spinal cord based on an orthogonal centerline plane resampling approach. [Default=0].\n' \
        '\n'\
        'EXAMPLES:\n' \
        '\n'\
        'Straighten the spine using only a slice-by-slice gaussian-weighted registration. For example:\n' \
        '     sct_straighten_spinalcord.py -i t2.nii.gz -p binary_t2.nii.gz\n' \
        '\n'\
        'Straighten the spine using a slice-by-slice gaussian-weighted registration and a fitted centerline (spline interpolation). For example:\n' \
        '     sct_straighten_spinalcord.py -i t2.nii.gz -p binary_t2.nii.gz -r 1\n' \
        '\n'\
        'Find the warping transformation (warping filed + affine transform) to straighten the spine using an orthogonal centerline planes resampling approach. For example:\n' \
        '     sct_straighten_spinalcord.py -i t2.nii.gz -p binary_t2.nii.gz -d 1\n'
    exit(2)

## Main function
def main():

    # Initialization
    VOLUME = ''
    BINARY = ''
    OUTPUT = ''
    GAUSS = ''
    DISTANCE = ''
    DEFORMATION = ''

    # manage intermediary files into python script directory
    if parameters.debug == 1:
        #create directories
        cmd = 'mkdir output_images'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)
        cmd = 'rm output_images/*'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)

        cmd = 'mkdir txt_files'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)
        cmd = 'rm txt_files/*'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)

        cmd = 'rm PSE*'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)


    # Check input parameters
    try:
        opts, args = getopt(argv[1:],'hi:p:o:m:g:d:')
    except GetoptError as err:
        print str(err)
        usage()
        exit(2)
    if not opts:
        # no option supplied
        print 'no option supplied'
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            VOLUME = arg
        elif opt in ('-p'):
            BINARY = arg
        elif opt in ('-o'):
            OUTPUT = arg
        elif opt in ('-m'):
            GAUSS = arg
        elif opt in ('-g'):
            DISTANCE = arg
        elif opt in ('-d'):
            DEFORMATION = arg


    # display usage if a mandatory argument is not provided
    if VOLUME == '' and BINARY == '':
        #print "no argument provided"
        usage()

    # run the registration if both mandatory argument are provided

    # detect the schedule file location
    # extract path of the script
    print 'Find the FSL schedule file for the FLIRT registration ...'
    path_script = path.dirname(__file__)+'/'

    # extract path of schedule file
    schedule_path = path_script[0:-8]+'src/' + parameters.schedule_file
    print schedule_path

    print '***************************************************************************************************************************************************'
    print '* Image to straighten: ' + VOLUME
    print '* Binary image used as initialization for the straightening: ' + BINARY
    print '* Straightened spinal cord using slice by slice gaussian weighted registration: ' + OUTPUT
    print '* Gaussian mask centered along the spinal cord: ' + GAUSS
    print '* Gap between slices used for registration. [Default=1]: ' + DISTANCE
    print '* Deformation. Determine a warping transformation to straighten the spinal cord" [Default=0]: ' + DEFORMATION
    print '***************************************************************************************************************************************************'

    # copy the input volume into the script directory
    print 'Verify if anatomical input volume exists ...'
    exist_image(VOLUME)
    print 'Anatomical input volume exists.'
    path_func_VOLUME, file_func_VOLUME, ext_func_VOLUME = extract_fname(VOLUME)
    cmd = 'cp ' + VOLUME + ' ' + file_func_VOLUME + '.nii.gz'
    print('>> '+ cmd)
    status, output = getstatusoutput(cmd)
    VOLUME = file_func_VOLUME

    if parameters.debug == 1:
        # remove unecessary files presented in the script directory
        cmd = 'rm ' + VOLUME + '_*'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)

    # copy the binary volume into the script directory
    print 'Verify if input binary volume exists ...'
    exist_image(BINARY)
    print 'Input binary volume exists.'
    path_func_BINARY, file_func_BINARY, ext_func_BINARY = extract_fname(BINARY)
    cmd = 'cp ' + BINARY + ' ' + file_func_BINARY + '.nii.gz'
    print('>> '+ cmd)
    status, output = getstatusoutput(cmd)
    BINARY = file_func_BINARY

    if parameters.debug == 1:
        # remove unecessary files presented in the script directory
        cmd = 'rm ' + BINARY + '_*'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)

    # recover the local name of future output images without absolute paths
    if OUTPUT != '':
        path_func_OUTPUT, file_func_OUTPUT, ext_func_OUTPUT = extract_fname(OUTPUT)
        OUTPUT = file_func_OUTPUT

    if GAUSS != '':
        path_func_GAUSS, file_func_GAUSS, ext_func_GAUSS = extract_fname(GAUSS)
        GAUSS = file_func_GAUSS

    # reorient binary image into AP RL IS orientation
    cmd = 'fslswapdim ' + BINARY + ' AP RL IS tmp.' + BINARY + '_APRLIS'
    print('>> '+ cmd)
    status, output = getstatusoutput(cmd)
    FILE_BINARY = 'tmp.' + BINARY + "_APRLIS"

    # reorient input anatomical volume into AP RL IS orientation
    cmd = 'fslswapdim ' + VOLUME + ' AP RL IS tmp.' + VOLUME + '_APRLIS'
    print('>> '+ cmd)
    status, output = getstatusoutput(cmd)
    FILE_VOLUME = 'tmp.' + VOLUME + '_APRLIS'

    FILE_VOLUME_OUTPUT = VOLUME

    # extract slices from nii volume
    cmd = 'fslsplit ' + FILE_VOLUME + ' ' + FILE_VOLUME + '_splitZ' + " -z"
    print('>> '+ cmd)
    status, output = getstatusoutput(cmd)

    #height of the entire input volume
    FILE_VOLUME_SLICE = load(FILE_VOLUME + '.nii.gz')
    FILE_VOLUME_DATA = FILE_VOLUME_SLICE.get_data()
    FILE_VOLUME_SHAPE = FILE_VOLUME_DATA.shape
    HEIGHT = FILE_VOLUME_SHAPE[2] - 2

    # get the user binary point height and the lenght and width of the input binary volume
    REFERENCE = find_initial_mask_z_coordinate(FILE_BINARY + '.nii.gz')
    print 'Binary point: height=' +  str(REFERENCE)
    LENGTH = find_initial_mask_lenght(FILE_BINARY + '.nii.gz')
    print 'Binary volume slice dim.: lenght= ' + str(LENGTH)
    WIDTH = find_initial_mask_width(FILE_BINARY + '.nii.gz')
    print 'Binary volume slice dim.: width= ' + str(WIDTH)

    print 'Input volume slices number: ' + str(HEIGHT)

    SLICE=HEIGHT

    if DISTANCE == '':
        print 'Gap between input slices not defined. [Default gap=1]'
        # set default gap
        DISTANCE=1
    else:
        print 'Gap between input slices defined: gap=' + str(DISTANCE)
        # set defined gap
        DISTANCE = int(DISTANCE)

    # go up, then down in reference to the binary point
    for iUpDown in range(1, 3):

        print '*************************'
        print 'iUpDown: ' + str(iUpDown)
        print '*************************'

        VARIABLE = REFERENCE

        while VARIABLE <= SLICE and VARIABLE >= DISTANCE:

            # define iterative variables
            if iUpDown == 1:

                # inter gap variable - up move
                VARIABLE_PLUS = VARIABLE + DISTANCE
                VARIABLE_PLUS_SUM = VARIABLE - REFERENCE + 1


            else:

                # inter gap variable - down move
                VARIABLE_PLUS = VARIABLE - DISTANCE
                VARIABLE_PLUS_SUM = REFERENCE - VARIABLE_PLUS


            # define input volume slice variables

            # inter gap transformation
            FILE_DEST = FILE_VOLUME + '_splitZ' + str(VARIABLE).zfill(4)
            FILE_SRC = FILE_VOLUME + '_splitZ' + str(VARIABLE_PLUS).zfill(4)

            # use to create the reference mask
            FILE_MASK = FILE_DEST + '-mask'

            #print 'Slide by slide registration using gaussian mask and iterative mean'


            if VARIABLE == REFERENCE:

                print '***************************'
                print 'z = ' + str(VARIABLE)
                print '***************************'

                # binary point centered in the centerline used as initialization for the straightening
                # get the reference slice from the binary point
                cmd = 'fslroi ' + FILE_BINARY + '.nii.gz ' + FILE_MASK + '.nii.gz' + ' 0 ' + str(LENGTH) + ' 0 ' + str(WIDTH) + ' ' + str(REFERENCE) + ' 1'
                print('>> ' + cmd)
                status, output = getstatusoutput(cmd)

                # FSL gaussian mask creation
                cmd = 'fslmaths ' + FILE_MASK + ' -kernel gauss 6.5 -dilM -s 3 ' + FILE_MASK + '_gaussian'
                print('>> ' + cmd)
                status, output = getstatusoutput(cmd)

                # implement the gap straightening

                VARIABLE_DISTANCE = VARIABLE

                for i in range(0, DISTANCE):

                    if iUpDown == 1:
                        # inner gap variable
                        VARIABLE_DISTANCE_PLUS = VARIABLE_DISTANCE + 1
                    else:
                        VARIABLE_DISTANCE_PLUS = VARIABLE_DISTANCE - 1

                    # inner gap transformation
                    FILE_DEST_DISTANCE = FILE_VOLUME + '_splitZ' + str(VARIABLE_DISTANCE).zfill(4)
                    FILE_SRC_DISTANCE = FILE_VOLUME + '_splitZ' + str(VARIABLE_DISTANCE_PLUS).zfill(4)

                    # use to define slice by slice gaussian mask
                    FILE_MASK_DISTANCE = FILE_DEST_DISTANCE + '-mask'
                    FILE_MASK_PLUS_DISTANCE = FILE_SRC_DISTANCE + '-mask'

                    if i == 0:
                        # using two gaussian masks for both src and ref images
                        cmd = 'flirt -in ' + FILE_SRC_DISTANCE + ' -ref ' + FILE_DEST_DISTANCE + ' -schedule ' + schedule_path + ' -verbose 0 -omat tmp.omat_tmp.mat -cost normcorr -forcescaling -inweight ' + FILE_MASK_DISTANCE + '_gaussian' + ' -refweight ' + FILE_MASK_DISTANCE + '_gaussian -out ' + FILE_SRC_DISTANCE + '_reg_mask -paddingsize 3'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        # if registration fails
                        omat_inv = loadtxt('tmp.omat_tmp.mat')
                        if (abs(omat_inv[0][3]) > DISTANCE or abs(omat_inv[1][3]) > DISTANCE):
                            print 'Matrice de transformation incorrecte'
                            cmd = 'flirt -in ' + FILE_SRC_DISTANCE + ' -ref ' + FILE_DEST_DISTANCE + ' -schedule ' + schedule_path + ' -verbose 0 -omat tmp.omat_tmp.mat -out ' + FILE_SRC_DISTANCE + '_reg_mask -paddingsize 3'
                            print('>> ' + cmd)
                            status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        cmd = 'cp tmp.omat_tmp.mat tmp.omat_cumul_tmp.mat'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        # apply the inverse transformation matrix to the first gaussian mask
                        cmd = 'convert_xfm -omat tmp.omat_cumul_inv_tmp.mat -inverse tmp.omat_cumul_tmp.mat'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_cumul_inv_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        cmd = 'flirt -in ' + FILE_VOLUME + '_splitZ' + str(REFERENCE).zfill(4) + '-mask_gaussian -ref ' + FILE_SRC_DISTANCE + ' -applyxfm -init tmp.omat_cumul_inv_tmp.mat -out ' + FILE_MASK_PLUS_DISTANCE + '_gaussian -paddingsize 3'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)

                        if iUpDown == 1:
                            VARIABLE_DISTANCE = VARIABLE_DISTANCE + 1
                        else:
                            VARIABLE_DISTANCE = VARIABLE_DISTANCE - 1

                    else:
                        cmd = 'flirt -in ' + FILE_SRC_DISTANCE + ' -ref ' + FILE_DEST_DISTANCE + ' -schedule ' + schedule_path + ' -verbose 0 -omat tmp.omat_tmp.mat -cost normcorr -forcescaling -inweight ' + FILE_MASK_DISTANCE + '_gaussian' + ' -refweight ' + FILE_MASK_DISTANCE + '_gaussian -out ' + FILE_SRC_DISTANCE + '_reg_mask -paddingsize 3'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        # if registration fails
                        omat_inv = loadtxt('tmp.omat_tmp.mat')
                        if (abs(omat_inv[0][3]) > DISTANCE or abs(omat_inv[1][3]) > DISTANCE):
                            print 'Matrice de transformation incorrecte'
                            cmd = 'flirt -in ' + FILE_SRC_DISTANCE + ' -ref ' + FILE_DEST_DISTANCE + ' -schedule ' + schedule_path + ' -verbose 0 -omat tmp.omat_tmp.mat -out ' + FILE_SRC_DISTANCE + '_reg_mask -paddingsize 3'
                            print('>> ' + cmd)
                            status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        cmd = 'convert_xfm -omat tmp.omat_cumul_tmp.mat -concat tmp.omat_cumul_tmp.mat tmp.omat_tmp.mat'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_cumul_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        cmd = 'convert_xfm -omat tmp.omat_cumul_inv_tmp.mat -inverse tmp.omat_cumul_tmp.mat'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_cumul_inv_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        cmd = 'flirt -in ' + FILE_VOLUME + '_splitZ' + str(REFERENCE).zfill(4) + '-mask_gaussian -ref ' + FILE_SRC_DISTANCE + ' -applyxfm -init tmp.omat_cumul_inv_tmp.mat -out ' + FILE_MASK_PLUS_DISTANCE + '_gaussian -paddingsize 3'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)

                        if iUpDown == 1:
                            VARIABLE_DISTANCE = VARIABLE_DISTANCE + 1
                        else:
                            VARIABLE_DISTANCE = VARIABLE_DISTANCE - 1


                cmd = 'cp tmp.omat_cumul_inv_tmp.mat tmp.omat_cumul_inv.mat'
                print('>> ' + cmd)
                status, output = getstatusoutput(cmd)

                cmd = 'cp tmp.omat_cumul_tmp.mat tmp.omat_cumul.mat'
                print('>> ' + cmd)
                status, output = getstatusoutput(cmd)

                # apply the cumulative transformation over the slice gap
                cmd = 'flirt -in ' + FILE_SRC + ' -ref ' + FILE_DEST + ' -applyxfm -init tmp.omat_cumul.mat -out ' + FILE_SRC + '_reg_mask'
                print('>> ' + cmd)
                status, output = getstatusoutput(cmd)

                # store the centerline points coordinates
                if iUpDown == 1:
                    # find and store in txt files all the centerline points coordinates above reference slice
                    print 'find_centerline_coordinates(' + str(VARIABLE) + ', ' + str(REFERENCE) + ', ' + str(DISTANCE) + ')'
                    find_centerline_coordinates(FILE_VOLUME, VARIABLE, REFERENCE, DISTANCE)
                else:
                    # find and store in txt files centerline points coordinates below reference slice
                    print 'find_centerline_coordinates(' + str(VARIABLE) + ', ' + str(REFERENCE) + ', -' + str(DISTANCE) + ')'
                    find_centerline_coordinates(FILE_VOLUME, VARIABLE, REFERENCE, -DISTANCE)

                # verify the cumulative transformation matrix
                #cmd = 'cat tmp.omat_cumul.mat'
                #print('>> ' + cmd)
                #status, output = getstatusoutput(cmd)
                #print output

                #cmd = 'cat tmp.omat_cumul_inv.mat'
                #print('>> ' + cmd)
                #status, output = getstatusoutput(cmd)
                #print output


            elif VARIABLE != REFERENCE:

                # i try to implement the gap straightening

                VARIABLE_DISTANCE = VARIABLE

                for i in range(0, DISTANCE):

                    # inner gap variable
                    if iUpDown == 1:
                        VARIABLE_DISTANCE_PLUS = VARIABLE_DISTANCE + 1
                    else:
                        VARIABLE_DISTANCE_PLUS = VARIABLE_DISTANCE - 1

                    # inner gap transformation
                    FILE_DEST_DISTANCE = FILE_VOLUME + '_splitZ' + str(VARIABLE_DISTANCE).zfill(4)
                    FILE_SRC_DISTANCE = FILE_VOLUME + '_splitZ' + str(VARIABLE_DISTANCE_PLUS).zfill(4)

                    # use to define slice by slice gaussian mask
                    FILE_MASK_DISTANCE = FILE_DEST_DISTANCE + '-mask'
                    FILE_MASK_PLUS_DISTANCE = FILE_SRC_DISTANCE + '-mask'

                    if i == 0:
                        # do not use iterative mean for t2 image
                        cmd = 'flirt -in ' + FILE_SRC_DISTANCE + ' -ref ' + FILE_DEST_DISTANCE + ' -schedule ' + schedule_path + ' -verbose 0 -omat tmp.omat_tmp.mat -cost normcorr -forcescaling -inweight ' + FILE_MASK_DISTANCE + '_gaussian' + ' -refweight ' + FILE_MASK_DISTANCE + '_gaussian -out ' + FILE_SRC_DISTANCE + '_reg_mask -paddingsize 3'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        # if registration fails
                        omat_inv = loadtxt('tmp.omat_tmp.mat')
                        if (abs(omat_inv[0][3]) > DISTANCE or abs(omat_inv[1][3]) > DISTANCE):
                            print 'Matrice de transformation incorrecte'
                            cmd = 'flirt -in ' + FILE_SRC_DISTANCE + ' -ref ' + FILE_DEST_DISTANCE + ' -schedule ' + schedule_path + ' -verbose 0 -omat tmp.omat_tmp.mat -out ' + FILE_SRC_DISTANCE + '_reg_mask -paddingsize 3'
                            print('>> ' + cmd)
                            status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        cmd = 'convert_xfm -omat tmp.omat_cumul_tmp.mat -concat tmp.omat_cumul_tmp.mat tmp.omat_tmp.mat'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_cumul_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        # apply the inverse transformation matrix to the first gaussian mask
                        cmd = 'convert_xfm -omat tmp.omat_cumul_inv_tmp.mat -inverse tmp.omat_cumul_tmp.mat'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_cumul_inv_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        cmd = 'flirt -in ' + FILE_VOLUME + '_splitZ' + str(REFERENCE).zfill(4) + '-mask_gaussian -ref ' + FILE_SRC_DISTANCE + ' -applyxfm -init tmp.omat_cumul_inv_tmp.mat -out ' + FILE_MASK_PLUS_DISTANCE + '_gaussian -paddingsize 3'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)

                        if iUpDown == 1:
                            VARIABLE_DISTANCE = VARIABLE_DISTANCE + 1
                        else:
                            VARIABLE_DISTANCE = VARIABLE_DISTANCE - 1

                    else:
                        cmd = 'flirt -in ' + FILE_SRC_DISTANCE + ' -ref ' + FILE_DEST_DISTANCE + ' -schedule ' + schedule_path + ' -verbose 0 -omat tmp.omat_tmp.mat -cost normcorr -forcescaling -inweight ' + FILE_MASK_DISTANCE + '_gaussian' + ' -refweight ' + FILE_MASK_DISTANCE + '_gaussian -out ' + FILE_SRC_DISTANCE + '_reg_mask -paddingsize 3'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        # if registration fails
                        omat_inv = loadtxt('tmp.omat_tmp.mat')
                        if (abs(omat_inv[0][3]) > DISTANCE or abs(omat_inv[1][3]) > DISTANCE):
                            print 'Matrice de transformation incorrecte'
                            cmd = 'flirt -in ' + FILE_SRC_DISTANCE + ' -ref ' + FILE_DEST_DISTANCE + ' -schedule ' + schedule_path + ' -verbose 0 -omat tmp.omat_tmp.mat -out ' + FILE_SRC_DISTANCE + '_reg_mask -paddingsize 3'
                            print('>> ' + cmd)
                            status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        cmd = 'convert_xfm -omat tmp.omat_cumul_tmp.mat -concat tmp.omat_cumul_tmp.mat tmp.omat_tmp.mat'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_cumul_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        cmd = 'convert_xfm -omat tmp.omat_cumul_inv_tmp.mat -inverse tmp.omat_cumul_tmp.mat'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)
                        #cmd = 'cat tmp.omat_cumul_inv_tmp.mat'
                        #print('>> ' + cmd)
                        #status, output = getstatusoutput(cmd)
                        #print output
                        cmd = 'flirt -in ' + FILE_VOLUME + '_splitZ' + str(REFERENCE).zfill(4) + '-mask_gaussian -ref ' + FILE_SRC_DISTANCE + ' -applyxfm -init tmp.omat_cumul_inv_tmp.mat -out ' + FILE_MASK_PLUS_DISTANCE + '_gaussian -paddingsize 3'
                        print('>> ' + cmd)
                        status, output = getstatusoutput(cmd)

                        if iUpDown == 1:
                            VARIABLE_DISTANCE = VARIABLE_DISTANCE + 1
                        else:
                            VARIABLE_DISTANCE = VARIABLE_DISTANCE - 1


                cmd = 'cp tmp.omat_cumul_inv_tmp.mat tmp.omat_cumul_inv.mat'
                print('>> ' + cmd)
                status, output = getstatusoutput(cmd)

                cmd = 'cp tmp.omat_cumul_tmp.mat tmp.omat_cumul.mat'
                print('>> ' + cmd)
                status, output = getstatusoutput(cmd)

                # apply the cumulative transformation over the slice gap
                cmd = 'flirt -in ' + FILE_SRC + ' -ref ' + FILE_DEST + ' -applyxfm -init tmp.omat_cumul.mat -out ' + FILE_SRC + '_reg_mask'
                print('>> ' + cmd)
                status, output = getstatusoutput(cmd)

                # store the centerline points coordinates
                if iUpDown == 1:
                    # find and store in txt files all the centerline points coordinates above reference slice
                    print 'find_centerline_coordinates(' + str(VARIABLE) + ', ' + str(REFERENCE) + ', ' + str(DISTANCE) + ')'
                    find_centerline_coordinates(FILE_VOLUME, VARIABLE, REFERENCE, DISTANCE)
                else:
                    # find and store in txt files centerline points coordinates below reference slice
                    print 'find_centerline_coordinates(' + str(VARIABLE) + ', ' + str(REFERENCE) + ', -' + str(DISTANCE) + ')'
                    find_centerline_coordinates(FILE_VOLUME, VARIABLE, REFERENCE, -DISTANCE)

                # verify the cumulative transformation matrix
                #cmd = 'cat tmp.omat_cumul.mat'
                #print('>> ' + cmd)
                #status, output = getstatusoutput(cmd)
                #print output

                #cmd = 'cat tmp.omat_cumul_inv.mat'
                #print('>> ' + cmd)
                #status, output = getstatusoutput(cmd)
                #print output


            # increase the loop variable
            if iUpDown == 1:
                VARIABLE = VARIABLE + DISTANCE
            else:
                VARIABLE = VARIABLE - DISTANCE

            print '***************************'
            print 'z = ' + str(VARIABLE)
            print '***************************'





    # merge straightened and gaussian mask spinal cord slices

    VARIABLE=0

    while VARIABLE <= SLICE:

        # iterative variables
        VARIABLE_PLUS=VARIABLE + DISTANCE

        # input volume slice variables
        FILE_DEST = FILE_VOLUME + '_splitZ' + str(VARIABLE).zfill(4)
        FILE_SRC = FILE_VOLUME + '_splitZ' + str(VARIABLE_PLUS).zfill(4)

        FILE_MASK=FILE_DEST + '-mask'
        FILE_MASK_PLUS=FILE_SRC + '-mask'

        if VARIABLE == 0:
            FILE_MASK_LIST=FILE_MASK + '_gaussian.nii.gz'
            FILE_MASK_REG_LIST=FILE_DEST + '_reg_mask.nii.gz'
        elif VARIABLE == REFERENCE:
            FILE_MASK_REG_LIST=FILE_MASK_REG_LIST + ' ' + FILE_DEST + '.nii.gz'
            FILE_MASK_LIST=FILE_MASK_LIST + ' ' + FILE_MASK + '_gaussian.nii.gz'
        elif VARIABLE == SLICE:
            FILE_MASK_REG_LIST=FILE_MASK_REG_LIST + ' ' + FILE_DEST + '_reg_mask.nii.gz ' + FILE_SRC + '_reg_mask.nii.gz'
            FILE_MASK_LIST=FILE_MASK_LIST + ' ' + FILE_MASK + '_gaussian.nii.gz ' + FILE_MASK_PLUS + '_gaussian.nii.gz'
        else:
            FILE_MASK_LIST=FILE_MASK_LIST + ' ' + FILE_MASK + '_gaussian.nii.gz'
            FILE_MASK_REG_LIST=FILE_MASK_REG_LIST + ' ' + FILE_DEST + '_reg_mask.nii.gz'

        VARIABLE=VARIABLE + DISTANCE

    if OUTPUT != '':
        # merge the new straightened images
        cmd = 'fslmerge -z ' + OUTPUT + '.nii.gz ' + FILE_MASK_REG_LIST
        print('>> ' + cmd)
        status, output = getstatusoutput(cmd)

    if OUTPUT == '':
        # merge the new straightened images
        cmd = 'fslmerge -z ' + FILE_VOLUME + '_straightened.nii.gz ' + FILE_MASK_REG_LIST
        print('>> ' + cmd)
        status, output = getstatusoutput(cmd)

    if GAUSS != '':
        # merge the new mask images
        cmd = 'fslmerge -z ' + GAUSS + '.nii.gz ' + FILE_MASK_LIST
        print('>> ' + cmd)
        status, output = getstatusoutput(cmd)

    if GAUSS == '':
        # merge the new mask images
        cmd = 'fslmerge -z ' + FILE_VOLUME + '_gaussian_mask.nii.gz ' + FILE_MASK_LIST
        print('>> ' + cmd)
        status, output = getstatusoutput(cmd)


    # get the original orientation of the input image

    cmd = 'sct_orientation -i ' + VOLUME + ".nii.gz -get"
    print('>> ' + cmd)
    status, output = getstatusoutput(cmd)
    index = output.find(':')
    orientation = output[index+2:len(output)]
    final_orientation = [0 for x in xrange(6)]
    for j in range(0, 3):
        if j == 0:
            if orientation[j] == 'A':
                final_orientation[j] = 'A'
                final_orientation[j+1] = 'P'
            elif orientation[j] == 'P':
                final_orientation[j] = 'P'
                final_orientation[j+1] = 'A'
            elif orientation[j] == 'R':
                final_orientation[j] = 'R'
                final_orientation[j+1] = 'L'
            elif orientation[j] == 'L':
                final_orientation[j] = 'L'
                final_orientation[j+1] = 'R'
            elif orientation[j] == 'I':
                final_orientation[j] = 'I'
                final_orientation[j+1] = 'S'
            elif orientation[j] == 'S':
                final_orientation[j] = 'S'
                final_orientation[j+1] = 'I'
        if j == 1:
            if orientation[j] == 'A':
                final_orientation[j+1] = 'A'
                final_orientation[j+2] = 'P'
            elif orientation[j] == 'P':
                final_orientation[j+1] = 'P'
                final_orientation[j+2] = 'A'
            elif orientation[j] == 'R':
                final_orientation[j+1] = 'R'
                final_orientation[j+2] = 'L'
            elif orientation[j] == 'L':
                final_orientation[j+1] = 'L'
                final_orientation[j+2] = 'R'
            elif orientation[j] == 'I':
                final_orientation[j+1] = 'I'
                final_orientation[j+2] = 'S'
            elif orientation[j] == 'S':
                final_orientation[j+1] = 'S'
                final_orientation[j+2] = 'I'
        if j == 2:
            if orientation[j] == 'A':
                final_orientation[j+2] = 'A'
                final_orientation[j+3] = 'P'
            elif orientation[j] == 'P':
                final_orientation[j+2] = 'P'
                final_orientation[j+3] = 'A'
            elif orientation[j] == 'R':
                final_orientation[j+2] = 'R'
                final_orientation[j+3] = 'L'
            elif orientation[j] == 'L':
                final_orientation[j+2] = 'L'
                final_orientation[j+3] = 'R'
            elif orientation[j] == 'I':
                final_orientation[j+2] = 'I'
                final_orientation[j+3] = 'S'
            elif orientation[j] == 'S':
                final_orientation[j+2] = 'S'
                final_orientation[j+3] = 'I'

    #apply the original orientation of the input volume to the gaussian mask and the spinal cord straightened using only slice by slice gaussian weighted registration
    if GAUSS != '':
        cmd = 'fslswapdim ' + GAUSS + '.nii.gz ' + str(final_orientation[0]) + str(final_orientation[1]) + ' ' + str(final_orientation[2]) + str(final_orientation[3]) + ' ' + str(final_orientation[4]) + str(final_orientation[5]) + ' ' + GAUSS + '.nii.gz'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)

    if GAUSS == '':
        cmd = 'fslswapdim ' + FILE_VOLUME + '_gaussian_mask.nii.gz ' + str(final_orientation[0]) + str(final_orientation[1]) + ' ' + str(final_orientation[2]) + str(final_orientation[3]) + ' ' + str(final_orientation[4]) + str(final_orientation[5]) + ' ' + FILE_VOLUME_OUTPUT + '_gaussian_mask.nii.gz'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)

    if OUTPUT != '':
        cmd = 'fslswapdim ' + OUTPUT + '.nii.gz ' + str(final_orientation[0]) + str(final_orientation[1]) + ' ' + str(final_orientation[2]) + str(final_orientation[3]) + ' ' + str(final_orientation[4]) + str(final_orientation[5]) + ' ' + OUTPUT + '.nii.gz'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)

    if OUTPUT == '':
        cmd = 'fslswapdim ' + FILE_VOLUME + '_straightened.nii.gz ' + str(final_orientation[0]) + str(final_orientation[1]) + ' ' + str(final_orientation[2]) + str(final_orientation[3]) + ' ' + str(final_orientation[4]) + str(final_orientation[5]) + ' ' + FILE_VOLUME_OUTPUT + '_straightened.nii.gz'
        print('>> '+ cmd)
        status, output = getstatusoutput(cmd)


    if parameters.debug ==1:

        if OUTPUT != '':
            cmd = 'cp ' + OUTPUT + '.nii.gz ' + 'output_images/' + OUTPUT + '.nii.gz'
            print('>> ' + cmd)
            status, output = getstatusoutput(cmd)

        if OUTPUT == '':
            cmd = 'cp ' + FILE_VOLUME_OUTPUT + '_straightened.nii.gz ' + 'output_images/' + FILE_VOLUME_OUTPUT + '_straightened.nii.gz'
            print('>> ' + cmd)
            status, output = getstatusoutput(cmd)

        if GAUSS != '':
            cmd = 'cp ' + GAUSS + '.nii.gz ' + 'output_images/' + GAUSS + '.nii.gz'
            print('>> '+ cmd)
            status, output = getstatusoutput(cmd)

        if GAUSS == '':
            cmd = 'cp ' + FILE_VOLUME_OUTPUT + '_gaussian_mask.nii.gz ' + 'output_images/' + FILE_VOLUME_OUTPUT + '_gaussian_mask.nii.gz'
            print('>> '+ cmd)
            status, output = getstatusoutput(cmd)









    ################################################################################################################
          ##### Centerline fitting (spline interpolation) to increase straightening robustness #####
    ################################################################################################################

    print 'Centerline regularized (spline interpolation) ...'
    # centerline fitting using spline interpolation
    apply_fitted_transfo_to_matrices(FILE_VOLUME, FILE_BINARY, FILE_VOLUME_OUTPUT, REFERENCE, SLICE, DISTANCE)
    # calculate the new spinal cord straightened using the fitted centerline previously calculated
    apply_fitted_transfo_to_image(FILE_VOLUME, REFERENCE, FILE_VOLUME_OUTPUT + '_fitted_straightened', DISTANCE)

    cmd = 'fslswapdim ' + FILE_VOLUME_OUTPUT + '_fitted_straightened.nii.gz ' + str(final_orientation[0]) + str(final_orientation[1]) + ' ' + str(final_orientation[2]) + str(final_orientation[3]) + ' ' + str(final_orientation[4]) + str(final_orientation[5]) + ' ' + FILE_VOLUME_OUTPUT + '_fitted_straightened.nii.gz'
    print('>> '+ cmd)
    status, output = getstatusoutput(cmd)

    if parameters.debug ==1:
        cmd = 'cp ' + FILE_VOLUME_OUTPUT + '_fitted_straightened.nii.gz ' + 'output_images/' + FILE_VOLUME_OUTPUT + '_fitted_straightened.nii.gz'
        print('>> ' + cmd)
        status, output = getstatusoutput(cmd)

        cmd = 'mv ' + FILE_VOLUME_OUTPUT + '_APRLIS_centerline_straightened.nii.gz ' + 'output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_centerline_straightened.nii.gz'
        print('>> ' + cmd)
        status, output = getstatusoutput(cmd)

        cmd = 'mv ' + FILE_VOLUME_OUTPUT + '_APRLIS_centerline_fitted.nii.gz ' + 'output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_centerline_fitted.nii.gz'
        print('>> ' + cmd)
        status, output = getstatusoutput(cmd)












    ################################################################################################################
          ##### FIND THE WARPING FIELD TO STRAIGHTEN THE SPINE CONSIDERING ORTHOGONAL CENTERLINE PLANES #####
    ################################################################################################################


    if DEFORMATION != '':

        if DEFORMATION == '1':

            # read the fitted centerline txt file
            file = open('tmp.centerline_fitted.txt', 'rb')
            data_centerline = reader(file, delimiter='\t')
            table_centerline_fitted = [row for row in data_centerline]

            # count all the lines not empty in the txt file to determine the size of the M matrix defined below
            lines_counter = 0

            with open('tmp.centerline_fitted.txt') as f:
                for line in f:
                    if line != '\n':
                        lines_counter += 1

            lenght = lines_counter - 1
            print "Number of centerline points:"
            print lenght

            X_init = [0 for x in xrange(lenght)]
            Y_init = [0 for x in xrange(lenght)]
            Y = array(Y_init)
            Z_init = [0 for x in xrange(lenght)]
            Z = array(Z_init)

            for i in range(0, lenght):
                X_init[i]=float(table_centerline_fitted[i][0])
                Y_init[i]=float(table_centerline_fitted[i][1])
                Z_init[i]=float(table_centerline_fitted[i][2])

            X = array(X_init)
            Y = array(Y_init)
            Z = array(Z_init)

            # centerline fitting using InterpolatedUnivariateSpline
            tck_X = interpolate.splrep(Z, X, s=0)
            Xnew = interpolate.splev(Z,tck_X,der=0)

            #plt.figure()
            #plt.plot(Z,X,'.-',Z,Xnew,'r')
            #plt.legend(['Linear','InterpolatedUnivariateSpline'])
            #plt.title('Z-X plane interpolation')
            #plt.show()

            tck_Y = interpolate.splrep(Z, Y, s=0)
            Ynew = interpolate.splev(Z,tck_Y,der=0)

            #plt.figure()
            #plt.plot(Z,Y,'.-',Z,Ynew,'r')
            #plt.legend(['Linear','InterpolatedUnivariateSpline'])
            #plt.title('Z-Y plane interpolation')
            #plt.show()

            # the idea now would be to calculate each point coordinates along the curved spine
            # calculate the derivatives of the two spline functions.
            sprime = interpolate.splev(Z,tck_X,der=1)
            tprime = interpolate.splev(Z,tck_Y,der=1)

            # Local functions
            # calculate the lenght of a 3d curve portion
            # ==========================================================================================
            def integrand(z):
                return (sqrt(1 + sprime[z]*sprime[z] + tprime[z]*tprime[z]))

            def integrale(x):
                return quad(integrand, 0, x)[0]

            # normalize vectors functions
            # ==========================================================================================
            def mag(V):
                return sqrt(sum([x * x for x in V]))

            def n(V):
                v_m = mag(V)
                return [vi / v_m for vi in V]


            # txt file of centerline points 25 mm away along the spine (usefull to create future PSE landmarks)
            step = parameters.step
            fileID = open('tmp.centerline_fitted_orthogonal_resampling_pad' + str(step) + '.txt','w')
            fileID.close()

            # txt file of centerline points 1 mm away along the spine (usefull to resample the spine in each orthogonal plane of the previous centerline points considered)
            nostep = 1
            fileID = open('tmp.centerline_fitted_orthogonal_resampling.txt','w')
            fileID.close()


            # calculate coordinates of centerline points 25 mm away along the spine (usefull to create future PSE landmarks)

            # use of a while loop
            count = 0

            while (count <= lenght):

                if count == 0:
                    #if round(quad(integrand, 0, count)[0],1) % step == 0:
                    if round(quad(integrand, 0, count)[0],0) % step == 0:
                        print 'The point of z: ' + str(count) + ' is selected'
                        #print round(quad(integrand, 0, count)[0],1)
                        print round(quad(integrand, 0, count)[0],0)
                        z = count
                        # find x and y of the point
                        x = interpolate.splev(z,tck_X,der=0)
                        y = interpolate.splev(z,tck_Y,der=0)
                        fileID = open('tmp.centerline_fitted_orthogonal_resampling_pad' + str(step) + '.txt','a')
                        #fileID.write("%f %f %f %f\n" %(x, y, z, round(quad(integrand, 0, count)[0],1)))
                        fileID.write("%f %f %f %f\n" %(x, y, z, round(quad(integrand, 0, count)[0],0)))
                        fileID.close()
                    #count = count + 0.01
                    count = count + 0.1
                else:
                    #if round(quad(integrand, 0, count)[0],1) % step == 0 and round(quad(integrand, 0, count)[0],1)!=round(quad(integrand, 0, z)[0],1):
                    if round(quad(integrand, 0, count)[0],0) % step == 0 and round(quad(integrand, 0, count)[0],0)!=round(quad(integrand, 0, z)[0],0):
                        print "Test"
                        #print round(quad(integrand, 0, count)[0],1)
                        #print round(quad(integrand, 0, z)[0],1)+step
                        print round(quad(integrand, 0, count)[0],0)
                        print round(quad(integrand, 0, z)[0],0)+step
                        #if round(quad(integrand, 0, count)[0],1)==round(quad(integrand, 0, z)[0],1)+step:
                        if round(quad(integrand, 0, count)[0],0)==round(quad(integrand, 0, z)[0],0)+step:
                            print "Consecutive values"
                            print 'The point of z: ' + str(count) + ' is selected'
                            #print round(quad(integrand, 0, count)[0],1)
                            print round(quad(integrand, 0, count)[0],0)
                            z = count
                            # find x and y of the point
                            x = interpolate.splev(z,tck_X,der=0)
                            y = interpolate.splev(z,tck_Y,der=0)
                            fileID = open('tmp.centerline_fitted_orthogonal_resampling_pad' + str(step) + '.txt','a')
                            #fileID.write("%f %f %f %f\n" %(x, y, z, round(quad(integrand, 0, count)[0],1)))
                            fileID.write("%f %f %f %f\n" %(x, y, z, round(quad(integrand, 0, count)[0],0)))
                            fileID.close()
                        else:
                            print "Values not consecutives"

                    #count = count + 0.01
                    count = count + 0.1

            if parameters.debug == 1:
                cmd = 'cp ' + 'tmp.centerline_fitted_orthogonal_resampling_pad' + str(step) + '.txt txt_files/' + 'centerline_fitted_orthogonal_resampling_pad' + str(step) + '.txt'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)


            # calculate coordinates of centerline points 1 mm away along the spine (usefull to resample the spine in each orthogonal plane of the previous centerline points considered)

            # use of a while loop
            count = 0

            while (count <= lenght):

                if count == 0:
                    #if round(quad(integrand, 0, count)[0],1) % nostep == 0:
                    if round(quad(integrand, 0, count)[0],0) % nostep == 0:
                        print 'The point of z: ' + str(count) + ' is selected'
                        #print round(quad(integrand, 0, count)[0],1)
                        print round(quad(integrand, 0, count)[0],0)
                        z = count
                        # find x and y of the point
                        x = interpolate.splev(z,tck_X,der=0)
                        y = interpolate.splev(z,tck_Y,der=0)
                        fileID = open('tmp.centerline_fitted_orthogonal_resampling.txt','a')
                        fileID.write("%f %f %f\n" %(x, y, z))
                        fileID.close()
                    #count = count + 0.01
                    count = count + 0.1
                else:
                    #if round(quad(integrand, 0, count)[0],1) % nostep == 0 and round(quad(integrand, 0, count)[0],1)!=round(quad(integrand, 0, z)[0],1):
                    if round(quad(integrand, 0, count)[0],0) % nostep == 0 and round(quad(integrand, 0, count)[0],0)!=round(quad(integrand, 0, z)[0],0):
                        print "Test"
                        #print round(quad(integrand, 0, count)[0],1)
                        #print round(quad(integrand, 0, z)[0],1)+nostep
                        print round(quad(integrand, 0, count)[0],0)
                        print round(quad(integrand, 0, z)[0],0)+nostep
                        #if round(quad(integrand, 0, count)[0],1)==round(quad(integrand, 0, z)[0],1)+nostep:
                        if round(quad(integrand, 0, count)[0],0)==round(quad(integrand, 0, z)[0],0)+nostep:
                            print "Consecutive values"
                            print 'The point of z: ' + str(count) + ' is selected'
                            #print round(quad(integrand, 0, count)[0],1)
                            print round(quad(integrand, 0, count)[0],0)
                            z = count
                            # find x and y of the point
                            x = interpolate.splev(z,tck_X,der=0)
                            y = interpolate.splev(z,tck_Y,der=0)
                            fileID = open('tmp.centerline_fitted_orthogonal_resampling.txt','a')
                            fileID.write("%f %f %f\n" %(x, y, z))
                            fileID.close()
                        else:
                            print "Values not consecutives"

                    #count = count + 0.01
                    count = count + 0.1


            #debug
            #FILE_VOLUME = 'tmp.errsm_24_t2_cropped_APRLIS'
            #FILE_VOLUME_OUTPUT = 'errsm_24_t2_cropped'
            #step = 25
            #nostep = 1

            # function which calculate the rotation of a vector in three dimensions using the Euler Rodrigues formula.
            # axis = axis of rotation, theta = angle of rotation in radian
            def rotation_matrix(axis,theta):
                axis = axis/np.sqrt(np.dot(axis,axis))
                a = np.cos(theta/2)
                b,c,d = -axis*np.sin(theta/2)
                return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                                 [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                                 [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


            if parameters.debug == 1:
                cmd = 'cp ' + 'tmp.centerline_fitted_orthogonal_resampling.txt txt_files/' + 'centerline_fitted_orthogonal_resampling.txt'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)

            # read nifti input file
            centerline = load(FILE_VOLUME + '.nii.gz')
            # 3d array for each x y z voxel values for the input nifti image
            data = centerline.get_data()
            shape = data.shape
            print "Input volume dim.:"
            print shape

            # read the txt file of fitted centerline points 25 mm away along the spine
            file = open('tmp.centerline_fitted_orthogonal_resampling_pad' + str(step) + '.txt', 'rb')
            data_centerline = reader(file, delimiter=' ')
            table_centerline = [row for row in data_centerline]

            # count all the lines not empty in the txt file to determine the size of the M matrix defined below
            lines_counter = 0

            with open('tmp.centerline_fitted_orthogonal_resampling_pad' + str(step) + '.txt') as f:
                for line in f:
                    if line != '\n':
                        lines_counter += 1

            lenght = lines_counter - 1
            #print "Number of centerline points:"
            #print lenght

            lines_counter_nopad = 0

            with open('tmp.centerline_fitted_orthogonal_resampling.txt') as f:
                for line in f:
                    if line != '\n':
                        lines_counter_nopad += 1

            lenght_nopad = lines_counter_nopad - 1
            #print "Number of centerline points:"
            #print lenght


            # creation of a matrix resuming coefficients for all equations of orthogonal plans to the fitted centerline at the previous points 25mm away: Ai*x + Bi*y + Ci*z + Di = 0

            # create a list containing other lists initialized to 0
            M = [[0 for x in xrange(4)] for x in xrange(lenght)]

            for i in range(0, lenght):
                M[i][0] = float(table_centerline[i + 1][0]) - float(table_centerline[i][0])
                M[i][1] = float(table_centerline[i + 1][1]) - float(table_centerline[i][1])
                M[i][2] = float(table_centerline[i + 1][2]) - float(table_centerline[i][2])
                M[i][3] = - float(table_centerline[i][0]) * (M[i][0]) - float(table_centerline[i][1]) * (M[i][1]) - float(table_centerline[i][2]) * (M[i][2])

            # initialize normal and tangent vectors for each orthogonal plan to the fitted centerline
            Normal_vect_orthogonal = [[0 for x in xrange(3)] for x in xrange(lenght)]
            Tangent_vect_orthogonal_x = [[0 for x in xrange(3)] for x in xrange(lenght)]
            Tangent_vect_orthogonal_y = [[0 for x in xrange(3)] for x in xrange(lenght)]

            # initialize normal and tangent vectors for each horizontal plan to the fitted centerline
            Normal_vect_horizontal = [[0 for x in xrange(3)] for x in xrange(lenght)]
            Tangent_vect_horizontal_x = [[0 for x in xrange(3)] for x in xrange(lenght)]
            Tangent_vect_horizontal_y = [[0 for x in xrange(3)] for x in xrange(lenght)]

            # compute normal and tangent vectors for each orthogonal plane to the fitted centerline
            for i in range(0, lenght):
                Normal_vect_orthogonal[i][0] = M[i][0]
                Normal_vect_orthogonal[i][1] = M[i][1]
                Normal_vect_orthogonal[i][2] = M[i][2]
                # normalize the normal vector
                Normal_vect_orthogonal[i] = n(Normal_vect_orthogonal[i])

                # solve a set of two equations with two unknown variables to find tangent vector for each orthogonal plan to the centerline (tangent vector chosen in y plane with x > x_centerline) -> old way not working
                #x = Symbol('x')
                #z = Symbol('z')
                #x, z = nsolve([Normal_vect_orthogonal[i][0] * x + Normal_vect_orthogonal[i][2] * z, float(table_centerline[i][0]) + x + float(table_centerline[i][2]) + z + M[i][3]], [x, z], [1, 1])
                # By default, the result chosen is the opposite vector we want

                # define the intersection between the orthogonal plane of the centerline and an arbitrary plane (here the plane where each x is equal to the abscisse of the centerline point considered for the actual orthogonal plane).
                # This intesection, defined as a vector, will be our rotation axis
                axis = cross(Normal_vect_orthogonal[i], [1, 0, 0])
                v = [Normal_vect_orthogonal[i][0],Normal_vect_orthogonal[i][1],Normal_vect_orthogonal[i][2]]
                theta = (np.pi)/2 #radian

                Tangent_vect_orthogonal_x[i] = np.dot(rotation_matrix(axis,theta),v)

                # TEST
                #print Tangent_vect_orthogonal_x[i]
                #print Normal_vect_orthogonal[i]
                # print the dot product to make sure the tangent vector is in the plane.
                #print sum([x * y for x, y in zip(Tangent_vect_orthogonal_x[i], Normal_vect_orthogonal[i])])
                #print 'Tangent_vect_orthogonal_x[' + str(i) + '][0] = ' + str(Tangent_vect_orthogonal_x[i][0])
                #print 'Tangent_vect_orthogonal_x[' + str(i) + '][1] = ' + str(Tangent_vect_orthogonal_x[i][1])
                #print 'Tangent_vect_orthogonal_x[' + str(i) + '][2] = ' + str(Tangent_vect_orthogonal_x[i][2])

                # normalize the tangent vector previously created
                Tangent_vect_orthogonal_x[i] = n(Tangent_vect_orthogonal_x[i])
                # calculate Tangent_vect_orthogonal_y: Normal_vect^Tangent_vect_orthogonal_x
                Tangent_vect_orthogonal_y[i] = cross(Normal_vect_orthogonal[i], Tangent_vect_orthogonal_x[i])
                # normalize tangent vector y
                Tangent_vect_orthogonal_y[i] = n(Tangent_vect_orthogonal_y[i])

            # compute normal and tangent vectors for each horizontal plan to the fitted centerline
            for i in range(0, lenght):
                Normal_vect_horizontal[i][0] = 0
                Normal_vect_horizontal[i][1] = 0
                Normal_vect_horizontal[i][2] = 1
                # normalize normal vector
                Normal_vect_horizontal[i] = n(Normal_vect_horizontal[i])
                Tangent_vect_horizontal_x[i][0] = 1
                Tangent_vect_horizontal_x[i][1] = 0
                Tangent_vect_horizontal_x[i][2] = 0
                # normalize tangent vector creation
                Tangent_vect_horizontal_x[i] = n(Tangent_vect_horizontal_x[i])

                #calculate Tangent_vect_horizontal_y: Normal_vect^Tangent_vect_horizontal_x
                Tangent_vect_horizontal_y[i] = cross(Normal_vect_horizontal[i], Tangent_vect_horizontal_x[i])
                Tangent_vect_horizontal_y[i] = n(Tangent_vect_horizontal_y[i])

            landmarks_orthogonal = [[[0 for x in xrange(shape[2])] for x in xrange(shape[1])] for x in xrange(shape[0])]
            landmarks_orthogonal_size = array(landmarks_orthogonal).shape
            landmarks_horizontal = [[[0 for x in xrange(lenght_nopad)] for x in xrange(shape[1])] for x in xrange(shape[0])]
            landmarks_horizontal_size = array(landmarks_horizontal).shape

            # create PSE landmarks
            # create a timer to increment landmarks value.
            landmark_value = 1
            landmark_value_horizontal = 1

            # define the padding value
            padding = 50

            # create txt files
            fileID = open(FILE_VOLUME + '_orthogonal_landmarks.txt', 'w')
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            fileID = open(FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.txt', 'w')
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            fileID = open(FILE_VOLUME + '_horizontal_landmarks.txt', 'w')
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()
            fileID = open(FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.txt', 'w')
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()

            for l in range(0, lenght):
                # calculate the origin of the finite orthogonal mesh
                orthogonal_mesh_origin = [0 for x in xrange(3)]
                orthogonal_mesh_origin[0] = float(table_centerline[l][0])
                orthogonal_mesh_origin[1] = float(table_centerline[l][1])
                orthogonal_mesh_origin[2] = float(table_centerline[l][2])
                horizontal_mesh_origin = [0 for x in xrange(3)]
                horizontal_mesh_origin[0] = shape[0]/2
                horizontal_mesh_origin[1] = shape[1]/2
                horizontal_mesh_origin[2] = float(table_centerline[l][3])

                orthogonal_mesh_pos_centerline = [0 for x in xrange(3)]
                orthogonal_mesh_pos_x = [0 for x in xrange(3)]
                orthogonal_mesh_pos__x = [0 for x in xrange(3)]
                orthogonal_mesh_pos_y = [0 for x in xrange(3)]
                orthogonal_mesh_pos__y = [0 for x in xrange(3)]
                horizontal_mesh_pos_centerline = [0 for x in xrange(3)]
                horizontal_mesh_pos_x = [0 for x in xrange(3)]
                horizontal_mesh_pos__x = [0 for x in xrange(3)]
                horizontal_mesh_pos_y = [0 for x in xrange(3)]
                horizontal_mesh_pos__y = [0 for x in xrange(3)]

                orthogonal_mesh_pos_centerline = orthogonal_mesh_origin
                horizontal_mesh_pos_centerline = horizontal_mesh_origin

                # here, it defines the coordinates of the orthogonal and horizontal cross
                orthogonal_mesh_pos_x[0] = orthogonal_mesh_origin[0] + parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_x[l][0])
                orthogonal_mesh_pos_x[1] = orthogonal_mesh_origin[1] + parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_x[l][1])
                orthogonal_mesh_pos_x[2] = orthogonal_mesh_origin[2] + parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_x[l][2])
                print 'orthogonal_mesh_pos_x: ' + str(orthogonal_mesh_pos_x)

                orthogonal_mesh_pos__x[0] = orthogonal_mesh_origin[0] - parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_x[l][0])
                orthogonal_mesh_pos__x[1] = orthogonal_mesh_origin[1] - parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_x[l][1])
                orthogonal_mesh_pos__x[2] = orthogonal_mesh_origin[2] - parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_x[l][2])
                print 'orthogonal_mesh_pos__x: ' + str(orthogonal_mesh_pos__x)

                orthogonal_mesh_pos_y[0] = orthogonal_mesh_origin[0] + parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_y[l][0])
                orthogonal_mesh_pos_y[1] = orthogonal_mesh_origin[1] + parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_y[l][1])
                orthogonal_mesh_pos_y[2] = orthogonal_mesh_origin[2] + parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_y[l][2])
                print 'orthogonal_mesh_pos_y: ' + str(orthogonal_mesh_pos_y)

                orthogonal_mesh_pos__y[0] = orthogonal_mesh_origin[0] - parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_y[l][0])
                orthogonal_mesh_pos__y[1] = orthogonal_mesh_origin[1] - parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_y[l][1])
                orthogonal_mesh_pos__y[2] = orthogonal_mesh_origin[2] - parameters.landmarks_cross_size*float(Tangent_vect_orthogonal_y[l][2])
                print 'orthogonal_mesh_pos__y: ' + str(orthogonal_mesh_pos__y)

                horizontal_mesh_pos_x[0] = horizontal_mesh_origin[0] + parameters.landmarks_cross_size*float(Tangent_vect_horizontal_x[l][0])
                horizontal_mesh_pos_x[1] = horizontal_mesh_origin[1] + parameters.landmarks_cross_size*float(Tangent_vect_horizontal_x[l][1])
                horizontal_mesh_pos_x[2] = horizontal_mesh_origin[2] + parameters.landmarks_cross_size*float(Tangent_vect_horizontal_x[l][2])
                print 'horizontal_mesh_pos_x: ' + str(horizontal_mesh_pos_x)

                horizontal_mesh_pos__x[0] = horizontal_mesh_origin[0] - parameters.landmarks_cross_size*float(Tangent_vect_horizontal_x[l][0])
                horizontal_mesh_pos__x[1] = horizontal_mesh_origin[1] - parameters.landmarks_cross_size*float(Tangent_vect_horizontal_x[l][1])
                horizontal_mesh_pos__x[2] = horizontal_mesh_origin[2] - parameters.landmarks_cross_size*float(Tangent_vect_horizontal_x[l][2])
                print 'horizontal_mesh_pos__x: ' + str(horizontal_mesh_pos__x)

                horizontal_mesh_pos_y[0] = horizontal_mesh_origin[0] + parameters.landmarks_cross_size*float(Tangent_vect_horizontal_y[l][0])
                horizontal_mesh_pos_y[1] = horizontal_mesh_origin[1] + parameters.landmarks_cross_size*float(Tangent_vect_horizontal_y[l][1])
                horizontal_mesh_pos_y[2] = horizontal_mesh_origin[2] + parameters.landmarks_cross_size*float(Tangent_vect_horizontal_y[l][2])
                print 'horizontal_mesh_pos_y: ' + str(horizontal_mesh_pos_y)

                horizontal_mesh_pos__y[0] = horizontal_mesh_origin[0] - parameters.landmarks_cross_size*float(Tangent_vect_horizontal_y[l][0])
                horizontal_mesh_pos__y[1] = horizontal_mesh_origin[1] - parameters.landmarks_cross_size*float(Tangent_vect_horizontal_y[l][1])
                horizontal_mesh_pos__y[2] = horizontal_mesh_origin[2] - parameters.landmarks_cross_size*float(Tangent_vect_horizontal_y[l][2])
                print 'horizontal_mesh_pos__y: ' + str(horizontal_mesh_pos__y)

                # allocate the value of the landmark to the center of the cross for the orthogonal case
                landmarks_orthogonal[int(round(orthogonal_mesh_pos_centerline[0]))][int(round(orthogonal_mesh_pos_centerline[1]))][int(round(orthogonal_mesh_pos_centerline[2]))] = landmark_value
                #landmarks_orthogonal[int(round(orthogonal_mesh_pos_centerline[0]))][int(round(orthogonal_mesh_pos_centerline[1]))][int(round(orthogonal_mesh_pos_centerline[2]))] = 1

                # write the point in a txt file
                fileID = open(FILE_VOLUME + '_orthogonal_landmarks.txt', 'a')
                fileID.write('%f\t%f\t%f\t%f\n' %(float(orthogonal_mesh_pos_centerline[0]), float(orthogonal_mesh_pos_centerline[1]), float(orthogonal_mesh_pos_centerline[2]), 1))
                fileID.close()

                # write the point in a txt file
                fileID = open(FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.txt', 'a')
                fileID.write('%f\t%f\t%f\t%f\n' %(float(orthogonal_mesh_pos_centerline[0]), float(orthogonal_mesh_pos_centerline[1]), float(orthogonal_mesh_pos_centerline[2])+padding, 1))
                fileID.close()

                # allocate the value of the landmark to the center of the cross for the horizontal case
                landmarks_horizontal[int(round(horizontal_mesh_pos_centerline[0]))][int(round(horizontal_mesh_pos_centerline[1]))][int(round(horizontal_mesh_pos_centerline[2]))] = landmark_value_horizontal
                #landmarks_horizontal[int(round(horizontal_mesh_pos_centerline[0]))][int(round(horizontal_mesh_pos_centerline[1]))][int(round(horizontal_mesh_pos_centerline[2]))] = 1


                # write the point in a txt file
                fileID = open(FILE_VOLUME + '_horizontal_landmarks.txt', 'a')
                fileID.write('%f\t%f\t%f\t%f\n' %(float(horizontal_mesh_pos_centerline[0]), float(horizontal_mesh_pos_centerline[1]), float(horizontal_mesh_pos_centerline[2]), 1))
                fileID.close()

                # write the point in a txt file
                fileID = open(FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.txt', 'a')
                fileID.write('%f\t%f\t%f\t%f\n' %(float(horizontal_mesh_pos_centerline[0]), float(horizontal_mesh_pos_centerline[1]), float(horizontal_mesh_pos_centerline[2])+padding, 1))
                fileID.close()

                landmark_value=landmark_value + 1
                print landmark_value
                landmark_value_horizontal=landmark_value_horizontal + 1
                print landmark_value_horizontal

                if (orthogonal_mesh_pos_x[0] > shape[0]-1):
                    print "x outside (upper limit)"
                    print orthogonal_mesh_pos_x[0]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_x[0]))][int(round(orthogonal_mesh_pos_x[1]))][int(round(orthogonal_mesh_pos_x[2]))] = 0
                elif (orthogonal_mesh_pos_x[1] > shape[1]-1):
                    print "y outside (upper limit)"
                    print orthogonal_mesh_pos_x[1]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_x[0]))][int(round(orthogonal_mesh_pos_x[1]))][int(round(orthogonal_mesh_pos_x[2]))] = 0
                elif (orthogonal_mesh_pos_x[2] > shape[2]-1):
                    print "z outside (upper limit)"
                    print orthogonal_mesh_pos_x[2]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_x[0]))][int(round(orthogonal_mesh_pos_x[1]))][int(round(orthogonal_mesh_pos_x[2]))] = 0
                elif (orthogonal_mesh_pos_x[0] < 0):
                    print "x outside (lower limit)"
                    print orthogonal_mesh_pos_x[0]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_x[0]))][int(round(orthogonal_mesh_pos_x[1]))][int(round(orthogonal_mesh_pos_x[2]))] = 0
                elif (orthogonal_mesh_pos_x[1] < 0):
                    print "y outside (lower limit)"
                    print orthogonal_mesh_pos_x[1]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_x[0]))][int(round(orthogonal_mesh_pos_x[1]))][int(round(orthogonal_mesh_pos_x[2]))] = 0
                elif (orthogonal_mesh_pos_x[2] < 0):
                    print "z outside (lower limit)"
                    print orthogonal_mesh_pos_x[2]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_x[0]))][int(round(orthogonal_mesh_pos_x[1]))][int(round(orthogonal_mesh_pos_x[2]))] = 0
                else:
                    print "point inside"

                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_x[0]))][int(round(orthogonal_mesh_pos_x[1]))][int(round(orthogonal_mesh_pos_x[2]))] = landmark_value
                    #landmarks_orthogonal[int(round(orthogonal_mesh_pos_x[0]))][int(round(orthogonal_mesh_pos_x[1]))][int(round(orthogonal_mesh_pos_x[2]))] = 1

                    # write the point in a txt file
                    fileID = open(FILE_VOLUME + '_orthogonal_landmarks.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(orthogonal_mesh_pos_x[0]), float(orthogonal_mesh_pos_x[1]), float(orthogonal_mesh_pos_x[2]), 1))
                    fileID.close()

                    fileID = open(FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(orthogonal_mesh_pos_x[0]), float(orthogonal_mesh_pos_x[1]), float(orthogonal_mesh_pos_x[2])+padding, 1))
                    fileID.close()

                    landmarks_horizontal[int(round(horizontal_mesh_pos_x[0]))][int(round(horizontal_mesh_pos_x[1]))][int(round(horizontal_mesh_pos_x[2]))] = landmark_value_horizontal
                    #landmarks_horizontal[int(round(horizontal_mesh_pos_x[0]))][int(round(horizontal_mesh_pos_x[1]))][int(round(horizontal_mesh_pos_x[2]))] = 1

                    # write the point in a txt file
                    fileID = open(FILE_VOLUME + '_horizontal_landmarks.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(horizontal_mesh_pos_x[0]), float(horizontal_mesh_pos_x[1]), float(horizontal_mesh_pos_x[2]), 1))
                    fileID.close()

                    fileID = open(FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(horizontal_mesh_pos_x[0]), float(horizontal_mesh_pos_x[1]), float(horizontal_mesh_pos_x[2])+padding, 1))
                    fileID.close()

                    landmark_value=landmark_value+1
                    print landmark_value
                    landmark_value_horizontal=landmark_value_horizontal+1
                    print landmark_value_horizontal


                if (orthogonal_mesh_pos__y[0] > shape[0]-1):
                    print "x outside (upper limit)"
                    print orthogonal_mesh_pos__y[0]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__y[0]))][int(round(orthogonal_mesh_pos__y[1]))][int(round(orthogonal_mesh_pos__y[2]))] = 0
                elif (orthogonal_mesh_pos__y[1] > shape[1]-1):
                    print "y outside (upper limit)"
                    print orthogonal_mesh_pos__y[1]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__y[0]))][int(round(orthogonal_mesh_pos__y[1]))][int(round(orthogonal_mesh_pos__y[2]))] = 0
                elif (orthogonal_mesh_pos__y[2] > shape[2]-1):
                    print "z outside (upper limit)"
                    print orthogonal_mesh_pos__y[2]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__y[0]))][int(round(orthogonal_mesh_pos__y[1]))][int(round(orthogonal_mesh_pos__y[2]))] = 0
                elif (orthogonal_mesh_pos__y[0] < 0):
                    print "x outside (lower limit)"
                    print orthogonal_mesh_pos__y[0]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__y[0]))][int(round(orthogonal_mesh_pos__y[1]))][int(round(orthogonal_mesh_pos__y[2]))] = 0
                elif (orthogonal_mesh_pos__y[1] < 0):
                    print "y outside (lower limit)"
                    print orthogonal_mesh_pos__y[1]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__y[0]))][int(round(orthogonal_mesh_pos__y[1]))][int(round(orthogonal_mesh_pos__y[2]))] = 0
                elif (orthogonal_mesh_pos__y[2] < 0):
                    print "z outside (lower limit)"
                    print orthogonal_mesh_pos__y[2]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__y[0]))][int(round(orthogonal_mesh_pos__y[1]))][int(round(orthogonal_mesh_pos__y[2]))] = 0
                else:
                    print "point inside"

                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__y[0]))][int(round(orthogonal_mesh_pos__y[1]))][int(round(orthogonal_mesh_pos__y[2]))] = landmark_value
                    #landmarks_orthogonal[int(round(orthogonal_mesh_pos__y[0]))][int(round(orthogonal_mesh_pos__y[1]))][int(round(orthogonal_mesh_pos__y[2]))] = 1

                    # write the point in a txt file
                    fileID = open(FILE_VOLUME + '_orthogonal_landmarks.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(orthogonal_mesh_pos__y[0]), float(orthogonal_mesh_pos__y[1]), float(orthogonal_mesh_pos__y[2]), 1))
                    fileID.close()

                    fileID = open(FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(orthogonal_mesh_pos__y[0]), float(orthogonal_mesh_pos__y[1]), float(orthogonal_mesh_pos__y[2])+padding, 1))
                    fileID.close()

                    landmarks_horizontal[int(round(horizontal_mesh_pos__y[0]))][int(round(horizontal_mesh_pos__y[1]))][int(round(horizontal_mesh_pos__y[2]))] = landmark_value_horizontal
                    #landmarks_horizontal[int(round(horizontal_mesh_pos__y[0]))][int(round(horizontal_mesh_pos__y[1]))][int(round(horizontal_mesh_pos__y[2]))] = 1

                    # write the point in a txt file
                    fileID = open(FILE_VOLUME + '_horizontal_landmarks.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(horizontal_mesh_pos__y[0]), float(horizontal_mesh_pos__y[1]), float(horizontal_mesh_pos__y[2]), 1))
                    fileID.close()

                    fileID = open(FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(horizontal_mesh_pos__y[0]), float(horizontal_mesh_pos__y[1]), float(horizontal_mesh_pos__y[2])+padding, 1))
                    fileID.close()

                    landmark_value=landmark_value+1
                    print landmark_value
                    landmark_value_horizontal=landmark_value_horizontal+1
                    print landmark_value_horizontal

                if (orthogonal_mesh_pos__x[0] > shape[0]-1):
                    print "x outside (upper limit)"
                    print orthogonal_mesh_pos__x[0]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__x[0]))][int(round(orthogonal_mesh_pos__x[1]))][int(round(orthogonal_mesh_pos__x[2]))] = 0
                elif (orthogonal_mesh_pos__x[1] > shape[1]-1):
                    print "y outside (upper limit)"
                    print orthogonal_mesh_pos__x[1]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__x[0]))][int(round(orthogonal_mesh_pos__x[1]))][int(round(orthogonal_mesh_pos__x[2]))] = 0
                elif (orthogonal_mesh_pos__x[2] > shape[2]-1):
                    print "z outside (upper limit)"
                    print orthogonal_mesh_pos__x[2]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__x[0]))][int(round(orthogonal_mesh_pos__x[1]))][int(round(orthogonal_mesh_pos__x[2]))] = 0
                elif (orthogonal_mesh_pos__x[0] < 0):
                    print "x outside (lower limit)"
                    print orthogonal_mesh_pos__x[0]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__x[0]))][int(round(orthogonal_mesh_pos__x[1]))][int(round(orthogonal_mesh_pos__x[2]))] = 0
                elif (orthogonal_mesh_pos__x[1] < 0):
                    print "y outside (lower limit)"
                    print orthogonal_mesh_pos__x[1]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__x[0]))][int(round(orthogonal_mesh_pos__x[1]))][int(round(orthogonal_mesh_pos__x[2]))] = 0
                elif (orthogonal_mesh_pos__x[2] < 0):
                    print "z outside (lower limit)"
                    print orthogonal_mesh_pos__x[2]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__x[0]))][int(round(orthogonal_mesh_pos__x[1]))][int(round(orthogonal_mesh_pos__x[2]))] = 0
                else:
                    print "point inside"

                    landmarks_orthogonal[int(round(orthogonal_mesh_pos__x[0]))][int(round(orthogonal_mesh_pos__x[1]))][int(round(orthogonal_mesh_pos__x[2]))] = landmark_value
                    #landmarks_orthogonal[int(round(orthogonal_mesh_pos__x[0]))][int(round(orthogonal_mesh_pos__x[1]))][int(round(orthogonal_mesh_pos__x[2]))] = 1

                    # write the point in a txt file
                    fileID = open(FILE_VOLUME + '_orthogonal_landmarks.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(orthogonal_mesh_pos__x[0]), float(orthogonal_mesh_pos__x[1]), float(orthogonal_mesh_pos__x[2]), 1))
                    fileID.close()

                    fileID = open(FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(orthogonal_mesh_pos__x[0]), float(orthogonal_mesh_pos__x[1]), float(orthogonal_mesh_pos__x[2])+padding, 1))
                    fileID.close()

                    landmarks_horizontal[int(round(horizontal_mesh_pos__x[0]))][int(round(horizontal_mesh_pos__x[1]))][int(round(horizontal_mesh_pos__x[2]))] = landmark_value_horizontal
                    #landmarks_horizontal[int(round(horizontal_mesh_pos__x[0]))][int(round(horizontal_mesh_pos__x[1]))][int(round(horizontal_mesh_pos__x[2]))] = 1

                    # write the point in a txt file
                    fileID = open(FILE_VOLUME + '_horizontal_landmarks.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(horizontal_mesh_pos__x[0]), float(horizontal_mesh_pos__x[1]), float(horizontal_mesh_pos__x[2]), 1))
                    fileID.close()

                    fileID = open(FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(horizontal_mesh_pos__x[0]), float(horizontal_mesh_pos__x[1]), float(horizontal_mesh_pos__x[2])+padding, 1))
                    fileID.close()

                    landmark_value=landmark_value+1
                    print landmark_value
                    landmark_value_horizontal=landmark_value_horizontal+1
                    print landmark_value_horizontal

                if (orthogonal_mesh_pos_y[0] > shape[0]-1):
                    print "x outside (upper limit)"
                    print orthogonal_mesh_pos_y[0]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_y[0]))][int(round(orthogonal_mesh_pos_y[1]))][int(round(orthogonal_mesh_pos_y[2]))] = 0
                elif (orthogonal_mesh_pos_y[1] > shape[1]-1):
                    print "y outside (upper limit)"
                    print orthogonal_mesh_pos_y[1]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_y[0]))][int(round(orthogonal_mesh_pos_y[1]))][int(round(orthogonal_mesh_pos_y[2]))] = 0
                elif (orthogonal_mesh_pos_y[2] > shape[2]-1):
                    print "z outside (upper limit)"
                    print orthogonal_mesh_pos_y[2]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_y[0]))][int(round(orthogonal_mesh_pos_y[1]))][int(round(orthogonal_mesh_pos_y[2]))] = 0
                elif (orthogonal_mesh_pos_y[0] < 0):
                    print "x outside (lower limit)"
                    print orthogonal_mesh_pos_y[0]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_y[0]))][int(round(orthogonal_mesh_pos_y[1]))][int(round(orthogonal_mesh_pos_y[2]))] = 0
                elif (orthogonal_mesh_pos_y[1] < 0):
                    print "y outside (lower limit)"
                    print orthogonal_mesh_pos_y[1]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_y[0]))][int(round(orthogonal_mesh_pos_y[1]))][int(round(orthogonal_mesh_pos_y[2]))] = 0
                elif (orthogonal_mesh_pos_y[2] < 0):
                    print "z outside (lower limit)"
                    print orthogonal_mesh_pos_y[2]
                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_y[0]))][int(round(orthogonal_mesh_pos_y[1]))][int(round(orthogonal_mesh_pos_y[2]))] = 0
                else:
                    print "point inside"

                    landmarks_orthogonal[int(round(orthogonal_mesh_pos_y[0]))][int(round(orthogonal_mesh_pos_y[1]))][int(round(orthogonal_mesh_pos_y[2]))] = landmark_value
                    #landmarks_orthogonal[int(round(orthogonal_mesh_pos_y[0]))][int(round(orthogonal_mesh_pos_y[1]))][int(round(orthogonal_mesh_pos_y[2]))] = 1

                    # write the point in a txt file
                    fileID = open(FILE_VOLUME + '_orthogonal_landmarks.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(orthogonal_mesh_pos_y[0]), float(orthogonal_mesh_pos_y[1]), float(orthogonal_mesh_pos_y[2]), 1))
                    fileID.close()

                    fileID = open(FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(orthogonal_mesh_pos_y[0]), float(orthogonal_mesh_pos_y[1]), float(orthogonal_mesh_pos_y[2])+padding, 1))
                    fileID.close()

                    landmarks_horizontal[int(round(horizontal_mesh_pos_y[0]))][int(round(horizontal_mesh_pos_y[1]))][int(round(horizontal_mesh_pos_y[2]))] = landmark_value_horizontal
                    #landmarks_horizontal[int(round(horizontal_mesh_pos_y[0]))][int(round(horizontal_mesh_pos_y[1]))][int(round(horizontal_mesh_pos_y[2]))] = 1

                    # write the point in a txt file
                    fileID = open(FILE_VOLUME + '_horizontal_landmarks.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(horizontal_mesh_pos_y[0]), float(horizontal_mesh_pos_y[1]), float(horizontal_mesh_pos_y[2]), 1))
                    fileID.close()

                    fileID = open(FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.txt', 'a')
                    fileID.write('%f\t%f\t%f\t%f\n' %(float(horizontal_mesh_pos_y[0]), float(horizontal_mesh_pos_y[1]), float(horizontal_mesh_pos_y[2])+padding, 1))
                    fileID.close()

                    landmark_value=landmark_value+1
                    print landmark_value
                    landmark_value_horizontal=landmark_value_horizontal+1
                    print landmark_value_horizontal


            # write the point in a txt file
            fileID = open(FILE_VOLUME + '_orthogonal_landmarks.txt', 'a')
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()

            fileID = open(FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.txt', 'a')
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()

            fileID = open(FILE_VOLUME + '_horizontal_landmarks.txt', 'a')
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()

            fileID = open(FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.txt', 'a')
            fileID.write('%f\t%f\t%f\t%f\n' %(0, 0, 0, 0))
            fileID.close()

            if parameters.debug == 1:
                cmd = 'cp ' + FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.txt txt_files/' + FILE_VOLUME_OUTPUT + '_APRLIS_horizontal_landmarks_pad' + str(padding) + '.txt'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.txt txt_files/' + FILE_VOLUME_OUTPUT + '_APRLIS_orthogonal_landmarks_pad' + str(padding) + '.txt'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME + '_horizontal_landmarks.txt txt_files/' + FILE_VOLUME_OUTPUT + '_APRLIS_horizontal_landmarks.txt'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME + '_orthogonal_landmarks.txt txt_files/' + FILE_VOLUME_OUTPUT + '_APRLIS_orthogonal_landmarks.txt'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)

            # copy the header of the input volume
            hdr = centerline.get_header()
            hdr_copy = hdr.copy()

            shape_output = (shape[1], shape[2], lenght)
            hdr_copy_output = hdr.copy()
            hdr_copy_output.set_data_shape(shape_output)

            data_numpy = array(landmarks_orthogonal)
            img = Nifti1Image(data_numpy, None, hdr_copy)
            #img = nib.Nifti1Image(data_numpy, np.eye(4))
            save(img, FILE_VOLUME + '_orthogonal_landmarks.nii.gz')

            data_numpy = array(landmarks_horizontal)
            img = Nifti1Image(data_numpy, None, hdr_copy_output)
            save(img, FILE_VOLUME + '_horizontal_landmarks.nii.gz')

            if parameters.debug == 1:
                cmd = 'cp ' + FILE_VOLUME + '_horizontal_landmarks.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_horizontal_landmarks.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME + '_orthogonal_landmarks.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_orthogonal_landmarks.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)



            # read the txt file of fitted centerline points 1 mm away along the spine
            file = open('tmp.centerline_fitted_orthogonal_resampling.txt', 'rb')
            data_centerline = reader(file, delimiter=' ')
            table_centerline = [row for row in data_centerline]

            # count all the lines not empty in the txt file to determine the size of the M matrix defined below
            lines_counter = 0

            with open('tmp.centerline_fitted_orthogonal_resampling.txt') as f:
                for line in f:
                    if line != '\n':
                        lines_counter += 1

            lenght = lines_counter - 1
            print "Number of centerline points:"
            print lenght

            # creation of a matrix resuming coefficients for all equations of orthogonal plans to the  fitted centerline at the previous points 1 mm away: Ai*x + Bi*y + Ci*z + Di = 0

            # create a list containing other lists initialized to 0
            M = [[0 for x in xrange(4)] for x in xrange(lenght)]

            for i in range(0, lenght):
                M[i][0] = float(table_centerline[i + 1][0]) - float(table_centerline[i][0])
                M[i][1] = float(table_centerline[i + 1][1]) - float(table_centerline[i][1])
                M[i][2] = float(table_centerline[i + 1][2]) - float(table_centerline[i][2])
                M[i][3] = - float(table_centerline[i][0]) * (M[i][0]) - float(table_centerline[i][1]) * (M[i][1]) - float(table_centerline[i][2]) * (M[i][2])

            # initialize normal and tangent vectors for each orthogonal plan to the fitted centerline
            Normal_vect_orthogonal = [[0 for x in xrange(3)] for x in xrange(lenght)]
            Tangent_vect_orthogonal_x = [[0 for x in xrange(3)] for x in xrange(lenght)]
            Tangent_vect_orthogonal_y = [[0 for x in xrange(3)] for x in xrange(lenght)]

            # compute normal and tangent vectors for each orthogonal plan to the fitted centerline
            for i in range(0, lenght):
                Normal_vect_orthogonal[i][0] = M[i][0]
                Normal_vect_orthogonal[i][1] = M[i][1]
                Normal_vect_orthogonal[i][2] = M[i][2]
                Normal_vect_orthogonal[i] = n(Normal_vect_orthogonal[i])
                # solve a set of two equations with two unknown variables to find tangent vector for each orthogonal plan to the centerline (tangent vector chosen in y plane with x > x_centerline)
                x = Symbol('x')
                z = Symbol('z')
                x, z = nsolve([Normal_vect_orthogonal[i][0] * x + Normal_vect_orthogonal[i][2] * z, float(table_centerline[i][0]) + x + float(table_centerline[i ][2]) + z + M[i][3]], [x, z], [1, 1])
                # by default, the result chosen is the opposite vector we want
                Tangent_vect_orthogonal_x[i][0] = -x
                Tangent_vect_orthogonal_x[i][1] = 0
                Tangent_vect_orthogonal_x[i][2] = -z
                # normalize tangent vector x
                Tangent_vect_orthogonal_x[i] = n(Tangent_vect_orthogonal_x[i])
                # calculate Tangent_vect_orthogonal_y: Normal_vect^Tangent_vect_orthogonal_x
                Tangent_vect_orthogonal_y[i] = cross(Normal_vect_orthogonal[i], Tangent_vect_orthogonal_x[i])
                # normalize tangent vector y
                Tangent_vect_orthogonal_y[i] = n(Tangent_vect_orthogonal_y[i])

            orthogonal_volume_resampled = [[[0 for x in xrange(lenght)] for x in xrange(shape[1])] for x in xrange(shape[0])]
            orthogonal_volume_resampled_size = array(orthogonal_volume_resampled).shape
            gaussian_mask_orthogonal_volume_resampled = [[[0 for x in xrange(lenght)] for x in xrange(shape[1])] for x in xrange(shape[0])]
            gaussian_mask_orthogonal_volume_resampled_size = array(gaussian_mask_orthogonal_volume_resampled).shape


            for l in range(0, lenght):
                # create the centerline points for the future gaussian mask creation
                horizontal_mesh_origin = [0 for x in xrange(3)]
                horizontal_mesh_origin[0] = (shape[0]/2)-10
                horizontal_mesh_origin[1] = shape[1]/2
                horizontal_mesh_origin[2] = l

                gaussian_mask_orthogonal_volume_resampled[int(round(horizontal_mesh_origin[0]))][int(round(horizontal_mesh_origin[1]))][int(round(horizontal_mesh_origin[2]))] = 1

            # write in nifti file the centerline binary volume usefull for the future gaussian mask
            data_numpy = array(gaussian_mask_orthogonal_volume_resampled)
            img = Nifti1Image(data_numpy, None, hdr_copy)
            save(img, FILE_VOLUME + '_gaussian_mask_orthogonal_resampling.nii.gz')

            if parameters.debug == 1:
                cmd = 'cp ' + FILE_VOLUME + '_gaussian_mask_orthogonal_resampling.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_gaussian_mask_orthogonal_resampling.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)


            # create a new volume where spinal cord is resampled along all of its orthogonal planes
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    for l in range(0, lenght):
                        # calculate the origin of the finite orthogonal mesh and the finite horizontal mesh
                        orthogonal_mesh_origin = [0 for x in xrange(3)]
                        orthogonal_mesh_origin[0] = float(table_centerline[l][0]) - (shape[0]/2) * float(Tangent_vect_orthogonal_x[l][0]) - (shape[1]/2) * float(Tangent_vect_orthogonal_y[l][0])
                        orthogonal_mesh_origin[1] = float(table_centerline[l][1]) - (shape[0]/2) * float(Tangent_vect_orthogonal_x[l][1]) - (shape[1]/2) * float(Tangent_vect_orthogonal_y[l][1])
                        orthogonal_mesh_origin[2] = float(table_centerline[l][2]) - (shape[0]/2) * float(Tangent_vect_orthogonal_x[l][2]) - (shape[1]/2) * float(Tangent_vect_orthogonal_y[l][2])

                        # fill the orthogonal mesh plane
                        orthogonal_mesh_pos = [0 for x in xrange(3)]
                        orthogonal_mesh_pos[0] = orthogonal_mesh_origin[0] + i*float(Tangent_vect_orthogonal_x[l][0]) + j*float(Tangent_vect_orthogonal_y[l][0])
                        orthogonal_mesh_pos[1] = orthogonal_mesh_origin[1] + i*float(Tangent_vect_orthogonal_x[l][1]) + j*float(Tangent_vect_orthogonal_y[l][1])
                        orthogonal_mesh_pos[2] = orthogonal_mesh_origin[2] + i*float(Tangent_vect_orthogonal_x[l][2]) + j*float(Tangent_vect_orthogonal_y[l][2])

                        if (orthogonal_mesh_pos[0] > shape[0]-1):
                            print "x outside (upper limit)"
                            print orthogonal_mesh_pos[0]
                            orthogonal_volume_resampled[i][j][l] = 0
                        elif (orthogonal_mesh_pos[1] > shape[1]-1):
                            print "y outside (upper limit)"
                            print orthogonal_mesh_pos[1]
                            orthogonal_volume_resampled[i][j][l] = 0
                        elif (orthogonal_mesh_pos[2] > shape[2]-1):
                            print "z outside (upper limit)"
                            print orthogonal_mesh_pos[2]
                            orthogonal_volume_resampled[i][j][l] = 0
                        elif (orthogonal_mesh_pos[0] < 0):
                            print "x outside (lower limit)"
                            print orthogonal_mesh_pos[0]
                            orthogonal_volume_resampled[i][j][l] = 0
                        elif (orthogonal_mesh_pos[1] < 0):
                            print "y outside (lower limit)"
                            print orthogonal_mesh_pos[1]
                            orthogonal_volume_resampled[i][j][l] = 0
                        elif (orthogonal_mesh_pos[2] < 0):
                            print "z outside (lower limit)"
                            print orthogonal_mesh_pos[2]
                            orthogonal_volume_resampled[i][j][l] = 0
                        else:
                            print "x inside"
                            print orthogonal_mesh_pos[0]
                            print "y inside"
                            print orthogonal_mesh_pos[1]
                            print "z inside"
                            print orthogonal_mesh_pos[2]
                            orthogonal_volume_resampled[i][j][l] = data[round(orthogonal_mesh_pos[0])][round(orthogonal_mesh_pos[1])][round(orthogonal_mesh_pos[2])]

            # write in nifti file the new orthogonal resampled along the spine volume
            data_numpy = array(orthogonal_volume_resampled)
            img = Nifti1Image(data_numpy, None, hdr_copy_output)
            save(img, FILE_VOLUME + '_orthogonal_resampling.nii.gz')

            if parameters.debug == 1:
                cmd = 'cp ' + FILE_VOLUME + '_orthogonal_resampling.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_orthogonal_resampling.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)

            # create a gaussian mask centered along the centerline in the orthogonally resampled volume
            cmd = 'fslsplit ' + str(FILE_VOLUME) + '_orthogonal_resampling.nii.gz ' + str(FILE_VOLUME) + '_orthogonal_resampling_splitZ -z'
            print('>> '+ cmd)
            status, output = getstatusoutput(cmd)

            cmd = 'fslsplit ' + str(FILE_VOLUME) + '_gaussian_mask_orthogonal_resampling.nii.gz ' + str(FILE_VOLUME) + '_gaussian_mask_orthogonal_resampling_splitZ -z'
            print('>> '+ cmd)
            status, output = getstatusoutput(cmd)

            # height of the entire input volume
            FILE_VOLUME_SLICE = load(str(FILE_VOLUME) + '_orthogonal_resampling.nii.gz')
            FILE_VOLUME_DATA = FILE_VOLUME_SLICE.get_data()
            FILE_VOLUME_SHAPE = FILE_VOLUME_DATA.shape
            HEIGHT = FILE_VOLUME_SHAPE[2] - 1

            if REFERENCE != HEIGHT:

                VARIABLE=0

                while VARIABLE <= HEIGHT:
                    FILE_DEST = FILE_VOLUME + '_orthogonal_resampling_splitZ' + str(VARIABLE).zfill(4)
                    FILE_BINARY = FILE_VOLUME + '_gaussian_mask_orthogonal_resampling_splitZ' + str(VARIABLE).zfill(4)
                    FILE_MASK = FILE_DEST + '-mask'

                    print 'Create a Gaussian mask in the orthogonal resampled space'

                    print '***************************'
                    print 'z = ' + str(VARIABLE)
                    print '***************************'

                    cmd = 'fslroi ' + FILE_BINARY + '.nii.gz ' + FILE_MASK + '.nii.gz' + ' 0 ' + '-1 ' + '0 ' + '-1 ' + '0 ' + '1'
                    print('>> ' + cmd)
                    status, output = getstatusoutput(cmd)

                    # FSL gaussian mask creation
                    cmd = 'fslmaths ' + FILE_MASK + ' -kernel gauss 6.5 -dilM -s 3 ' + FILE_MASK + '_gaussian'
                    print('>> ' + cmd)
                    status, output = getstatusoutput(cmd)

                    VARIABLE = VARIABLE + 1


            # merge the new gaussian mask
            VARIABLE=0

            while VARIABLE <= HEIGHT:
                FILE_DEST = FILE_VOLUME + '_orthogonal_resampling_splitZ' + str(VARIABLE).zfill(4)
                FILE_MASK = FILE_DEST + '-mask_gaussian'

                # merge each slice file into a pseudo list of image registered files
                if VARIABLE == 0:
                    FILE_MASK_LIST = FILE_MASK
                else:
                    FILE_MASK_LIST = FILE_MASK_LIST + ' ' + FILE_MASK

                VARIABLE=VARIABLE + 1

            # merge the images with -z axis [concatenate]
            cmd = 'fslmerge -z ' + FILE_VOLUME + '_orthogonal_resampled_gaussian_mask.nii.gz ' + FILE_MASK_LIST
            print('>> '+ cmd)
            status, PWD = getstatusoutput(cmd)

            if parameters.debug == 1:
                cmd = 'cp ' + FILE_VOLUME + '_orthogonal_resampled_gaussian_mask.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_orthogonal_resampled_gaussian_mask.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)



            print "End of the orthogonal resampling part"






            ############################################################################################################
            # Estimate a deformation field between the input image and the orthogonally straightened one.
            # considering the case when output file names have not been specified at the beginning by the user
            ############################################################################################################

            # Padding of all images involved with ANTs
            print 'Pad source image, straightened image, gaussian mask and landmarks ...'
            cmd = 'isct_c3d ' + FILE_VOLUME + '.nii.gz' + ' -pad 0x0x'+str(padding)+'vox 0x0x' + str(padding) + 'vox 0 -o ' + FILE_VOLUME + '_pad' + str(padding) + '.nii.gz'
            print(">> "+cmd)
            status, output = getstatusoutput(cmd)
            cmd = 'isct_c3d ' + FILE_VOLUME + '_orthogonal_landmarks.nii.gz' + ' -pad 0x0x' + str(padding) + 'vox 0x0x' + str(padding) + 'vox 0 -o ' + FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.nii.gz'
            print(">> "+cmd)
            status, output = getstatusoutput(cmd)
            cmd = 'isct_c3d ' + FILE_VOLUME + '_horizontal_landmarks.nii.gz' + ' -pad 0x0x' + str(padding) + 'vox 0x0x' + str(padding) + 'vox 0 -o ' + FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.nii.gz'
            print(">> "+cmd)
            status, output = getstatusoutput(cmd)
            if GAUSS != '':
                cmd = 'isct_c3d ' + GAUSS + '.nii.gz' + ' -pad 0x0x' + str(padding) + 'vox 0x0x' + str(padding) + 'vox 0 -o ' + GAUSS + '_pad' + str(padding) + '.nii.gz'
                print(">> "+cmd)
                status, output = getstatusoutput(cmd)
            if GAUSS == '':
                cmd = 'isct_c3d ' + FILE_VOLUME + '_gaussian_mask.nii.gz' + ' -pad 0x0x' + str(padding) + 'vox 0x0x' + str(padding) + 'vox 0 -o ' + FILE_VOLUME + '_gaussian_mask_pad' + str(padding) + '.nii.gz'
                print(">> "+cmd)
                status, output = getstatusoutput(cmd)
            cmd = 'isct_c3d ' + FILE_VOLUME + '_fitted_straightened.nii.gz' + ' -pad 0x0x' + str(padding) + 'vox 0x0x' + str(padding) + 'vox 0 -o ' + FILE_VOLUME + '_fitted_straightened_pad' + str(padding) + '.nii.gz'
            print(">> "+cmd)
            status, output = getstatusoutput(cmd)
            cmd = 'isct_c3d ' + FILE_VOLUME + '_orthogonal_resampling.nii.gz' + ' -pad 0x0x' + str(padding) + 'vox 0x0x' + str(padding) + 'vox 0 -o ' + FILE_VOLUME + '_orthogonal_resampling_pad' + str(padding) + '.nii.gz'
            print(">> "+cmd)
            status, output = getstatusoutput(cmd)
            cmd = 'isct_c3d ' + FILE_VOLUME + '_orthogonal_resampled_gaussian_mask.nii.gz' + ' -pad 0x0x' + str(padding) + 'vox 0x0x' + str(padding) + 'vox 0 -o ' + FILE_VOLUME + '_orthogonal_resampled_gaussian_mask_pad' + str(padding) + '.nii.gz'
            print(">> "+cmd)
            status, output = getstatusoutput(cmd)

            if parameters.debug == 1:
                cmd = 'cp ' + FILE_VOLUME + '_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_orthogonal_landmarks_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME + '_gaussian_mask_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_gaussian_mask_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME + '_fitted_straightened_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_fitted_straightened_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME + '_orthogonal_resampling_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_orthogonal_resampling_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME + '_orthogonal_resampled_gaussian_mask_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_orthogonal_resampled_gaussian_mask_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_APRLIS_horizontal_landmarks_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)


            # put all images usefull for ANTS (two landmarks images + orthogonal resampling + input image) in original input volume orientation to get a warping field in this orientation
            cmd = 'fslswapdim ' + FILE_VOLUME + '_orthogonal_resampling_pad' + str(padding) + '.nii.gz ' + str(final_orientation[0]) + str(final_orientation[1]) + ' ' + str(final_orientation[2]) + str(final_orientation[3]) + ' ' + str(final_orientation[4]) + str(final_orientation[5]) + ' tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling_pad' + str(padding) + '.nii.gz'
            print('>> '+ cmd)
            status, output = getstatusoutput(cmd)
            cmd = 'fslswapdim ' + FILE_VOLUME + '_orthogonal_resampling.nii.gz ' + str(final_orientation[0]) + str(final_orientation[1]) + ' ' + str(final_orientation[2]) + str(final_orientation[3]) + ' ' + str(final_orientation[4]) + str(final_orientation[5]) + ' tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling.nii.gz'
            print('>> '+ cmd)
            status, output = getstatusoutput(cmd)
            cmd = 'fslswapdim ' + FILE_VOLUME + '_pad' + str(padding) + '.nii.gz ' + str(final_orientation[0]) + str(final_orientation[1]) + ' ' + str(final_orientation[2]) + str(final_orientation[3]) + ' ' + str(final_orientation[4]) + str(final_orientation[5]) + ' tmp.' + FILE_VOLUME_OUTPUT + '_pad' + str(padding) + '.nii.gz'
            print('>> '+ cmd)
            status, output = getstatusoutput(cmd)
            cmd = 'fslswapdim ' + FILE_VOLUME + '_horizontal_landmarks_pad' + str(padding) + '.nii.gz ' + str(final_orientation[0]) + str(final_orientation[1]) + ' ' + str(final_orientation[2]) + str(final_orientation[3]) + ' ' + str(final_orientation[4]) + str(final_orientation[5]) + ' tmp.' + FILE_VOLUME_OUTPUT + '_horizontal_landmarks_pad' + str(padding) + '.nii.gz'
            print('>> '+ cmd)
            status, output = getstatusoutput(cmd)
            cmd = 'fslswapdim ' + FILE_VOLUME + '_orthogonal_landmarks_pad' + str(padding) + '.nii.gz ' + str(final_orientation[0]) + str(final_orientation[1]) + ' ' + str(final_orientation[2]) + str(final_orientation[3]) + ' ' + str(final_orientation[4]) + str(final_orientation[5]) + ' tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_landmarks_pad' + str(padding) + '.nii.gz'
            print('>> '+ cmd)
            status, output = getstatusoutput(cmd)
            if parameters.debug == 1:
                cmd = 'cp tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp tmp.' + FILE_VOLUME_OUTPUT + '_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp tmp.' + FILE_VOLUME_OUTPUT + '_horizontal_landmarks_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_horizontal_landmarks_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp tmp.' + FILE_VOLUME_OUTPUT + '_horizontal_landmarks_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_horizontal_landmarks_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)


            # use ANTs to find a warping field to straighten orthogonally the spine
            # apparently, using a gaussian mask on the orthogonal resampled image is not usefull at all
            cmd = 'ants 3 -m PSE[tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling_pad' + str(padding) + '.nii.gz,tmp.' + FILE_VOLUME_OUTPUT + '_pad' + str(padding) + '.nii.gz,tmp.' + FILE_VOLUME_OUTPUT + '_horizontal_landmarks_pad' + str(padding) + '.nii.gz,tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_landmarks_pad' + str(padding) + '.nii.gz,' + '0.2,100,1,0,1,100000] -o PSE -i 1000x1000x0 --number-of-affine-iterations 1000x1000x1000 -m CC[tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling_pad' + str(padding) + '.nii.gz,tmp.' + FILE_VOLUME_OUTPUT + '_pad' + str(padding) + '.nii.gz,' + '0.8,4] --use-all-metrics-for-convergence 1'
            print(">> "+cmd)
            status, output = getstatusoutput(cmd)
            print output

            # apply the PSE transformation previously calculated
            cmd = 'WarpImageMultiTransform 3 tmp.' + FILE_VOLUME_OUTPUT + '_pad' + str(padding) + '.nii.gz tmp.' + FILE_VOLUME_OUTPUT + '_reg_PSE_pad' + str(padding) + '.nii.gz -R tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling_pad' + str(padding) + '.nii.gz --use-BSpline PSEWarp.nii.gz PSEAffine.txt'
            print(">> "+cmd)
            status, output = getstatusoutput(cmd)
            print output

            cmd = 'WarpImageMultiTransform 3 ' + FILE_VOLUME_OUTPUT + '.nii.gz ' + FILE_VOLUME_OUTPUT + '_reg_PSE.nii.gz -R tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling.nii.gz --use-BSpline PSEWarp.nii.gz PSEAffine.txt'
            print(">> "+cmd)
            status, output = getstatusoutput(cmd)
            print output

            #store in png slice the result of the middle caracteristic slice
            # read nifti input file
            img = load(FILE_VOLUME_OUTPUT + '_reg_PSE.nii.gz')
            # 3d array for each x y z voxel values for the input nifti image
            data = img.get_data()
            shape = data.shape
            print "Input volume dimensions:"
            print shape

            snapshot = [[0 for x in xrange(shape[1])] for x in xrange(shape[0])]
            for i in range(0,shape[0]):
                for j in range(0,shape[1]):
                    snapshot[i][j] = data[i][j][int(shape[2]/2)]
            rotate_snapshot = ndimage.rotate(snapshot, 90)
            plt.imshow(rotate_snapshot, cmap = cm.Greys_r)
            plt.savefig('snapshot.png')
        
            # crop the warping field due to previous padding
            # bad way i think because i do not modify/crop the affine txt file
            cmd = 'fslroi PSEWarp.nii.gz PSEWarp_cropped.nii.gz 0 -1 0 -1 ' + str(padding) + ' ' + str(lenght-padding)
            print('>> '+ cmd)
            status, output = getstatusoutput(cmd)
        
            cmd = 'WarpImageMultiTransform 3 ' + FILE_VOLUME_OUTPUT + '.nii.gz tmp.' + FILE_VOLUME_OUTPUT + '_reg_PSE_warping_field_cropped.nii.gz -R tmp.' + FILE_VOLUME_OUTPUT + '_orthogonal_resampling.nii.gz --use-BSpline PSEWarp_cropped.nii.gz PSEAffine.txt'
            print(">> "+cmd)
            status, output = getstatusoutput(cmd)
            print output
        

            if parameters.debug == 1:
                cmd = 'cp ' + 'tmp.' + FILE_VOLUME_OUTPUT + '_reg_PSE_pad' + str(padding) + '.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_reg_PSE_pad' + str(padding) + '.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp ' + FILE_VOLUME_OUTPUT + '_reg_PSE.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_reg_PSE.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)
                cmd = 'cp tmp.' + FILE_VOLUME_OUTPUT + '_reg_PSE_warping_field_cropped.nii.gz output_images/' + FILE_VOLUME_OUTPUT + '_reg_PSE_warping_field_cropped.nii.gz'
                print('>> '+ cmd)
                status, output = getstatusoutput(cmd)



        elif DEFORMATION == '0':
            print 'Warping filed not calculated (no orthogonal centerline plane resampling approach)'

    elif DEFORMATION == '':
        print 'Warping field not calculated (no orthogonal centerline plane resampling approach) [Default]'










    ################################################################################################################
                                            ##### Remove temporary files #####
    ################################################################################################################

    cmd = 'rm tmp.*'
    print('>> ' + cmd)
    status, output = getstatusoutput(cmd)
    print 'Temporary files deleted'





########################################################################################################################
# START PROGRAM
########################################################################################################################
if __name__ == "__main__":
    # call the important variable structures
    parameters = parameters()
    main()

