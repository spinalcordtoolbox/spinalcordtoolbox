#!/usr/bin/env python

## @package sct_smooth_spinal_cord.py
#
# - from spinal cord RMI volume 3D (as nifty format) and centerline of the spinal cord (given by sct_get_centerline.py),
#   smooth the image along the spinalcord
#
# input:
# - full width at half maximum, FWHM, for Gaussian Kernel mask (in mm)
# - Anatomical image (.nii or .nii.gz)
# - Centerline (given by the function sct_get_centerline.py)
#
# ----------------------------------------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# - nibabel : <http://nipy.sourceforge.net/nibabel/>
# - numpy   : <http://www.numpy.org>
#
# EXTERNAL SOFTWARE
# - FSL: <http://fsl.fmrib.ox.ac.uk/fsl/>
# ----------------------------------------------------------------------------------------------------------------------
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Authors: Julien Cohen-Adad, Isabelle Bouchard
#
# License: see the LICENSE.TXT
# ======================================================================================================================

import os
import getopt
import sys
import time
import math
import sct_utils as sct
import nibabel
import numpy as np
from scipy import ndimage
from sct_convert import convert

## Default parameters
class param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.fwhm = 3
        self.remove_temp_files= 1
        self.deg_poly= 10
        self.width=25
        self.verbose = 1



#=======================================================================================================================
# main
#=======================================================================================================================
def main():

# Initialization
    fname_anat = ''
    fname_centerline = ''
    fwhm = param.fwhm
    width=param.width
    remove_temp_files = param.remove_temp_files
    start_time = time.time()
    verbose = param.verbose

    # extract path of the script
    path_script = os.path.dirname(__file__) + '/'

    # Parameters for debug mode
    if param.debug == 1:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_anat = '/home/django/ibouchard/errsm_22_t2_cropped_rpi.nii.gz'
        fname_centerline = '/home/django/ibouchard//errsm_22_t2_cropped_centerline.nii.gz'
        fwhm=1
        width=20

    # Check input param
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:c:f:w:r:')
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
            fwhm = int(arg)
        elif opt in ('w'):
            width=int(arg)
        elif opt in ('-r'):
            remove_temp_files = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_anat == '' or fname_centerline == '':
        usage()

    # check existence of input files
    sct.check_file_exist(fname_anat)
    sct.check_file_exist(fname_centerline)

    # extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)

    # extract path/file/extension
    path_centerline, file_centerline, ext_centerline = sct.extract_fname(fname_centerline)

    # Display arguments
    print '\nCheck input arguments...'
    print '.. Anatomical image:           ' + fname_anat
    print '.. Centerline:                 ' + fname_centerline
    print '.. Full width at half maximum:  ' + str(fwhm)
    print '.. Width of the square window: ' + str(width)

    path_tmp = sct.tmp_create(basename="smooth_spinal_cord")

    # Copying input data to tmp folder and convert to nii
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    sct.run('cp '+fname_anat+' '+path_tmp+'data'+ext_anat, verbose)
    sct.run('cp '+fname_centerline+' '+path_tmp+'centerline'+ext_centerline, verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # convert to nii format
    convert('data'+ext_anat, 'data.nii')
    convert('centerline'+ext_centerline, 'centerline.nii')

    # # Get dimensions of data
    # sct.printv('\nGet dimensions of data...', param.verbose)
    # nx, ny, nz, nt, px, py, pz, pt = Image('data.nii').dim

    #
    # #Delete existing tmp file in the current folder to avoid problems
    #     #Delete existing tmp file in the current folder to avoid problems
    # if os.path.isfile('tmp.anat.nii'):
    #     sct.run('rm tmp.anat.nii')
    # if os.path.isfile('tmp.centerline.nii'):
    #     sct.run('rm tmp.centerline.nii')
    #
    # # Convert to nii and delete nii.gz if still existing
    # print '\nCopy input data...'
    # sct.run('cp ' + fname_anat + ' tmp.anat'+ext_anat)
    # convert('data'+ext_data, 'data.nii')
    #
    # sct.run('fslchfiletype NIFTI tmp.anat')
    # if os.path.isfile('tmp.anat.nii.gz'):
    #     sct.run('rm tmp.anat.nii.gz')
    # print '.. Anatomical image copied'
    # sct.run('cp ' + fname_centerline + ' tmp.centerline'+ext_centerline)
    # sct.run('fslchfiletype NIFTI tmp.centerline')
    # if os.path.isfile('tmp.centerline.nii.gz'):
    #     sct.run('rm tmp.centerline.nii.gz')
    # print '.. Centerline image copied'


    # Open anatomical image
    #==========================================================================================
    # Reorient input anatomical volume into RL PA IS orientation
    print '\nReorient input volume to RL PA IS orientation...'
    sct.run(sct.fsloutput + 'fslswapdim tmp.anat RL PA IS tmp.anat_orient')


    print '\nGet dimensions of input anatomical image...'
    nx_a, ny_a, nz_a, nt_a, px_a, py_a, pz_a, pt_a = sct.get_dimension('tmp.anat_orient')
    #nx_a, ny_a, nz_a, nt_a, px_a, py_a, pz_a, pt_a = sct.get_dimension(fname_anat)
    print '.. matrix size: ' + str(nx_a) + ' x ' + str(ny_a) + ' x ' + str(nz_a)
    print '.. voxel size:  ' + str(px_a) + 'mm x ' + str(py_a) + 'mm x ' + str(pz_a) + 'mm'

    print '\nOpen anatomical volume...'
    file = nibabel.load('tmp.anat_orient.nii')
    #file = nibabel.load(fname_anat)
    data_anat = file.get_data()
    data_anat=np.array(data_anat)

    data_anat_smoothed=np.copy(data_anat)


    # Open centerline
    #==========================================================================================
    # Reorient binary point into RL PA IS orientation
    print '\nReorient centerline volume into RL PA IS orientation...'
    sct.run(sct.fsloutput + 'fslswapdim tmp.centerline RL PA IS tmp.centerline_orient')

    print '\nGet dimensions of input centerline...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('tmp.centerline_orient')
    #nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_centerline)
    print '.. matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz)
    print '.. voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm'

    print '\nOpen centerline volume...'
    file = nibabel.load('tmp.centerline_orient.nii')
    #file = nibabel.load(fname_centerline)
    data_centerline = file.get_data()

    #Loop across z and associate x,y coordinate with the point having maximum intensity
    x_centerline = [0 for iz in range(0, nz, 1)]
    y_centerline = [0 for iz in range(0, nz, 1)]
    z_centerline = [iz for iz in range(0, nz, 1)]
    for iz in range(0, nz, 1):
        x_centerline[iz], y_centerline[iz] = np.unravel_index(data_centerline[:, :, iz].argmax(),
                                                              data_centerline[:, :, iz].shape)
    del data_centerline


    # Fit polynomial function through centerline
    #==========================================================================================

    #Fit centerline in the Z-X plane using polynomial function
    print '\nFit centerline in the Z-X plane using polynomial function...'
    coeffsx = np.polyfit(z_centerline, x_centerline, deg=param.deg_poly)
    polyx = np.poly1d(coeffsx)
    x_centerline_fit = np.polyval(polyx, z_centerline)

    #Fit centerline in the Z-Y plane using polynomial function
    print '\nFit centerline in the Z-Y plane using polynomial function...'
    coeffsy = np.polyfit(z_centerline, y_centerline, deg=param.deg_poly)
    polyy = np.poly1d(coeffsy)
    y_centerline_fit = np.polyval(polyy, z_centerline)

    # Find tangent function of centerline along z
    #==========================================================================================

    # Find tangent to centerline in zx plane, along z
    print '\nFind tangent to centerline along z, in the Z-X plane...'
    poly_tangent_xz = np.polyder(polyx)
    tangent_xz = np.polyval(poly_tangent_xz, z_centerline)

    # Find tangent to centerline in zy plane, along z
    print '\nFind tangent to centerline along z, in the Z-Y plane...'
    poly_tangent_yz = np.polyder(polyy)
    tangent_yz = np.polyval(poly_tangent_yz, z_centerline)

	# Create a Gaussian kernel with users parameters
    #==========================================================================================
    print '\nGenerate a Gaussian kernel with users parameters...     '

    # Convert the fwhm given by users in standard deviation (sigma) and find the size of gaussian kernel knowing
    # that size_kernel=(6*sigma-1) must be odd
    sigma = int(np.round((fwhm/pz_a)*(math.sqrt(1/(2*(math.log(2)))))))
    size_kernel= (np.round(6*sigma))
    if size_kernel%2==0:
        size_kernel=size_kernel-1


    #Creates an  1D-array impulsion and apply a gaussian filter. The result is a Gaussian kernel.
    kernel_temp = np.zeros(size_kernel)
    kernel_temp[math.ceil(size_kernel/2)] = 1
    kernel= ndimage.filters.gaussian_filter1d(kernel_temp, sigma, order=0)
    sum_kernel=np.sum(kernel)

    print '.. Full width at half maximum: ' + str(fwhm)
    print '.. Kernel size : '+str(size_kernel)
    print '.. Sigma (Standard deviation): ' + str(sigma)

    del kernel_temp


    ## Smooth along the spinal cord
    ##==========================================================================================
    print '\nSmooth along the spinal cord...'


    print '\n Voxel position along z axis...'

    # Initialisations
    position=np.zeros(3)
    flag=np.zeros((nx_a,ny_a,nz_a))
    data_weight=np.ones((nx_a,ny_a,nz_a))
    smoothing_array=np.zeros(size_kernel)
    x_near=np.zeros(2)
    y_near=np.zeros(2)
    z_near=np.zeros(2)
    floor_position=np.zeros(3)
    ceil_position=np.zeros(3)
    position_d=np.zeros(3)

    #For every voxel along z axis,
    for iz in range(0,nz_a,1):

        print '.. '+str(iz+1)+ '/'+str(nz_a)

        # Determine the square area to smooth around the centerline
        xmin=x_centerline[iz]-int(width/2)
        xmax=x_centerline[iz]+int(width/2)
        ymin=y_centerline[iz]-int(width/2)
        ymax=y_centerline[iz]+int(width/2)

        #Find the angle between the tangent and the x axis in xz plane.
        theta_xz = -(math.atan(tangent_xz[iz]))

        #Find the angle between the tangent and the y axis in yz plane.
        theta_yz = -(math.atan(tangent_yz[iz]))

        #Construct a rotation array around y axis.
        Rxz=np.zeros((3,3))
        Rxz[1,1]=1
        Rxz[0,0]=(math.cos(theta_xz))
        Rxz[2,0]=(math.sin(theta_xz))
        Rxz[0,2]=-(math.sin(theta_xz))
        Rxz[2,2]=(math.cos(theta_xz))

        #Construct a rotation array around x axis.
        Ryz=np.zeros((3,3))
        Ryz[0,0]=1
        Ryz[1,1]=(math.cos(theta_yz))
        Ryz[1,2]=(math.sin(theta_yz))
        Ryz[2,1]=-(math.sin(theta_yz))
        Ryz[2,2]=(math.cos(theta_yz))


        #For every voxels in the given plane, included in the square area
        for ix in range(xmin,xmax,1):
            for iy in range(ymin,ymax,1):

                #The area to smooth has the same high as the 1D mask length
                isize=0
                centerline_point=[np.copy(x_centerline[iz]), np.copy(y_centerline[iz]), np.copy(iz)]


                #For every voxels along the line orthogonal to the considered plane and included in the kernel.
                #(Here we full a vector called smoothing_array, which has the same length as the kernel, is oriented in the direction of centerline and contains interpolated values of intensity)
                for isize in range(0,size_kernel, 1):

                    #Find the position in the xy plane, before rotation
                    position = [ix, iy, iz+isize-(np.floor(size_kernel/2))]

                    #Find the position after rotation by multiplying the position centered on centerline point with rotation array around x and y axis.
                    new_position= np.dot((np.dot((np.subtract(np.copy(position),centerline_point)), Rxz)), Ryz) + centerline_point

                    #If the resulting voxel is out of image boundaries, pad the smoothing array with a zero
                    if (new_position[0]<0)or (new_position[1]<0)or(new_position[2]<0)or(new_position[0]>nx_a-1)or (new_position[1]>ny_a-1)or(new_position[2]>nz_a-1):
                        smoothing_array[isize]=0
                    #Otherwise, fill the smoothing array with the linear interpolation of values around the actual position
                    else:

                    # Trilinear interpolation
                    #==========================================================================================================================================
                    # Determine the coordinates in grid surrounding the position of the central voxel and perform a trilinear interpolation
                        x_near[0]=np.copy(np.floor(new_position[0]))
                        x_near[1]=np.copy(np.ceil(new_position[0]))
                        xd=(new_position[0]-x_near[0])
                        y_near[0]=np.copy(np.floor(new_position[1]))
                        y_near[1]=np.copy(np.ceil(new_position[1]))
                        yd=(new_position[1]-y_near[0])
                        z_near[0]=np.copy(np.floor(new_position[2]))
                        z_near[1]=np.copy(np.ceil(new_position[2]))
                        zd=(new_position[2]-z_near[0])

                        c00=((data_anat[x_near[0],y_near[0],z_near[0]])*(1-xd))+((data_anat[x_near[1],y_near[0],z_near[0]])*(xd))
                        c10=((data_anat[x_near[0],y_near[1],z_near[0]])*(1-xd))+((data_anat[x_near[1],y_near[1],z_near[0]])*(xd))
                        c01=((data_anat[x_near[0],y_near[0],z_near[1]])*(1-xd))+((data_anat[x_near[1],y_near[0],z_near[1]])*(xd))
                        c11=((data_anat[x_near[0],y_near[1],z_near[1]])*(1-xd))+((data_anat[x_near[1],y_near[1],z_near[1]])*(xd))

                        c0=c00*(1-yd)+c10*yd
                        c1=c01*(1-yd)+c11*yd

                        smoothing_array[isize]=c0*(1-zd)+c1*zd

                    #If actual position is in the z=z_centerline plane, save the coordinates in the variable central_position. (Otherwise, don't save it).
                    if isize==(np.floor(size_kernel/2)):
                        central_position=np.copy(new_position)


                #If the central_position is out of boundaries, don't consider it anymore.
                if (central_position[0]<0)or (central_position[1]<0)or(central_position[2]<0)or(central_position[0]>nx_a-1)or (central_position[1]>ny_a-1)or(central_position[2]>nz_a-1):
                    continue

                else:
                    #Otherwise, perform the convolution of the smoothing_array and the kernel for the central voxel only (equivalent to element-wise multiply). Normalize the result.
                    result=((np.sum(np.copy(smoothing_array)*kernel))/sum_kernel)

                    # Determine the coordinates in grid surrounding the position of the central voxel
                    for i in range(0,3,1):
                        floor_position[i]=math.floor(central_position[i])
                        ceil_position[i]=math.ceil(central_position[i])
                        position_d[i]=central_position[i]-floor_position[i]



                    # Reverse trilinear interpolation
                    #==========================================================================================================================================
                    # Split the resuling intensity given by the convolution between the 8 voxels surrounding the point where the convolution is calculated (central_position).
                    # The array data_anat_smoothed is the the volume os the anatomical image smoothed alog the spinal cord.
                    # The array flag is a volume that indicates if a the corresponding voxel in the anatomical image is inside the smoothing area around the spinal cord and if there is already been an operation on this voxel.
                    # The default value of flag is 0. If it is set to 1, it means there is an operation on the corresponding voxel in anatomical image. Then we clear both the data_anat_smoothed and data_weight corresponding voxel to 0.
                    # The array data_weight represent the is represent the sum of weights used to calculate the intensity for every voxel. In a perfect case, this sum would be 1, but because there is an angle between
                    # two adjacent planes, the sum will be lower so we need to normalize the result. The default value for data_weight is 1, but once there is an operation on the corresponding voxel (flag=1), we accumulate the weights used.

                    if (flag[ceil_position[0],ceil_position[1],ceil_position[2]]==0):
                        data_anat_smoothed[ceil_position[0],ceil_position[1],ceil_position[2]]=0
                        data_weight[ceil_position[0],ceil_position[1],ceil_position[2]]=0
                        flag[ceil_position[0],ceil_position[1],ceil_position[2]]=1
                    weight=(position_d[0])*(position_d[1])*(position_d[2])
                    data_anat_smoothed[ceil_position[0],ceil_position[1],ceil_position[2]]=data_anat_smoothed[ceil_position[0],ceil_position[1],ceil_position[2]]+(weight*result)
                    data_weight[ceil_position[0],ceil_position[1],ceil_position[2]]=data_weight[ceil_position[0],ceil_position[1],ceil_position[2]]+(weight)

                    if (flag[floor_position[0],floor_position[1],floor_position[2]]==0):
                        data_anat_smoothed[floor_position[0],floor_position[1],floor_position[2]]=0
                        data_weight[floor_position[0],floor_position[1],floor_position[2]]=0
                        flag[floor_position[0],floor_position[1],floor_position[2]]=1
                    weight=(1-position_d[0])*(1-position_d[1])*(1-position_d[2])
                    data_anat_smoothed[floor_position[0],floor_position[1],floor_position[2]]=data_anat_smoothed[floor_position[0],floor_position[1],floor_position[2]]+(weight*result)
                    data_weight[floor_position[0],floor_position[1],floor_position[2]]=data_weight[floor_position[0],floor_position[1],floor_position[2]]+(weight)


                    if (flag[ceil_position[0],floor_position[1],floor_position[2]]==0):
                        data_anat_smoothed[ceil_position[0],floor_position[1],floor_position[2]]=0
                        data_weight[ceil_position[0],floor_position[1],floor_position[2]]=0
                        flag[ceil_position[0],floor_position[1],floor_position[2]]=1
                    weight=(position_d[0])*(1-position_d[1])*(1-position_d[2])
                    data_anat_smoothed[ceil_position[0],floor_position[1],floor_position[2]]=data_anat_smoothed[ceil_position[0],floor_position[1],floor_position[2]]+(weight*result)
                    data_weight[ceil_position[0],floor_position[1],floor_position[2]]=data_weight[ceil_position[0],floor_position[1],floor_position[2]]+(weight)

                    if (flag[ceil_position[0],ceil_position[1],floor_position[2]]==0):
                        data_anat_smoothed[ceil_position[0],ceil_position[1],floor_position[2]]=0
                        data_weight[ceil_position[0],ceil_position[1],floor_position[2]]=0
                        flag[ceil_position[0],ceil_position[1],floor_position[2]]=1
                    weight=(position_d[0])*(position_d[1])*(1-position_d[2])
                    data_anat_smoothed[ceil_position[0],ceil_position[1],floor_position[2]]=data_anat_smoothed[ceil_position[0],ceil_position[1],floor_position[2]]+(weight*result)
                    data_weight[ceil_position[0],ceil_position[1],floor_position[2]]=data_weight[ceil_position[0],ceil_position[1],floor_position[2]]+(weight)

                    if (flag[ceil_position[0],floor_position[1],ceil_position[2]]==0):
                        data_anat_smoothed[ceil_position[0],floor_position[1],ceil_position[2]]=0
                        data_weight[ceil_position[0],floor_position[1],ceil_position[2]]=0
                        flag[ceil_position[0],floor_position[1],ceil_position[2]]=1
                    weight=(position_d[0])*(1-position_d[1])*(position_d[2])
                    data_anat_smoothed[ceil_position[0],floor_position[1],ceil_position[2]]=data_anat_smoothed[ceil_position[0],floor_position[1],ceil_position[2]]+(weight*result)
                    data_weight[ceil_position[0],floor_position[1],ceil_position[2]]=data_weight[ceil_position[0],floor_position[1],ceil_position[2]]+(weight)

                    if (flag[floor_position[0],ceil_position[1],floor_position[2]]==0):
                        data_anat_smoothed[floor_position[0],ceil_position[1],floor_position[2]]=0
                        data_weight[floor_position[0],ceil_position[1],floor_position[2]]=0
                        flag[floor_position[0],ceil_position[1],floor_position[2]]=1
                    weight=(1-position_d[0])*(position_d[1])*(1-position_d[2])
                    data_anat_smoothed[floor_position[0],ceil_position[1],floor_position[2]]=data_anat_smoothed[floor_position[0],ceil_position[1],floor_position[2]]+(weight*result)
                    data_weight[floor_position[0],ceil_position[1],floor_position[2]]=data_weight[floor_position[0],ceil_position[1],floor_position[2]]+(weight)

                    if (flag[floor_position[0],ceil_position[1],ceil_position[2]]==0):
                        data_anat_smoothed[floor_position[0],ceil_position[1],ceil_position[2]]=0
                        data_weight[floor_position[0],ceil_position[1],ceil_position[2]]=0
                        flag[floor_position[0],ceil_position[1],ceil_position[2]]=1
                    weight=(1-position_d[0])*(position_d[1])*(position_d[2])
                    data_anat_smoothed[floor_position[0],ceil_position[1], ceil_position[2]]= data_anat_smoothed[floor_position[0],ceil_position[1], ceil_position[2]]+(weight*result)
                    data_weight[floor_position[0],ceil_position[1], ceil_position[2]]= data_weight[floor_position[0],ceil_position[1], ceil_position[2]]+(weight)

                    if (flag[floor_position[0],floor_position[1],ceil_position[2]]==0):
                        data_anat_smoothed[floor_position[0],floor_position[1],ceil_position[2]]=0
                        flag[floor_position[0],floor_position[1],ceil_position[2]]=1
                        data_weight[floor_position[0],floor_position[1],ceil_position[2]]=0
                    weight=(1-position_d[0])*(1-position_d[1])*(position_d[2])
                    data_anat_smoothed[floor_position[0],floor_position[1],ceil_position[2]]=data_anat_smoothed[floor_position[0],floor_position[1],ceil_position[2]]+(weight*result)
                    data_weight[floor_position[0],floor_position[1],ceil_position[2]]=data_weight[floor_position[0],floor_position[1],ceil_position[2]]+(weight)


    # Once we covered the whole spinal cord along z, we normalize the resulting image considering the weight used to calculate each voxel intensity
    data_anat_smoothed=data_anat_smoothed/data_weight



    #Generate output file
    #==========================================================================================

    # Write NIFTI volumes
    print '\nWrite NIFTI volumes...'
    if os.path.isfile('tmp.im_smoothed.nii'):
        sct.run('rm tmp.im_smoothed.nii')
    img = nibabel.Nifti1Image(data_anat_smoothed, None)
    nibabel.save(img, 'tmp.im_smoothed.nii')
    print '.. File created: tmp.im_smoothed.nii'

    #Copy header geometry from input data
    print '\nCopy header geometry from input data and reorient the volume...'
    sct.run(sct.fsloutput+'fslcpgeom tmp.anat_orient.nii tmp.im_smoothed.nii ')

    #Generate output file
    print '\nGenerate output file (in current folder)...'
    sct.generate_output_file('tmp.im_smoothed.nii','./',file_anat+'_smoothed',ext_anat)

    # Delete temporary files
    if remove_temp_files == 1:
        print '\nDelete temporary files...'
        sct.run('rm tmp.anat.nii')
        sct.run('rm tmp.centerline.nii')
        sct.run('rm tmp.anat_orient.nii')
        sct.run('rm tmp.centerline_orient.nii')


    #Display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished!'
    print '.. '+str(int(round(elapsed_time)))+'s\n'


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
          'sct_smooth_spinal_cord\n' \
          '--------------------------------------------------------------------------------------------------------------\n' \
          'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
          '\n' \
          'DESCRIPTION\n' \
          '  This program smooth an anatomical image along  the spinal cord. It works by applying a convolution of a 1D' \
          'kernel with the anatomical IRM image along the spinal cord. The output is is the anatomical image smoothed ' \
          'and oriented as the FSL standard.' \
          '\n' \
          'USAGE\n' \
          '  sct_smooth_spinal_cord.py -i <anat> -c <centerline>\n' \
          '\n' \
          'MANDATORY ARGUMENTS\n' \
          '  -i <anat>         anatomic nifti file. Image to smooth.\n' \
          '  -c <centerline>   equation of the centerline (given by the function sct_get_centerline.py).\n' \
          '\n' \
          'OPTIONAL ARGUMENTS\n' \
          '  -f <fwhm>         full width at half maximum (in mm). Default=' + str(
        param.fwhm) + 'mm.\n' \
          '  -w                width of a square window within which the smoothing will occur (in mm). Smaller is faster. Default=' + str(param.width) + 'mm. \n' \
          '  -r <0,1>          remove temporary files. Default=' + str(param.remove_temp_files) + '. \n' \
          '  -h                help. Show this message.\n'

    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
