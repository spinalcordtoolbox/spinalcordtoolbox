#!/usr/bin/env python

# INPUT:
#   volume_src
#   volume_dest
#   paramreg (class inherited in sct_register_multimodal)
#   (mask)
#   fname_warp
# OUTPUT:
#   none
#   (write warping field)




import os, sys, commands

# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')


import sct_utils as sct
import nibabel
from sct_orientation import get_orientation, set_orientation
from scipy import ndimage
from scipy.io import loadmat
from numpy import array, asarray, zeros, int8, mean, std, sqrt, convolve, hanning
from copy import copy
from msct_image import Image
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.interpolate import splrep,splev
from sct_register_multimodal import Paramreg
import time
import csv
from math import acos




# FUNCTION
# register_seg
# First step to register a segmentation onto another by calculating precisely the displacement.
# (The segmentations can be of different size but the output segmentation must be smaller thant the input segmentation)?????? necessary or done before????

def register_seg(seg_input, seg_dest):
    seg_input_img = nibabel.load(seg_input)
    seg_dest_img = nibabel.load(seg_dest)
    seg_input_data = seg_input_img.get_data()
    seg_dest_data = seg_dest_img.get_data()


    x_center_of_mass_input = [0 for i in range(seg_input_data.shape[2])]
    y_center_of_mass_input = [0 for i in range(seg_input_data.shape[2])]
    print '\nGet center of mass of the input segmentation for each slice (corresponding to a slice int the output segmentation)...' #different if size of the two seg are different
    #TO DO: select only the slices corresponding to the output segmentation
    for iz in xrange(seg_dest_data.shape[2]):
        x_center_of_mass_input[iz], y_center_of_mass_input[iz] = ndimage.measurements.center_of_mass(array(seg_input_data[:,:,iz]))


    x_center_of_mass_output = [0 for i in range(seg_input_data.shape[2])]
    y_center_of_mass_output = [0 for i in range(seg_input_data.shape[2])]
    print '\nGet center of mass of the output segmentation for each slice ...'
    for iz in xrange(seg_dest_data.shape[2]):
        x_center_of_mass_output[iz], y_center_of_mass_output[iz] = ndimage.measurements.center_of_mass(array(seg_dest_data[:,:,iz]))

    x_displacement = [0 for i in range(seg_input_data.shape[2])]
    y_displacement = [0 for i in range(seg_input_data.shape[2])]
    print '\nGet displacement by voxel...'
    for iz in xrange(seg_dest_data.shape[2]):
        x_displacement[iz] = -(x_center_of_mass_output[iz] - x_center_of_mass_input[iz])    #strangely, this is the inverse of x_displacement when the same equation defines y_displacement
        y_displacement[iz] = y_center_of_mass_output[iz] - y_center_of_mass_input[iz]

    return x_displacement, y_displacement



# FUNCTION
# register_image
# First step to register a 3D image onto another by splitting the images into 2D images and calculating the displacement by splice by image correlation algorithms.
# (The images can be of different size but the output image must be smaller thant the input image)?????? necessary or done before????
# If the mask is inputed, it must also be 3D.

def register_images(im_input, im_dest, mask='', paramreg=Paramreg(step='0', type='im', algo='Translation', metric='MI', iter='5', shrink='1', smooth='0', gradStep='0.5'), remove_tmp_folder = 1):

    path_i, root_i, ext_i = sct.extract_fname(im_input)
    path_d, root_d, ext_d = sct.extract_fname(im_dest)
    path_m, root_m, ext_m = sct.extract_fname(mask)

    # set metricSize
    if paramreg.metric == 'MI':
        metricSize = '32'  # corresponds to number of bins
    else:
        metricSize = '4'  # corresponds to radius (for CC, MeanSquares...)

    # initiate default parameters of antsRegistration transformation
    ants_registration_params = {'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '',
                                'bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',3,32'}

    # Get image dimensions and retrieve nz
    print '\nGet image dimensions of destination image...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(im_dest)
    print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
    print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'

    # Define x and y displacement as list
    x_displacement = [0 for i in range(nz)]
    y_displacement = [0 for i in range(nz)]
    theta_rotation = [0 for i in range(nz)]

    # create temporary folder
    print('\nCreate temporary folder...')
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.create_folder(path_tmp)
    print '\nCopy input data...'
    sct.run('cp '+im_input+ ' ' + path_tmp +'/'+ root_i+ext_i)
    sct.run('cp '+im_dest+ ' ' + path_tmp +'/'+ root_d+ext_d)
    if mask:
        sct.run('cp '+mask+ ' '+path_tmp +'/mask.nii.gz')

    # go to temporary folder
    os.chdir(path_tmp)

    # Split input volume along z
    print '\nSplit input volume...'
    sct.run(sct.fsloutput + 'fslsplit '+im_input+' '+root_i+'_z -z')
    #file_anat_split = ['tmp.anat_orient_z'+str(z).zfill(4) for z in range(0,nz,1)]

    # Split destination volume along z
    print '\nSplit destination volume...'
    sct.run(sct.fsloutput + 'fslsplit '+im_dest+' '+root_d+'_z -z')
    #file_anat_split = ['tmp.anat_orient_z'+str(z).zfill(4) for z in range(0,nz,1)]

    # Split mask volume along z
    if mask:
        print '\nSplit mask volume...'
        sct.run(sct.fsloutput + 'fslsplit mask.nii.gz mask_z -z')
        #file_anat_split = ['tmp.anat_orient_z'+str(z).zfill(4) for z in range(0,nz,1)]

    # loop across slices
    for i in range(nz):
        # set masking
        num = numerotation(i)
        if mask:
            masking = '-x mask_z' +num+ '.nii'
        else:
            masking = ''

        cmd = ('isct_antsRegistration '
               '--dimensionality 2 '
               '--transform '+paramreg.algo+'['+paramreg.gradStep +
               ants_registration_params[paramreg.algo.lower()]+'] '
               '--metric '+paramreg.metric+'['+root_d+'_z'+ num +'.nii' +','+root_i+'_z'+ num +'.nii' +',1,'+metricSize+'] '  #[fixedImage,movingImage,metricWeight +nb_of_bins (MI) or radius (other)
               '--convergence '+paramreg.iter+' '
               '--shrink-factors '+paramreg.shrink+' '
               '--smoothing-sigmas '+paramreg.smooth+'mm '
               #'--restrict-deformation 1x1x0 '    # how to restrict? should not restrict here, if transform is precised...?
               '--output [transform_' + num + '] '    #--> file.txt (contains Tx,Ty)    [outputTransformPrefix,<outputWarpedImage>,<outputInverseWarpedImage>]
               '--interpolation BSpline[3] '
               +masking)

        # sct.run(cmd)

        try:
            sct.run(cmd)

            if paramreg.algo == 'Rigid' or paramreg.algo == 'Translation':
                f = 'transform_' +num+ '0GenericAffine.mat'
                matfile = loadmat(f, struct_as_record=True)
                array_transfo = matfile['AffineTransform_double_2_2']
                x_displacement[i] = -array_transfo[4][0]  #is it? or is it y?
                y_displacement[i] = array_transfo[5][0]
                theta_rotation[i] = acos(array_transfo[0])

        except:
                if paramreg.algo == 'Rigid' or paramreg.algo == 'Translation':
                    x_displacement[i] = x_displacement[i-1]  #is it? or is it y?
                    y_displacement[i] = y_displacement[i-1]
                    theta_rotation[i] = theta_rotation[i-1]

        # # get displacement form this slice and complete x and y displacement lists
        # with open('transform_'+num+'.csv') as f:
        #     reader = csv.reader(f)
        #     count = 0
        #     for line in reader:
        #         count += 1
        #         if count == 2:
        #             x_displacement[i] = line[0]
        #             y_displacement[i] = line[1]
        #             f.close()

        # # get matrix of transfo for a rigid transform   (pb slicereg fait une rotation ie le deplacement n'est pas homogene par slice)
        # # recuperer le deplacement ne donnerait pas une liste mais un warping field: mieux vaut recup la matrice output
        # # pb du smoothing du deplacement par slice !!   on peut smoother les param theta tx ty
        # if paramreg.algo == 'Rigid' or paramreg.algo == 'Translation':
        #     f = 'transform_' +num+ '0GenericAffine.mat'
        #     matfile = loadmat(f, struct_as_record=True)
        #     array_transfo = matfile['AffineTransform_double_2_2']
        #     x_displacement[i] = -array_transfo[4][0]  #is it? or is it y?
        #     y_displacement[i] = array_transfo[5][0]
        #     theta_rotation[i] = acos(array_transfo[0])

        #TO DO: different treatment for other algo


    #Delete tmp folder
    os.chdir('../')
    if remove_tmp_folder:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)
    if paramreg.algo == 'Rigid':
        return x_displacement, y_displacement, theta_rotation   # check if the displacement are not inverted (x_dis = -x_disp...)   theta is in radian
    if paramreg.algo == 'Translation':
        return x_displacement, y_displacement


def numerotation(nb):
    if nb < 0:
        print 'ERROR: the number is negative.'
        sys.exit(status = 2)
    elif -1 < nb < 10:
        nb_output = '000'+str(nb)
    elif 9 < nb < 100:
        nb_output = '00'+str(nb)
    elif 99 < nb < 1000:
        nb_output = '0'+str(nb)
    elif 999 < nb < 10000:
        nb_output = str(nb)
    elif nb > 9999:
        print 'ERROR: the number is superior to 9999.'
        sys.exit(status = 2)
    return nb_output



# FUNCTION REGULARIZE
# input: x_displacement,y_displacement
# output: x_displacement_reg,y_displacement_reg
# use msct_smooth smoothing window




# FUNCTION: generate warping field that is homogeneous by slice   (for rigid transform only for now)
#input:
#   image_destination to recover the header
#   x_trans
#   y_trans
#   theta_rot
#   fname_warp

def generate_warping_field(im_dest, x_trans, y_trans, theta_rot=None, fname = 'warping_field.nii.gz'):
    from nibabel import load
    from math import cos, sin


    #Make sure image is in rpi format

    file_dest = load(im_dest)
    hdr_file_dest = file_dest.get_header()
    hdr_warp = hdr_file_dest.copy()

    # Get image dimensions
    print '\nGet image dimensions of destination image...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(im_dest)
    print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
    print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'

    # Calculate displacement for each voxel
    data_warp = zeros(((((nx, ny, nz, 1, 3)))))
    if theta_rot != None:
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    # matrix_rot = [[cos(theta_rot[k]), -sin(theta_rot[k])], [sin(theta_rot[k], cos(theta_rot[k]]]
                    # data_warp[i,j,k,0,:] = matrix_rot
                    data_warp[i, j, k, 0, 0] = (cos(theta_rot[k])-1) * i - sin(theta_rot[k]) * j + x_trans[k]
                    data_warp[i, j, k, 0, 1] = sin(theta_rot[k]) * i + (cos(theta_rot[k])-1) * j + y_trans[k]
                    data_warp[i, j, k, 0, 2] = 0
    if theta_rot == None:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    data_warp[i, j, k, 0, 0] = x_trans[k]
                    data_warp[i, j, k, 0, 1] = y_trans[k]
                    data_warp[i, j, k, 0, 2] = 0


    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = nibabel.Nifti1Image(data_warp, None, hdr_warp)
    nibabel.save(img, fname)