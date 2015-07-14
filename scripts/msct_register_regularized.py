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
from scipy import ndimage
from scipy.io import loadmat
from numpy import array, asarray, zeros, sqrt, dot
from msct_image import Image
from sct_register_multimodal import Paramreg
import time
from math import asin




# FUNCTION
# register_seg
# First step to register a segmentation onto another by calculating precisely the displacement.
# (The segmentations can be of different size but the output segmentation must be smaller thant the input segmentation)?????? necessary or done before????

def register_seg(seg_input, seg_dest):
    seg_input_img = Image(seg_input)
    seg_dest_img = Image(seg_dest)
    seg_input_data = seg_input_img.data
    seg_dest_data = seg_dest_img.data


    x_center_of_mass_input = [0 for i in range(seg_dest_data.shape[2])]
    y_center_of_mass_input = [0 for i in range(seg_dest_data.shape[2])]
    print '\nGet center of mass of the input segmentation for each slice (corresponding to a slice in the output segmentation)...' #different if size of the two seg are different
    #TO DO: select only the slices corresponding to the output segmentation
    coord_origin_dest = seg_dest_img.transfo_pix2phys([[0,0,0]])
    [[x_o, y_o, z_o]] = seg_input_img.transfo_phys2pix(coord_origin_dest)
    for iz in xrange(seg_dest_data.shape[2]):
        x_center_of_mass_input[iz], y_center_of_mass_input[iz] = ndimage.measurements.center_of_mass(array(seg_input_data[:, :, z_o + iz]))


    x_center_of_mass_output = [0 for i in range(seg_dest_data.shape[2])]
    y_center_of_mass_output = [0 for i in range(seg_dest_data.shape[2])]
    print '\nGet center of mass of the output segmentation for each slice ...'
    for iz in xrange(seg_dest_data.shape[2]):
        x_center_of_mass_output[iz], y_center_of_mass_output[iz] = ndimage.measurements.center_of_mass(array(seg_dest_data[:,:,iz]))

    x_displacement = [0 for i in range(seg_input_data.shape[2])]
    y_displacement = [0 for i in range(seg_input_data.shape[2])]
    print '\nGet displacement by voxel...'
    for iz in xrange(seg_dest_data.shape[2]):
        x_displacement[iz] = -(x_center_of_mass_output[iz] - x_center_of_mass_input[iz])    # WARNING: in ITK's coordinate system, this is actually Tx and not -Tx
        y_displacement[iz] = y_center_of_mass_output[iz] - y_center_of_mass_input[iz]      # This is Ty in ITK's and fslview' coordinate systems

    return x_displacement, y_displacement



# FUNCTION
# register_image
# First step to register a 3D image onto another by splitting the images into 2D images and calculating the displacement by splice by image correlation algorithms.
# (The images can be of different size but the output image must be smaller thant the input image)?????? necessary or done before????
# If the mask is inputed, it must also be 3D and it must be in the same space as the destination image.

def register_images(im_input, im_dest, mask='', paramreg=Paramreg(step='0', type='im', algo='Translation', metric='MI', iter='5', shrink='1', smooth='0', gradStep='0.5'),
                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}, remove_tmp_folder = 1):

    path_i, root_i, ext_i = sct.extract_fname(im_input)
    path_d, root_d, ext_d = sct.extract_fname(im_dest)

    # set metricSize
    if paramreg.metric == 'MI':
        metricSize = '32'  # corresponds to number of bins
    else:
        metricSize = '4'  # corresponds to radius (for CC, MeanSquares...)

    # # initiate default parameters of antsRegistration transformation
    # ants_registration_params = {'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '',
    #                             'bspline': ',10', 'gaussiandisplacementfield': ',3,0',
    #                             'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}

    # Get image dimensions and retrieve nz
    print '\nGet image dimensions of destination image...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(im_dest)
    print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
    print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'

    # Define x and y displacement as list
    x_displacement = [0 for i in range(nz)]
    y_displacement = [0 for i in range(nz)]
    theta_rotation = [0 for i in range(nz)]
    matrix_def = [0 for i in range(nz)]

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

    im_dest_img = Image(im_dest)
    im_input_img = Image(im_input)
    coord_origin_dest = im_dest_img.transfo_pix2phys([[0,0,0]])
    coord_origin_input = im_input_img.transfo_pix2phys([[0,0,0]])
    coord_diff_origin = (asarray(coord_origin_dest[0]) - asarray(coord_origin_input[0])).tolist()
    [x_o, y_o, z_o] = [coord_diff_origin[0] * 1.0/px, coord_diff_origin[1] * 1.0/py, coord_diff_origin[2] * 1.0/pz]

    if paramreg.algo == 'BSplineSyN' or paramreg.algo == 'SyN':
            list_warp_x = []
            list_warp_x_inv = []
            list_warp_y = []
            list_warp_y_inv = []
            name_warp_syn = 'Warp_total' #if modified, name should also be modified in msct_register (algo slicereg2d_bsplinesyn and slicereg2d_syn)
    # loop across slices
    for i in range(nz):
        # set masking
        num = numerotation(i)
        num_2 = numerotation(int(num) + int(z_o))
        if mask:
            masking = '-x mask_z' +num+ '.nii'
        else:
            masking = ''

        cmd = ('isct_antsRegistration '
               '--dimensionality 2 '
               '--transform '+paramreg.algo+'['+paramreg.gradStep +
               ants_registration_params[paramreg.algo.lower()]+'] '
               '--metric '+paramreg.metric+'['+root_d+'_z'+ num +'.nii' +','+root_i+'_z'+ num_2 +'.nii' +',1,'+metricSize+'] '  #[fixedImage,movingImage,metricWeight +nb_of_bins (MI) or radius (other)
               '--convergence '+paramreg.iter+' '
               '--shrink-factors '+paramreg.shrink+' '
               '--smoothing-sigmas '+paramreg.smooth+'mm '
               #'--restrict-deformation 1x1x0 '    # how to restrict? should not restrict here, if transform is precised...?
               '--output [transform_' + num + '] '    #--> file.mat (contains Tx,Ty, theta)
               '--interpolation BSpline[3] '
               +masking)

        try:
            sct.run(cmd)

            if paramreg.algo == 'Rigid' or paramreg.algo == 'Translation':
                f = 'transform_' +num+ '0GenericAffine.mat'
                matfile = loadmat(f, struct_as_record=True)
                array_transfo = matfile['AffineTransform_double_2_2']
                x_displacement[i] = array_transfo[4][0]  # Tx in ITK'S coordinate system
                y_displacement[i] = array_transfo[5][0]  # Ty  in ITK'S and fslview's coordinate systems
                theta_rotation[i] = asin(array_transfo[2]) # angle of rotation theta in ITK'S coordinate system (minus theta for fslview)

            if paramreg.algo == 'Affine':
                f = 'transform_' +num+ '0GenericAffine.mat'
                matfile = loadmat(f, struct_as_record=True)
                array_transfo = matfile['AffineTransform_double_2_2']
                x_displacement[i] = array_transfo[4][0] # Tx in ITK'S coordinate system -Tx in fslview coordinate system
                y_displacement[i] = array_transfo[5][0] # Ty  in ITK'S and fslview's coordinate systems
                matrix_def[i] = [[array_transfo[0][0], array_transfo[1][0]], [array_transfo[2][0], array_transfo[3][0]]]  # array_transfo[2][0] refers to lambda * sin(theta) in ITK'S coordinate system (lambda * (minus theta) for fslview)

            if paramreg.algo == 'BSplineSyN' or paramreg.algo == 'SyN':
                # Split the warping fields into two for displacement along x and y before merge
                sct.run('isct_c3d -mcs transform_'+num+'0Warp.nii.gz -oo transform_'+num+'0Warp_x.nii.gz transform_'+num+'0Warp_y.nii.gz')
                sct.run('isct_c3d -mcs transform_'+num+'0InverseWarp.nii.gz -oo transform_'+num+'0InverseWarp_x.nii.gz transform_'+num+'0InverseWarp_y.nii.gz')
                # List names of warping fields for futur merge
                list_warp_x.append('transform_'+num+'0Warp_x.nii.gz')
                list_warp_x_inv.append('transform_'+num+'0InverseWarp_x.nii.gz')
                list_warp_y.append('transform_'+num+'0Warp_y.nii.gz')
                list_warp_y_inv.append('transform_'+num+'0InverseWarp_y.nii.gz')
        #if an exception occurs with ants, take the last values for the transformation
        except:
                if paramreg.algo == 'Rigid' or paramreg.algo == 'Translation':
                    x_displacement[i] = x_displacement[i-1]
                    y_displacement[i] = y_displacement[i-1]
                    theta_rotation[i] = theta_rotation[i-1]
                if paramreg.algo == 'Affine':
                    x_displacement[i] = x_displacement[i-1]
                    y_displacement[i] = y_displacement[i-1]
                    matrix_def[i] = matrix_def[i-1]
                if paramreg.algo == 'BSplineSyN' or paramreg.algo == 'SyN':
                    print'Problem with ants for slice '+str(i)+'. Copy of the last warping field.'
                    sct.run('cp transform_' + numerotation(i-1) + '0Warp.nii.gz transform_' + num + '0Warp.nii.gz')
                    sct.run('cp transform_' + numerotation(i-1) + '0InverseWarp.nii.gz transform_' + num + '0InverseWarp.nii.gz')
                    # Split the warping fields into two for displacement along x and y before merge
                    sct.run('isct_c3d -mcs transform_'+num+'0Warp.nii.gz -oo transform_'+num+'0Warp_x.nii.gz transform_'+num+'0Warp_y.nii.gz')
                    sct.run('isct_c3d -mcs transform_'+num+'0InverseWarp.nii.gz -oo transform_'+num+'0InverseWarp_x.nii.gz transform_'+num+'0InverseWarp_y.nii.gz')
                    # List names of warping fields for futur merge
                    list_warp_x.append('transform_'+num+'0Warp_x.nii.gz')
                    list_warp_x_inv.append('transform_'+num+'0InverseWarp_x.nii.gz')
                    list_warp_y.append('transform_'+num+'0Warp_y.nii.gz')
                    list_warp_y_inv.append('transform_'+num+'0InverseWarp_y.nii.gz')

        #TO DO: different treatment for other algo

    if paramreg.algo == 'BSplineSyN' or paramreg.algo == 'SyN':
        print'\nMerge along z of the warping fields...' #need to separate the merge for x and y displacement
        sct.run('fslmerge -z ' + name_warp_syn + '_x ' + " ".join(list_warp_x))
        sct.run('fslmerge -z ' + name_warp_syn + '_x_inverse ' + " ".join(list_warp_x_inv))
        sct.run('fslmerge -z ' + name_warp_syn + '_y ' + " ".join(list_warp_y))
        sct.run('fslmerge -z ' + name_warp_syn + '_y_inverse ' + " ".join(list_warp_y_inv))
        print'\nChange resolution of warping fields to match the resolution of the destination image...'
        sct.run('fslcpgeom ' + im_dest + ' ' + name_warp_syn + '_x.nii.gz')
        sct.run('fslcpgeom ' + im_input + ' ' + name_warp_syn + '_x_inverse.nii.gz')
        sct.run('fslcpgeom ' + im_dest + ' ' + name_warp_syn + '_y.nii.gz')
        sct.run('fslcpgeom ' + im_input + ' ' + name_warp_syn + '_y_inverse.nii.gz')
        print'\nMerge translation fields along x and y into one global warping field '
        sct.run('isct_c3d ' + name_warp_syn + '_x.nii.gz ' + name_warp_syn + '_y.nii.gz -omc 2 ' + name_warp_syn + '.nii.gz')
        sct.run('isct_c3d ' + name_warp_syn + '_x_inverse.nii.gz ' + name_warp_syn + '_y_inverse.nii.gz -omc 2 ' + name_warp_syn + '_inverse.nii.gz')
        print'\nCopy to parent folder...'
        sct.run('cp ' + name_warp_syn + '.nii.gz ../')
        sct.run('cp ' + name_warp_syn + '_inverse.nii.gz ../')

    #Delete tmp folder
    os.chdir('../')
    if remove_tmp_folder:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)
    if paramreg.algo == 'Rigid':
        return x_displacement, y_displacement, theta_rotation   # check if the displacement are not inverted (x_dis = -x_disp...)   theta is in radian
    if paramreg.algo == 'Translation':
        return x_displacement, y_displacement
    if paramreg.algo == 'Affine':
        return x_displacement, y_displacement,matrix_def


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




# FUNCTION: generate warping field that is homogeneous by slice
#input:
#   image_destination to recover the header
#   x_trans in ITK'S coordinate system -x_trans in fslview coordinate system
#   y_trans in ITK'S and fslview's coordinate systems
#   theta_rot in ITK'S coordinate system (minus theta for fslview)
#   fname_warp
#   center_rotation if different from the middle of the image (in fslview coordinate system)
#   matrix_def for affine transformation

def generate_warping_field(im_dest, x_trans, y_trans, theta_rot=None, center_rotation = None, matrix_def=None, fname = 'warping_field.nii.gz'):
    from nibabel import load
    from math import cos, sin
    from sct_orientation import get_orientation
    from numpy import matrix

    #Make sure image is in rpi format
    print '\nChecking if the image of destination is in RPI orientation for the warping field generation ...'
    orientation = get_orientation(im_dest)
    if orientation != 'RPI':
        print '\nWARNING: The image of destination is not in RPI format. Dimensions of the warping field might be inverted.'
    else: print'\tOK'

    print'\n\nCreating warping field ' + fname + ' for transformations along z...'

    file_dest = load(im_dest)
    hdr_file_dest = file_dest.get_header()
    hdr_warp = hdr_file_dest.copy()

    # Get image dimensions
    print '\nGet image dimensions of destination image...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(im_dest)
    print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
    print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'

    #Center of rotation
    if center_rotation == None:
        x_a = int(round(nx/2)) # int(round(nx/2))-200
        y_a = int(round(ny/2)) # 0
    else:
        x_a = center_rotation[0]
        y_a = center_rotation[1]

    # Calculate displacement for each voxel
    data_warp = zeros(((((nx, ny, nz, 1, 3)))))
    # For translations
    if theta_rot == None and matrix_def == None:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    data_warp[i, j, k, 0, 0] = x_trans[k]
                    data_warp[i, j, k, 0, 1] = y_trans[k]
                    data_warp[i, j, k, 0, 2] = 0
    # # For rigid transforms (not optimized)
    # if theta_rot != None:
    #     for k in range(nz):
    #         for i in range(nx):
    #             for j in range(ny):
    #                 # data_warp[i, j, k, 0, 0] = (cos(theta_rot[k])-1) * (i - x_a) - sin(theta_rot[k]) * (j - y_a) - x_trans[k]
    #                 # data_warp[i, j, k, 0, 1] = -(sin(theta_rot[k]) * (i - x_a) + (cos(theta_rot[k])-1) * (j - y_a)) + y_trans[k]
    #
    #                 data_warp[i, j, k, 0, 0] = (cos(theta_rot[k]) - 1) * (i - x_a) - sin(theta_rot[k]) * (j - y_a) + x_trans[k] #+ sin(theta_rot[k]) * (int(round(nx/2))-x_a)
    #                 data_warp[i, j, k, 0, 1] = - sin(theta_rot[k]) * (i - x_a) - (cos(theta_rot[k]) - 1) * (j - y_a) + y_trans[k] #- sin(theta_rot[k]) * (int(round(nx/2))-x_a)
    #                 data_warp[i, j, k, 0, 2] = 0

    # For rigid transforms with array (time optimization)
    if theta_rot != None:
        vector_i = [[[i-x_a],[j-y_a]] for i in range(nx) for j in range(ny)]
        for k in range(nz):
            matrix_rot_a = asarray([[cos(theta_rot[k]), - sin(theta_rot[k])],[-sin(theta_rot[k]), -cos(theta_rot[k])]])
            tmp = matrix_rot_a + array(((-1,0),(0,1)))
            result = dot(tmp, array(vector_i).T[0]) + array([[x_trans[k]], [y_trans[k]]])
            for i in range(nx):
                data_warp[i, :, k, 0, 0] = result[0][i*nx:i*nx+ny]
                data_warp[i, :, k, 0, 1] = result[1][i*nx:i*nx+ny]
    # # For affine transforms (not optimized)
    # if theta_rot == None and matrix_def != None:
    #     matrix_def_0 = [matrix_def[j][0][0] for j in range(len(matrix_def))]
    #     matrix_def_1 = [matrix_def[j][0][1] for j in range(len(matrix_def))]
    #     matrix_def_2 = [matrix_def[j][1][0] for j in range(len(matrix_def))]
    #     matrix_def_3 = [matrix_def[j][1][1] for j in range(len(matrix_def))]
    #     for i in range(nx):
    #         for j in range(ny):
    #             for k in range(nz):
    #                 data_warp[i, j, k, 0, 0] = (matrix_def_0[k] - 1) * (i - x_a) + matrix_def_1[k] * (j - y_a) + x_trans[k]  # same equation as for rigid
    #                 data_warp[i, j, k, 0, 1] = matrix_def_2[k] * (i - x_a) - (matrix_def_3[k] - 1) * (j - y_a) + y_trans[k] # same equation as for rigid (est ce qu'il y a vraiment un moins devant matrix_def_3?)
    #                 data_warp[i, j, k, 0, 2] = 0

    # Passer en mode matrice   (checker le signe de matrix_def_3)
    if theta_rot == None and matrix_def != None:
        vector_i = [[[i-x_a],[j-y_a]] for i in range(nx) for j in range(ny)]
        #matrix_def_a = asarray(matrix_def)
        matrix_def_a = asarray([[[matrix_def[j][0][0],matrix_def[j][0][1]],[matrix_def[j][1][0],matrix_def[j][1][1]]] for j in range(len(matrix_def))])
        for k in range(nz):
            tmp = matrix_def_a[k] + array(((-1,0),(0,-1)))
            result = dot(tmp, array(vector_i).T[0]) + array([[x_trans[k]], [y_trans[k]]])
            for i in range(nx):
                data_warp[i, :, k, 0, 0] = result[0][i*nx:i*nx+ny]
                data_warp[i, :, k, 0, 1] = result[1][i*nx:i*nx+ny]

    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = nibabel.Nifti1Image(data_warp, None, hdr_warp)
    nibabel.save(img, fname)
    print '\nDONE ! Warping field generated: '+fname
