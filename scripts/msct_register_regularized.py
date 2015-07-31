#!/usr/bin/env python
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Tanguy Magnan
# Modified: 2015-07-29
#
# License: see the LICENSE.TXT
#=======================================================================================================================
#


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



def register_seg(seg_input, seg_dest):
    """Slice-by-slice registration by translation of two segmentations.

    For each slice, we estimate the translation vector by calculating the difference of position of the two centers of
    mass.
    The segmentations can be of different sizes but the output segmentation must be smaller than the input segmentation.

    input:
        seg_input: name of moving segmentation file (type: string)
        seg_dest: name of fixed segmentation file (type: string)

    output:
        x_displacement: list of translation along x axis for each slice (type: list)
        y_displacement: list of translation along y axis for each slice (type: list)

    """
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


def register_images(im_input, im_dest, mask='', paramreg=Paramreg(step='0', type='im', algo='Translation', metric='MI', iter='5', shrink='1', smooth='0', gradStep='0.5'),
                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}, remove_tmp_folder = 1):
    """Slice-by-slice registration of two images.

    We first split the 3D images into 2D images (and the mask if inputted). Then we register slices of the two images
    that physically correspond to one another looking at the physical origin of each image. The images can be of
    different sizes but the destination image must be smaller thant the input image. We do that using antsRegistration
    in 2D. Once this has been done for each slices, we gather the results and return them.
    Algorithms implemented: translation, rigid, affine, syn and BsplineSyn.
    N.B.: If the mask is inputted, it must also be 3D and it must be in the same space as the destination image.

    input:
        im_input: name of moving image (type: string)
        im_dest: name of fixed image (type: string)
        mask[optional]: name of mask file (type: string) (parameter -x of antsRegistration)
        paramreg[optional]: parameters of antsRegistration (type: Paramreg class from sct_register_multimodal)
        ants_registration_params[optional]: specific algorithm's parameters for antsRegistration (type: dictionary)

    output:
        if algo==translation:
            x_displacement: list of translation along x axis for each slice (type: list)
            y_displacement: list of translation along y axis for each slice (type: list)
        if algo==rigid:
            x_displacement: list of translation along x axis for each slice (type: list)
            y_displacement: list of translation along y axis for each slice (type: list)
            theta_rotation: list of rotation angle in radian (and in ITK's coordinate system) for each slice (type: list)
        if algo==affine or algo==syn or algo==bsplinesyn:
            creation of two 3D warping fields (forward and inverse) that are the concatenations of the slice-by-slice
            warps.
    """
    # Extracting names
    path_i, root_i, ext_i = sct.extract_fname(im_input)
    path_d, root_d, ext_d = sct.extract_fname(im_dest)

    # set metricSize
    if paramreg.metric == 'MI':
        metricSize = '32'  # corresponds to number of bins
    else:
        metricSize = '4'  # corresponds to radius (for CC, MeanSquares...)


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

    # Split destination volume along z
    print '\nSplit destination volume...'
    sct.run(sct.fsloutput + 'fslsplit '+im_dest+' '+root_d+'_z -z')

    # Split mask volume along z
    if mask:
        print '\nSplit mask volume...'
        sct.run(sct.fsloutput + 'fslsplit mask.nii.gz mask_z -z')

    im_dest_img = Image(im_dest)
    im_input_img = Image(im_input)
    coord_origin_dest = im_dest_img.transfo_pix2phys([[0,0,0]])
    coord_origin_input = im_input_img.transfo_pix2phys([[0,0,0]])
    coord_diff_origin = (asarray(coord_origin_dest[0]) - asarray(coord_origin_input[0])).tolist()
    [x_o, y_o, z_o] = [coord_diff_origin[0] * 1.0/px, coord_diff_origin[1] * 1.0/py, coord_diff_origin[2] * 1.0/pz]

    if paramreg.algo == 'BSplineSyN' or paramreg.algo == 'SyN' or paramreg.algo == 'Affine':
        list_warp_x = []
        list_warp_x_inv = []
        list_warp_y = []
        list_warp_y_inv = []
        name_warp_final = 'Warp_total' #if modified, name should also be modified in msct_register (algo slicereg2d_bsplinesyn and slicereg2d_syn)

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
               '--transform '+paramreg.algo+'['+str(paramreg.gradStep) +
               ants_registration_params[paramreg.algo.lower()]+'] '
               '--metric '+paramreg.metric+'['+root_d+'_z'+ num +'.nii' +','+root_i+'_z'+ num_2 +'.nii' +',1,'+metricSize+'] '  #[fixedImage,movingImage,metricWeight +nb_of_bins (MI) or radius (other)
               '--convergence '+str(paramreg.iter)+' '
               '--shrink-factors '+str(paramreg.shrink)+' '
               '--smoothing-sigmas '+str(paramreg.smooth)+'mm '
               #'--restrict-deformation 1x1x0 '    # how to restrict? should not restrict here, if transform is precised...?
               '--output [transform_' + num + ','+root_i+'_z'+ num_2 +'reg.nii] '    #--> file.mat (contains Tx,Ty, theta)
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
                # New process added for generating total nifti warping field from mat warp
                name_dest = root_d+'_z'+ num +'.nii'
                name_reg = root_i+'_z'+ num +'reg.nii'
                name_output_warp = 'warp_from_mat_' + num_2 + '.nii.gz'
                name_output_warp_inverse = 'warp_from_mat_' + num + '_inverse.nii.gz'
                name_warp_null = 'warp_null_' + num + '.nii.gz'
                name_warp_null_dest = 'warp_null_dest' + num + '.nii.gz'
                name_warp_mat = 'transform_' + num + '0GenericAffine.mat'
                # Generating null nifti warping fields
                nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(name_reg)
                nx_d, ny_d, nz_d, nt_d, px_d, py_d, pz_d, pt_d = sct.get_dimension(name_dest)
                x_trans = [0 for i in range(nz)]
                x_trans_d = [0 for i in range(nz_d)]
                y_trans= [0 for i in range(nz)]
                y_trans_d = [0 for i in range(nz_d)]
                generate_warping_field(name_reg, x_trans=x_trans, y_trans=y_trans, fname=name_warp_null, verbose=0)
                generate_warping_field(name_dest, x_trans=x_trans_d, y_trans=y_trans_d, fname=name_warp_null_dest, verbose=0)
                # Concatenating mat wrp and null nifti warp to obtain equivalent nifti warp to mat warp
                sct.run('isct_ComposeMultiTransform 2 ' + name_output_warp + ' -R ' + name_reg + ' ' + name_warp_null + ' ' + name_warp_mat)
                sct.run('isct_ComposeMultiTransform 2 ' + name_output_warp_inverse + ' -R ' + name_dest + ' ' + name_warp_null_dest + ' -i ' + name_warp_mat)
                # Split the warping fields into two for displacement along x and y before merge
                sct.run('isct_c3d -mcs ' + name_output_warp + ' -oo transform_'+num+'0Warp_x.nii.gz transform_'+num+'0Warp_y.nii.gz')
                sct.run('isct_c3d -mcs ' + name_output_warp_inverse + ' -oo transform_'+num+'0InverseWarp_x.nii.gz transform_'+num+'0InverseWarp_y.nii.gz')
                # List names of warping fields for futur merge
                list_warp_x.append('transform_'+num+'0Warp_x.nii.gz')
                list_warp_x_inv.append('transform_'+num+'0InverseWarp_x.nii.gz')
                list_warp_y.append('transform_'+num+'0Warp_y.nii.gz')
                list_warp_y_inv.append('transform_'+num+'0InverseWarp_y.nii.gz')

            if paramreg.algo == 'BSplineSyN' or paramreg.algo == 'SyN':
                # Split the warping fields into two for displacement along x and y before merge
                # Need to separate the merge for x and y displacement as merge of 3d warping fields does not work properly
                sct.run('isct_c3d -mcs transform_'+num+'0Warp.nii.gz -oo transform_'+num+'0Warp_x.nii.gz transform_'+num+'0Warp_y.nii.gz')
                sct.run('isct_c3d -mcs transform_'+num+'0InverseWarp.nii.gz -oo transform_'+num+'0InverseWarp_x.nii.gz transform_'+num+'0InverseWarp_y.nii.gz')
                # List names of warping fields for futur merge
                list_warp_x.append('transform_'+num+'0Warp_x.nii.gz')
                list_warp_x_inv.append('transform_'+num+'0InverseWarp_x.nii.gz')
                list_warp_y.append('transform_'+num+'0Warp_y.nii.gz')
                list_warp_y_inv.append('transform_'+num+'0InverseWarp_y.nii.gz')
        # if an exception occurs with ants, take the last value for the transformation
        except:
                if paramreg.algo == 'Rigid' or paramreg.algo == 'Translation':
                    x_displacement[i] = x_displacement[i-1]
                    y_displacement[i] = y_displacement[i-1]
                    theta_rotation[i] = theta_rotation[i-1]


                if paramreg.algo == 'BSplineSyN' or paramreg.algo == 'SyN' or paramreg.algo == 'Affine':
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

    if paramreg.algo == 'BSplineSyN' or paramreg.algo == 'SyN' or paramreg.algo == 'Affine':
        print'\nMerge along z of the warping fields...'
        sct.run('fslmerge -z ' + name_warp_final + '_x ' + " ".join(list_warp_x))
        sct.run('fslmerge -z ' + name_warp_final + '_x_inverse ' + " ".join(list_warp_x_inv))
        sct.run('fslmerge -z ' + name_warp_final + '_y ' + " ".join(list_warp_y))
        sct.run('fslmerge -z ' + name_warp_final + '_y_inverse ' + " ".join(list_warp_y_inv))
        print'\nChange resolution of warping fields to match the resolution of the destination image...'
        sct.run('fslcpgeom ' + im_dest + ' ' + name_warp_final + '_x.nii.gz')
        sct.run('fslcpgeom ' + im_input + ' ' + name_warp_final + '_x_inverse.nii.gz')
        sct.run('fslcpgeom ' + im_dest + ' ' + name_warp_final + '_y.nii.gz')
        sct.run('fslcpgeom ' + im_input + ' ' + name_warp_final + '_y_inverse.nii.gz')
        print'\nMerge translation fields along x and y into one global warping field '
        sct.run('isct_c3d ' + name_warp_final + '_x.nii.gz ' + name_warp_final + '_y.nii.gz -omc 2 ' + name_warp_final + '.nii.gz')
        sct.run('isct_c3d ' + name_warp_final + '_x_inverse.nii.gz ' + name_warp_final + '_y_inverse.nii.gz -omc 2 ' + name_warp_final + '_inverse.nii.gz')
        print'\nCopy to parent folder...'
        sct.run('cp ' + name_warp_final + '.nii.gz ../')
        sct.run('cp ' + name_warp_final + '_inverse.nii.gz ../')




    #Delete tmp folder
    os.chdir('../')
    if remove_tmp_folder:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)
    if paramreg.algo == 'Rigid':
        return x_displacement, y_displacement, theta_rotation
    if paramreg.algo == 'Translation':
        return x_displacement, y_displacement



def numerotation(nb):
    """Indexation of number for matching fslsplit's index.

    Given a slice number, this function returns the corresponding number in fslsplit indexation system.

    input:
        nb: the number of the slice (type: int)

    output:
        nb_output: the number of the slice for fslsplit (type: string)
    """
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


def generate_warping_field(im_dest, x_trans, y_trans, theta_rot=None, center_rotation=None, fname='warping_field.nii.gz', verbose=1):
    """Generation of a warping field towards an image and given transformation parameters.

    Given a destination image and transformation parameters this functions creates a NIFTI 3D warping field that can be
    applied afterwards. The transformation parameters corresponds to a slice-by-slice registration of images, thus the
    transformation parameters must be precised for each slice of the image.

    inputs:
        im_dest: name of destination image (type: string)
        x_trans: list of translations along x axis for each slice (type: list, length: height of im_dest)
        y_trans: list of translations along y axis for each slice (type: list, length: height of im_dest)
        theta_rot[optional]: list of rotation angles in radian (and in ITK's coordinate system) for each slice (type: list)
        center_rotation[optional]: pixel coordinates in plan xOy of the wanted center of rotation (type: list,
            length: 2, example: [0,ny/2])
        fname[optional]: name of output warp (type: string)
        verbose: display parameter (type: int)

    output:
        creation of a warping field of name 'fname' with an header similar to the destination image.
    """
    from nibabel import load
    from math import cos, sin
    from sct_orientation import get_orientation

    #Make sure image is in rpi format
    sct.printv('\nChecking if the image of destination is in RPI orientation for the warping field generation ...', verbose)
    orientation = get_orientation(im_dest)
    if orientation != 'RPI':
        sct.printv('\nWARNING: The image of destination is not in RPI format. Dimensions of the warping field might be inverted.', verbose)
    else: sct.printv('\tOK', verbose)

    sct.printv('\n\nCreating warping field ' + fname + ' for transformations along z...', verbose)

    file_dest = load(im_dest)
    hdr_file_dest = file_dest.get_header()
    hdr_warp = hdr_file_dest.copy()

    # Get image dimensions
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(im_dest)
    sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
    sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)

    #Center of rotation
    if center_rotation == None:
        x_a = int(round(nx/2))
        y_a = int(round(ny/2))
    else:
        x_a = center_rotation[0]
        y_a = center_rotation[1]

    # Calculate displacement for each voxel
    data_warp = zeros(((((nx, ny, nz, 1, 3)))))
    # For translations
    if theta_rot == None:
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

    # Generate warp file as a warping field
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = nibabel.Nifti1Image(data_warp, None, hdr_warp)
    nibabel.save(img, fname)
    sct.printv('\nDONE ! Warping field generated: '+fname, verbose)

