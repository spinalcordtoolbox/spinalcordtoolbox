#!/usr/bin/env python
#
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Tanguy Magnan
#
# License: see the LICENSE.TXT
#=======================================================================================================================
#
import sys, commands
from os import chdir
import sct_utils as sct
from numpy import array, asarray, zeros, sqrt, dot
from scipy import ndimage
from scipy.io import loadmat
from msct_image import Image
from math import asin, cos, sin
from nibabel import load, Nifti1Image, save
from sct_convert import convert
from sct_register_multimodal import Paramreg


def register_slicewise(fname_src,
                        fname_dest,
                        fname_mask='',
                        warp_forward_out='step0Warp.nii.gz',
                        warp_inverse_out='step0InverseWarp.nii.gz',
                        paramreg=None,
                        ants_registration_params=None,
                        verbose=0):

    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose)
    path_tmp = sct.tmp_create(verbose)

    # copy data to temp folder
    sct.printv('\nCopy input data to temp folder...', verbose)
    convert(fname_src, path_tmp+'src.nii')
    convert(fname_dest, path_tmp+'dest.nii')
    if fname_mask != '':
        convert(fname_mask, path_tmp+'mask.nii.gz')

    # go to temporary folder
    chdir(path_tmp)

    # Calculate displacement
    if paramreg.algo == 'centermass':
        # calculate translation of center of mass between source and destination in voxel space
        register2d_centermass('src.nii', 'dest.nii', fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out, verbose=verbose)
    else:
        # convert SCT flags into ANTs-compatible flags
        algo_dic = {'translation': 'Translation', 'rigid': 'Rigid', 'affine': 'Affine', 'syn': 'SyN', 'bsplinesyn': 'BSplineSyN', 'centermass': 'centermass'}
        paramreg.algo = algo_dic[paramreg.algo]
        # run slicewise registration
        register2d('src.nii', 'dest.nii', fname_mask=fname_mask, fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out, paramreg=paramreg, ants_registration_params=ants_registration_params, verbose=verbose)

    sct.printv('\nMove warping fields to parent folder...', verbose)
    sct.run('mv '+warp_forward_out+' ../')
    sct.run('mv '+warp_inverse_out+' ../')

    # go back to parent folder
    chdir('../')



def register2d_centermass(fname_src, fname_dest, fname_warp='warp_forward.nii.gz', fname_warp_inv='warp_inverse.nii.gz', verbose=1):
    """Slice-by-slice registration by translation of two segmentations.
    For each slice, we estimate the translation vector by calculating the difference of position of the two centers of
    mass in voxel unit.
    The segmentations can be of different sizes but the output segmentation must be smaller than the input segmentation.

    input:
        seg_input: name of moving segmentation file (type: string)
        seg_dest: name of fixed segmentation file (type: string)
    input optional:
        fname_warp: name of output 3d forward warping field
        fname_warp_inv: name of output 3d inverse warping field
        verbose
    output:
        x_displacement: list of translation along x axis for each slice (type: list)
        y_displacement: list of translation along y axis for each slice (type: list)

    """
    seg_input_img = Image('src.nii')
    seg_dest_img = Image('dest.nii')
    seg_input_data = seg_input_img.data
    seg_dest_data = seg_dest_img.data

    x_center_of_mass_input = [0] * seg_dest_data.shape[2]
    y_center_of_mass_input = [0] * seg_dest_data.shape[2]

    sct.printv('\nGet center of mass of source image...', verbose)
    # TODO: select only the slices corresponding to the output segmentation

    # grab physical coordinates of destination origin
    coord_origin_dest = seg_dest_img.transfo_pix2phys([[0, 0, 0]])

    # grab the voxel coordinates of the destination origin from the source image
    [[x_o, y_o, z_o]] = seg_input_img.transfo_phys2pix(coord_origin_dest)

    # calculate center of mass for each slice of the input image
    for iz in xrange(seg_dest_data.shape[2]):
        # starts from z_o, which is the origin of the destination image in the source image
        x_center_of_mass_input[iz], y_center_of_mass_input[iz] = ndimage.measurements.center_of_mass(array(seg_input_data[:, :, z_o + iz]))

    # initialize data
    x_center_of_mass_output = [0] * seg_dest_data.shape[2]
    y_center_of_mass_output = [0] * seg_dest_data.shape[2]

    # calculate center of mass for each slice of the destination image
    sct.printv('\nGet center of mass of destination image...', verbose)
    for iz in xrange(seg_dest_data.shape[2]):
        try:
            x_center_of_mass_output[iz], y_center_of_mass_output[iz] = ndimage.measurements.center_of_mass(array(seg_dest_data[:, :, iz]))
        except Exception as e:
            sct.printv('WARNING: Exception error in msct_register during register_seg:', 1, 'warning')
            print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
            print e

    # calculate displacement in voxel space
    x_displacement = [0] * seg_input_data.shape[2]
    y_displacement = [0] * seg_input_data.shape[2]
    sct.printv('\nGet X-Y displacement for each slice...', verbose)
    for iz in xrange(seg_dest_data.shape[2]):
        x_displacement[iz] = -(x_center_of_mass_output[iz] - x_center_of_mass_input[iz])    # WARNING: in ITK's coordinate system, this is actually Tx and not -Tx
        y_displacement[iz] = y_center_of_mass_output[iz] - y_center_of_mass_input[iz]      # This is Ty in ITK's and fslview' coordinate systems

    # convert to array
    x_disp_a = asarray(x_displacement)
    y_disp_a = asarray(y_displacement)

    # create theta vector (for easier code management)
    theta_rot_a = zeros(seg_dest_data.shape[2])

    # Generate warping field
    generate_warping_field('dest.nii', x_disp_a, y_disp_a, theta_rot_a, fname=fname_warp)  #name_warp= 'step'+str(paramreg.step)
    # Inverse warping field
    generate_warping_field('src.nii', -x_disp_a, -y_disp_a, theta_rot_a, fname=fname_warp_inv)



def register2d(fname_src, fname_dest, fname_mask='', fname_warp='warp_forward.nii.gz', fname_warp_inv='warp_inverse.nii.gz', paramreg=Paramreg(step='0', type='im', algo='Translation', metric='MI', iter='5', shrink='1', smooth='0', gradStep='0.5'),
                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '','bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                              'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}, verbose=0):
    """Slice-by-slice registration of two images.

    We first split the 3D images into 2D images (and the mask if inputted). Then we register slices of the two images
    that physically correspond to one another looking at the physical origin of each image. The images can be of
    different sizes but the destination image must be smaller thant the input image. We do that using antsRegistration
    in 2D. Once this has been done for each slices, we gather the results and return them.
    Algorithms implemented: translation, rigid, affine, syn and BsplineSyn.
    N.B.: If the mask is inputted, it must also be 3D and it must be in the same space as the destination image.

    input:
        fname_source: name of moving image (type: string)
        fname_dest: name of fixed image (type: string)
        mask[optional]: name of mask file (type: string) (parameter -x of antsRegistration)
        fname_warp: name of output 3d forward warping field
        fname_warp_inv: name of output 3d inverse warping field
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

    # set metricSize
    if paramreg.metric == 'MI':
        metricSize = '32'  # corresponds to number of bins
    else:
        metricSize = '4'  # corresponds to radius (for CC, MeanSquares...)

    # Get image dimensions and retrieve nz
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
    sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
    sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)

    # Split input volume along z
    sct.printv('\nSplit input volume...', verbose)
    from sct_image import split_data
    im_src = Image('src.nii')
    split_source_list = split_data(im_src, 2)
    for im in split_source_list:
        im.save()

    # Split destination volume along z
    sct.printv('\nSplit destination volume...', verbose)
    im_dest = Image('dest.nii')
    split_dest_list = split_data(im_dest, 2)
    for im in split_dest_list:
        im.save()

    # Split mask volume along z
    if fname_mask != '':
        sct.printv('\nSplit mask volume...', verbose)
        im_mask = Image('mask.nii.gz')
        split_mask_list = split_data(im_mask, 2)
        for im in split_mask_list:
            im.save()

    # coord_origin_dest = im_dest.transfo_pix2phys([[0,0,0]])
    # coord_origin_input = im_src.transfo_pix2phys([[0,0,0]])
    # coord_diff_origin = (asarray(coord_origin_dest[0]) - asarray(coord_origin_input[0])).tolist()
    # [x_o, y_o, z_o] = [coord_diff_origin[0] * 1.0/px, coord_diff_origin[1] * 1.0/py, coord_diff_origin[2] * 1.0/pz]

    # initialization
    if paramreg.algo in ['Rigid', 'Translation']:
        x_displacement = [0 for i in range(nz)]
        y_displacement = [0 for i in range(nz)]
        theta_rotation = [0 for i in range(nz)]
    if paramreg.algo in ['Affine', 'BSplineSyN', 'SyN']:
        list_warp = []
        list_warp_inv = []

    # loop across slices
    for i in range(nz):
        # set masking
        sct.printv('Registering slice '+str(i)+'/'+str(nz-1)+'...', verbose)
        num = numerotation(i)
        prefix_warp2d = 'warp2d_'+num
        # if mask is used, prepare command for ANTs
        if fname_mask != '':
            masking = '-x mask_Z' +num+ '.nii.gz'
        else:
            masking = ''
        # main command for registration
        cmd = ('isct_antsRegistration '
               '--dimensionality 2 '
               '--transform '+paramreg.algo+'['+str(paramreg.gradStep) +
               ants_registration_params[paramreg.algo.lower()]+'] '
               '--metric '+paramreg.metric+'[dest_Z' + num + '.nii' + ',src_Z' + num + '.nii' +',1,'+metricSize+'] '  #[fixedImage,movingImage,metricWeight +nb_of_bins (MI) or radius (other)
               '--convergence '+str(paramreg.iter)+' '
               '--shrink-factors '+str(paramreg.shrink)+' '
               '--smoothing-sigmas '+str(paramreg.smooth)+'mm '
               '--output ['+prefix_warp2d+',src_Z'+ num +'_reg.nii] '    #--> file.mat (contains Tx,Ty, theta)
               '--interpolation BSpline[3] '
               + masking)
        # add init translation
        if not paramreg.init == '':
            init_dict = {'geometric': '0', 'centermass': '1', 'origin': '2'}
            cmd += ' -r [dest_Z'+num+'.nii'+',src_Z'+num+'.nii,'+init_dict[paramreg.init]+']'

        try:
            # run registration
            sct.run(cmd)

            if paramreg.algo in ['Rigid', 'Translation']:
                file_mat = prefix_warp2d+'0GenericAffine.mat'
                matfile = loadmat(file_mat, struct_as_record=True)
                array_transfo = matfile['AffineTransform_double_2_2']
                x_displacement[i] = array_transfo[4][0]  # Tx in ITK'S coordinate system
                y_displacement[i] = array_transfo[5][0]  # Ty  in ITK'S and fslview's coordinate systems
                theta_rotation[i] = asin(array_transfo[2]) # angle of rotation theta in ITK'S coordinate system (minus theta for fslview)

            if paramreg.algo in ['Affine', 'BSplineSyN', 'SyN']:
                # List names of 2d warping fields for subsequent merge along Z
                file_warp2d = prefix_warp2d+'0Warp.nii.gz'
                file_warp2d_inv = prefix_warp2d+'0InverseWarp.nii.gz'
                list_warp.append(file_warp2d)
                list_warp_inv.append(file_warp2d_inv)

            if paramreg.algo == 'Affine':
                # Generating null 2d warping field (for subsequent concatenation with affine transformation)
                sct.run('isct_antsRegistration -d 2 -t SyN[1, 1, 1] -c 0 -m MI[dest_Z'+num+'.nii, src_Z'+num+'.nii, 1, 32] -o warp2d_null -f 1 -s 0')
                # --> outputs: warp2d_null0Warp.nii.gz, warp2d_null0InverseWarp.nii.gz
                file_mat = prefix_warp2d + '0GenericAffine.mat'
                # Concatenating mat transfo and null 2d warping field to obtain 2d warping field of affine transformation
                sct.run('isct_ComposeMultiTransform 2 ' + file_warp2d + ' -R dest_Z'+num+'.nii warp2d_null0Warp.nii.gz ' + file_mat)
                sct.run('isct_ComposeMultiTransform 2 ' + file_warp2d_inv + ' -R src_Z'+num+'.nii warp2d_null0InverseWarp.nii.gz -i ' + file_mat)

        # if an exception occurs with ants, take the last value for the transformation
        # TODO: DO WE NEED TO DO THAT??? (julien 2016-03-01)
        except Exception, e:
            sct.printv('ERROR: Exception occurred.\n'+str(e), 1, 'error')

    # Merge warping field along z
    sct.printv('\nMerge warping fields along z...', verbose)

    if paramreg.algo in ['Rigid', 'Translation']:
        # convert to array
        x_disp_a = asarray(x_displacement)
        y_disp_a = asarray(y_displacement)
        theta_rot_a = asarray(theta_rotation)
        # Generate warping field
        generate_warping_field('dest.nii', x_disp_a, y_disp_a, theta_rot_a, fname=fname_warp)  #name_warp= 'step'+str(paramreg.step)
        # Inverse warping field
        generate_warping_field('src.nii', -x_disp_a, -y_disp_a, theta_rot_a, fname=fname_warp_inv)

    if paramreg.algo in ['BSplineSyN', 'SyN', 'Affine']:
        from sct_image import concat_warp2d
        # concatenate 2d warping fields along z
        concat_warp2d(list_warp, fname_warp, 'dest.nii')
        concat_warp2d(list_warp_inv, fname_warp_inv, 'src.nii')



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



def generate_warping_field(fname_dest, x_trans, y_trans, theta_rot, center_rotation=None, fname='warping_field.nii.gz', verbose=1):
    """Generation of a warping field towards an image and given transformation parameters.

    Given a destination image and transformation parameters this functions creates a NIFTI 3D warping field that can be
    applied afterwards. The transformation parameters corresponds to a slice-by-slice registration of images, thus the
    transformation parameters must be precised for each slice of the image.

    inputs:
        fname_dest: name of destination image (type: string). NEEDS TO BE RPI ORIENTATION!!!
        x_trans: list of translations along x axis for each slice (type: list, length: height of fname_dest)
        y_trans: list of translations along y axis for each slice (type: list, length: height of fname_dest)
        theta_rot: list of rotation angles in radian (and in ITK's coordinate system) for each slice (type: list)
    inputs (optional):
        center_rotation: pixel coordinates in plan xOy of the wanted center of rotation (type: list, length: 2, example: [0,ny/2])
        fname: name of output warp (type: string)
        verbose: display parameter (type: int)
    output:
        creation of a warping field of name 'fname' with an header similar to the destination image.
    """
    sct.printv('\n\nCreating warping field ' + fname + ' for transformations along z...', verbose)

    file_dest = load(fname_dest)
    hdr_file_dest = file_dest.get_header()
    hdr_warp = hdr_file_dest.copy()

    # Get image dimensions
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
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
    vector_i = [[[i-x_a], [j-y_a]] for i in range(nx) for j in range(ny)]
    for k in range(nz):
        matrix_rot_a = asarray([[cos(theta_rot[k]), - sin(theta_rot[k])], [-sin(theta_rot[k]), -cos(theta_rot[k])]])
        tmp = matrix_rot_a + array(((-1, 0), (0, 1)))
        result = dot(tmp, array(vector_i).T[0]) + array([[x_trans[k]], [y_trans[k]]])
        for i in range(nx):
            data_warp[i, :, k, 0, 0] = result[0][i*nx:i*nx+ny]
            data_warp[i, :, k, 0, 1] = result[1][i*nx:i*nx+ny]

    # Generate warp file as a warping field
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = Nifti1Image(data_warp, None, hdr_warp)
    save(img, fname)
    sct.printv('\nDONE ! Warping field generated: '+fname, verbose)

