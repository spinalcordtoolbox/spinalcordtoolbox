#!/usr/bin/env python
#########################################################################################
# Various modules for registration.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Tanguy Magnan, Julien Cohen-Adad
#
# License: see the LICENSE.TXT
#########################################################################################

# TODO: before running the PCA, correct for the "stretch" effect caused by curvature
# TODO: columnwise: check inverse field
# TODO: columnwise: add regularization: should not binarize at 0.5, especially problematic for edge (because division by zero to compute Sx, Sy).
# TODO: remove register2d_centermass and generalize register2d_centermassrot
# TODO: add flag for setting threshold on PCA
# TODO: clean code for generate_warping_field (unify with centermass_rot)

import sys
from math import asin, cos, sin, acos
from os import chdir
import sct_utils as sct
import numpy as np
from scipy import ndimage
from scipy.io import loadmat
from msct_image import Image
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
        # translation of center of mass between source and destination in voxel space
        register2d_centermass('src.nii', 'dest.nii', fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out, verbose=verbose)
    elif paramreg.algo == 'centermassrot':
        # translation of center of mass and rotation based on source and destination first eigenvectors from PCA.
        register2d_centermassrot('src.nii', 'dest.nii', fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out, verbose=verbose)
    elif paramreg.algo == 'columnwise':
        # scaling R-L, then column-wise center of mass alignment and scaling
        register2d_columnwise('src.nii', 'dest.nii', fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out, verbose=verbose)
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
        x_center_of_mass_input[iz], y_center_of_mass_input[iz] = ndimage.measurements.center_of_mass(np.array(seg_input_data[:, :, z_o + iz]))

    # initialize data
    x_center_of_mass_output = [0] * seg_dest_data.shape[2]
    y_center_of_mass_output = [0] * seg_dest_data.shape[2]

    # calculate center of mass for each slice of the destination image
    sct.printv('\nGet center of mass of destination image...', verbose)
    for iz in xrange(seg_dest_data.shape[2]):
        try:
            x_center_of_mass_output[iz], y_center_of_mass_output[iz] = ndimage.measurements.center_of_mass(np.array(seg_dest_data[:, :, iz]))
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
    x_disp_a = np.asarray(x_displacement)
    y_disp_a = np.asarray(y_displacement)

    # create theta vector (for easier code management)
    theta_rot_a = np.zeros(seg_dest_data.shape[2])

    # Generate warping field
    generate_warping_field('dest.nii', x_disp_a, y_disp_a, theta_rot_a, fname=fname_warp)  #name_warp= 'step'+str(paramreg.step)
    # Inverse warping field
    generate_warping_field('src.nii', -x_disp_a, -y_disp_a, theta_rot_a, fname=fname_warp_inv)



def register2d_centermassrot(fname_src, fname_dest, fname_warp='warp_forward.nii.gz', fname_warp_inv='warp_inverse.nii.gz', verbose=0):
    """Rotate the source image to match the orientation of the destination image, using the first and second eigenvector
    of the PCA. This function should be used on segmentations (not images).

    This works for 2D and 3D images.  If 3D, it splits the image and performs the rotation slice-by-slice.

    input:
        fname_source: name of moving image (type: string)
        fname_dest: name of fixed image (type: string)
        fname_warp: name of output 3d forward warping field
        fname_warp_inv: name of output 3d inverse warping field
        mode:
    output:
        none
    """

    # Get image dimensions and retrieve nz
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
    sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
    sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)

    # Split source volume along z
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

    # display image
    data_src = im_src.data
    data_dest = im_dest.data

    # initialize forward warping field (defined in destination space)
    warp_x = np.zeros(data_dest.shape)
    warp_y = np.zeros(data_dest.shape)

    # initialize inverse warping field (defined in source space)
    warp_inv_x = np.zeros(data_src.shape)
    warp_inv_y = np.zeros(data_src.shape)

    # Loop across slices
    for iz in range(0, nz):
        # initialize src/dest angle at zero if exception is raised
        angle_src_dest = 0

        # iz = 0
        import matplotlib.pyplot as plt
        # plt.figure(figsize=(15, 4))
        # plt.subplot(121)
        # plt.imshow(data_src[:, :, iz], cmap=plt.cm.gray)
        # plt.title('src')
        # plt.subplot(122)
        # plt.imshow(data_dest[:, :, iz], cmap=plt.cm.gray)
        # plt.title('dest')
        # plt.show()
        # coord_src = array
        # coord_dest = array
        try:
            coord_src, pca_src, centermass_src = compute_pca(data_src[:, :, iz])
            coord_dest, pca_dest, centermass_dest = compute_pca(data_dest[:, :, iz])
            # compute (src,dest) angle for first eigenvector
            eigenv_src = pca_src.components_.T[0][0], pca_src.components_.T[1][0]  # pca_src.components_.T[0]
            eigenv_dest = pca_dest.components_.T[0][0], pca_dest.components_.T[1][0]  # pca_dest.components_.T[0]
            angle_src_dest = angle_between(eigenv_src, eigenv_dest)

            # import numpy as np
            R = np.matrix( ((cos(angle_src_dest), sin(angle_src_dest)), (-sin(angle_src_dest), cos(angle_src_dest))) )

            # display rotations
            if verbose == 2 and not angle_src_dest == 0:
                # compute new coordinates
                coord_src_rot = coord_src * R
                coord_dest_rot = coord_dest * R.T
                # generate figure
                import matplotlib.pyplot as plt
                plt.figure('iz='+str(iz)+', angle_src_dest='+str(angle_src_dest), figsize=(9, 9))
                plt.ion()  # enables interactive mode (allows keyboard interruption)
                # plt.title('iz='+str(iz))
                for isub in [221, 222, 223, 224]:
                    # plt.figure
                    plt.subplot(isub)
                    #ax = matplotlib.pyplot.axis()
                    if isub == 221:
                        plt.scatter(coord_src[:, 0], coord_src[:, 1], s=5, marker='o', zorder=10, color='steelblue', alpha=0.5)
                        pcaaxis = pca_src.components_.T
                        plt.title('src')
                    elif isub == 222:
                        plt.scatter(coord_src_rot[:, 0], coord_src_rot[:, 1], s=5, marker='o', zorder=10, color='steelblue', alpha=0.5)
                        pcaaxis = pca_dest.components_.T
                        plt.title('src_rot')
                    elif isub == 223:
                        plt.scatter(coord_dest[:, 0], coord_dest[:, 1], s=5, marker='o', zorder=10, color='red', alpha=0.5)
                        pcaaxis = pca_dest.components_.T
                        plt.title('dest')
                    elif isub == 224:
                        plt.scatter(coord_dest_rot[:, 0], coord_dest_rot[:, 1], s=5, marker='o', zorder=10, color='red', alpha=0.5)
                        pcaaxis = pca_src.components_.T
                        plt.title('dest_rot')
                    plt.text(-2.5, -2.5, str(pcaaxis), horizontalalignment='left', verticalalignment='bottom')
                    plt.plot([0, pcaaxis[0, 0]], [0, pcaaxis[1, 0]], linewidth=2, color='red')
                    plt.plot([0, pcaaxis[0, 1]], [0, pcaaxis[1, 1]], linewidth=2, color='orange')
                    plt.axis([-3, 3, -3, 3])
                    plt.gca().set_aspect('equal', adjustable='box')
                    # plt.axis('equal')
                plt.savefig('fig_pca_z'+str(iz)+'.png')
                plt.close()

            # get indices of x and y coordinates
            row, col = np.indices((nx, ny))
            # build 2xn array of coordinates in pixel space
            coord_init_pix = np.array([row.ravel(), col.ravel(), np.array(np.ones(len(row.ravel()))*iz)]).T
            # convert coordinates to physical space
            coord_init_phy = np.array(im_src.transfo_pix2phys(coord_init_pix))
            # get centermass coortinates in physical space
            centermass_src_phy = im_src.transfo_pix2phys([[centermass_src.T[0], centermass_src.T[1], iz]])
            centermass_dest_phy = im_src.transfo_pix2phys([[centermass_dest.T[0], centermass_dest.T[1], iz]])
            # build 3D rotation matrix
            R3d = np.eye(3)
            R3d[0:2, 0:2] = R
            # apply forward transformation (in physical space)
            coord_forward_phy = np.array( np.dot( (coord_init_phy - np.transpose(centermass_dest_phy[0])), R3d) + np.transpose(centermass_src_phy[0]))
            # apply inverse transformation (in physical space)
            coord_inverse_phy = np.array( np.dot( (coord_init_phy - np.transpose(centermass_src_phy[0])), R3d.T) + np.transpose(centermass_dest_phy[0]))
            # compute displacement per pixel in destination space (for forward warping field)
            warp_x[:, :, iz] = np.array([coord_forward_phy[i, 0] - coord_init_phy[i, 0] for i in xrange(nx*ny)]).reshape((nx, ny))
            warp_y[:, :, iz] = np.array([coord_forward_phy[i, 1] - coord_init_phy[i, 1] for i in xrange(nx*ny)]).reshape((nx, ny))
            # compute displacement per pixel in source space (for inverse warping field)
            warp_inv_x[:, :, iz] = np.array([coord_inverse_phy[i, 0] - coord_init_phy[i, 0] for i in xrange(nx*ny)]).reshape((nx, ny))
            warp_inv_y[:, :, iz] = np.array([coord_inverse_phy[i, 1] - coord_init_phy[i, 1] for i in xrange(nx*ny)]).reshape((nx, ny))

        # if one of the slice is empty, ignore it
        except ValueError:
            sct.printv('WARNING: Slice #'+str(iz)+' is empty. It will be ignored.', verbose, 'warning')

        # display init and new coordinates
        # plt.figure('iz='+str(iz)) #, figsize=(9, 4))
        # plt.scatter([coord_init_phy[i][0] for i in xrange(coord_init_phy.shape[0])],
        #             [coord_init_phy[i][1] for i in xrange(coord_init_phy.shape[0])],
        #             s=5, marker='o', zorder=10, color='steelblue', alpha=0.5)
        # plt.scatter([coord_forward_phy[i][0] for i in xrange(coord_forward_phy.shape[0])],
        #             [coord_forward_phy[i][1] for i in xrange(coord_forward_phy.shape[0])],
        #             s=5, marker='o', zorder=10, color='red', alpha=0.5)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()
        # del coord_src, coord_dest, pca_src, pca_dest

    # Generate forward warping field (defined in destination space)
    data_warp = np.zeros((nx, ny, nz, 1, 3))
    data_warp[:, :, :, 0, 0] = -warp_x  # need to invert due to ITK conventions
    data_warp[:, :, :, 0, 1] = -warp_y
    im_dest = load(fname_dest)
    hdr_dest = im_dest.get_header()
    hdr_warp = hdr_dest.copy()
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = Nifti1Image(data_warp, None, hdr_warp)
    save(img, fname_warp)
    sct.printv('\nDone! Warping field generated: '+fname_warp, verbose)
    # generate inverse warping field (defined in source space)
    data_warp = np.zeros((nx, ny, nz, 1, 3))
    data_warp[:, :, :, 0, 0] = -warp_inv_x  # need to invert due to ITK conventions
    data_warp[:, :, :, 0, 1] = -warp_inv_y  # need
    im = load(fname_src)
    hdr = im.get_header()
    hdr_warp = hdr.copy()
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = Nifti1Image(data_warp, None, hdr_warp)
    save(img, fname_warp_inv)
    sct.printv('\nDone! Warping field generated: '+fname_warp_inv, verbose)



def register2d_columnwise(fname_src, fname_dest, fname_warp='warp_forward.nii.gz', fname_warp_inv='warp_inverse.nii.gz', verbose=0):
    """
    Column-wise non-linear registration of segmentations. Based on an idea from Allan Martin.
    - Assumes src/dest are segmentations (not necessarily binary), and already registered by center of mass
    - Assumes src/dest are in RPI orientation.
    - Split along Z, then for each slice:
    - scale in R-L direction to match src/dest
    - loop across R-L columns and register by (i) matching center of mass and (ii) scaling.
    :param fname_src:
    :param fname_dest:
    :param fname_warp:
    :param fname_warp_inv:
    :param verbose:
    :return:
    """

    # initialization
    th_nonzero = 0.5  # values below are considered zero

    # for display stuff
    if verbose == 2:
        import matplotlib.pyplot as plt

    # Get image dimensions and retrieve nz
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
    sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
    sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)

    # Split source volume along z
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

    # open image
    data_src = im_src.data
    data_dest = im_dest.data

    # initialize forward warping field (defined in destination space)
    warp_x = np.zeros(data_dest.shape)
    warp_y = np.zeros(data_dest.shape)

    # initialize inverse warping field (defined in source space)
    warp_inv_x = np.zeros(data_src.shape)
    warp_inv_y = np.zeros(data_src.shape)

    # Loop across slices
    for iz in range(0, nz):

        # PREPARE COORDINATES
        # ============================================================
        # get indices of x and y coordinates
        row, col = np.indices((nx, ny))
        # build 2xn array of coordinates in pixel space
        # ordering of indices is as follows:
        # coord_init_pix[:, 0] = 0, 0, 0, ..., 1, 1, 1..., nx, nx, nx
        # coord_init_pix[:, 1] = 0, 1, 2, ..., 0, 1, 2..., 0, 1, 2
        coord_init_pix = np.array([row.ravel(), col.ravel(), np.array(np.ones(len(row.ravel()))*iz)]).T
        # convert coordinates to physical space
        coord_init_phy = np.array(im_src.transfo_pix2phys(coord_init_pix))
        # get 2d data from the selected slice
        src2d = data_src[:, :, iz]
        dest2d = data_dest[:, :, iz]
        # get non-zero coordinates, and transpose to obtain nx2 dimensions
        # here we use 0.5 as threshold for non-zero value
        coord_src2d = np.array(np.where(src2d > th_nonzero)).T
        coord_dest2d = np.array(np.where(dest2d > th_nonzero)).T
        # coord_src2d = np.array(src2d.nonzero()).T
        # coord_dest2d = np.array(dest2d.nonzero()).T

        # display image
        if verbose == 2:
            plt.figure(figsize=(15, 4))
            plt.subplot(121)
            plt.imshow(np.flipud(src2d.T), cmap=plt.cm.gray, interpolation='none')
            plt.title('src')
            plt.subplot(122)
            plt.imshow(np.flipud(dest2d.T), cmap=plt.cm.gray, interpolation='none')
            plt.title('dest')
            plt.show()

        # SCALING R-L (X dimension)
        # ============================================================
        # sum data across Y to obtain 1D signal: src_y and dest_y
        src1d = np.sum(src2d, 1)
        dest1d = np.sum(dest2d, 1)
        # retrieve min/max of non-zeros elements (edge of the segmentation)
        for i in xrange(len(src1d)):
            if src1d[i] > 0.5:
                # found index above 0.5, exit loop
                break
        ind_before = i-1
        ind_after = i
        # get indices (in continuous space) at half-maximum of upward and downward slope
        src1d_min, src1d_max = find_index_halfmax(src1d)
        dest1d_min, dest1d_max = find_index_halfmax(dest1d)
        # 1D matching between src_y and dest_y
        mean_dest = (dest1d_max + dest1d_min)/2
        mean_src = (src1d_max + src1d_min)/2
        # Tx = (dest1d_max + dest1d_min)/2 - (src1d_max + src1d_min)/2
        Sx = (dest1d_max - dest1d_min) / float(src1d_max - src1d_min)
        # apply translation and scaling to src (interpolate)
        # display
        # if verbose == 2:
        #     matrix = [[1/Sx, 0], [0, 1]]
        #     src2d_scaleX = ndimage.affine_transform(src2d, matrix, offset=[-Tx-nx, 0]) #Ty+ny/2, nx/2])
        #     plt.figure(figsize=(15, 4))
        #     plt.subplot(131)
        #     plt.imshow(np.swapaxes(src2d, 1, 0), cmap=plt.cm.gray, interpolation='none')
        #     plt.title('src')
        #     plt.xlabel('x')
        #     plt.ylabel('y')
        #     plt.subplot(132)
        #     plt.imshow(np.swapaxes(src2d_scaleX, 1, 0), cmap=plt.cm.gray, interpolation='none')
        #     plt.title('src_scaleX')
        #     plt.xlabel('x')
        #     plt.ylabel('y')
        #     plt.subplot(133)
        #     plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.gray, interpolation='none')
        #     plt.title('dest')
        #     plt.show()

        # apply forward transformation (in pixel space)
        # below: only for debugging purpose
        coord_src2d_scaleX = np.copy(coord_src2d)  # need to use np.copy to avoid copying pointer
        coord_src2d_scaleX[:, 0] = (coord_src2d[:, 0] - mean_src) * Sx + mean_dest
        coord_init_pix_scaleX = np.copy(coord_init_pix)
        coord_init_pix_scaleX[:, 0] = (coord_init_pix[:, 0] - mean_src ) * Sx + mean_dest
        coord_init_pix_scaleXinv = np.copy(coord_init_pix)
        coord_init_pix_scaleXinv[:, 0] = (coord_init_pix[:, 0] - mean_dest ) / float(Sx) + mean_src

        # ============================================================
        # COLUMN-WISE REGISTRATION
        # ============================================================
        coord_init_pix_scaleY = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
        coord_init_pix_scaleYinv = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
        # coord_src2d_scaleXY = np.copy(coord_src2d_scaleX)  # need to use np.copy to avoid copying pointer
        # loop across columns (X dimension)
        for ix in xrange(nx):
            # retrieve 1D signal along Y
            src1d = src2d[ix, :]
            dest1d = dest2d[ix, :]
            # make sure there are non-zero data in src or dest
            if np.any(src1d) and np.any(dest1d):
                # retrieve min/max of non-zeros elements (edge of the segmentation)
                # src1d_min, src1d_max = min(np.nonzero(src1d)[0]), max(np.nonzero(src1d)[0])
                # dest1d_min, dest1d_max = min(np.nonzero(dest1d)[0]), max(np.nonzero(dest1d)[0])
                # 1D matching between src_y and dest_y
                # Ty = (dest1d_max + dest1d_min)/2 - (src1d_max + src1d_min)/2
                # Sy = (dest1d_max - dest1d_min) / float(src1d_max - src1d_min)
                # apply translation and scaling to coordinates in column
                # get indices (in continuous space) at half-maximum of upward and downward slope
                src1d_min, src1d_max = find_index_halfmax(src1d)
                dest1d_min, dest1d_max = find_index_halfmax(dest1d)
                # src1d_min, src1d_max = np.min(np.where(src1d > th_nonzero)), np.max(np.where(src1d > th_nonzero))
                # dest1d_min, dest1d_max = np.min(np.where(dest1d > th_nonzero)), np.max(np.where(dest1d > th_nonzero))
                # 1D matching between src_y and dest_y
                mean_dest = (dest1d_max + dest1d_min)/2
                mean_src = (src1d_max + src1d_min)/2
                # Tx = (dest1d_max + dest1d_min)/2 - (src1d_max + src1d_min)/2
                Sy = (dest1d_max - dest1d_min) / float(src1d_max - src1d_min)
                # apply forward transformation (in pixel space)
                # below: only for debugging purpose
                # coord_src2d_scaleX = np.copy(coord_src2d)  # need to use np.copy to avoid copying pointer
                # coord_src2d_scaleX[:, 0] = (coord_src2d[:, 0] - mean_src) * Sx + mean_dest
                # coord_init_pix_scaleY = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
                # coord_init_pix_scaleY[:, 0] = (coord_init_pix[:, 0] - mean_src ) * Sx + mean_dest
                coord_init_pix_scaleY[ix*nx:ny+ix*nx, 1] = (coord_init_pix[ix*nx:ny+ix*nx, 1] - mean_src) * Sy + mean_dest
                coord_init_pix_scaleYinv[ix*nx:ny+ix*nx, 1] = (coord_init_pix[ix*nx:ny+ix*nx, 1] - mean_dest) / float(Sy) + mean_src
        # display
        if verbose == 2:
            plt.figure(figsize=(15, 4))
            list_data = [coord_init_pix, coord_init_pix_scaleX, coord_init_pix_scaleY]
            list_subplot = [131, 132, 133]
            list_title = ['src', 'src_scaleX', 'src_scaleY']
            for i in xrange(len(list_subplot)):
                plt.subplot(list_subplot[i])
                plt.scatter([list_data[i][ipix][0] for ipix in xrange(list_data[i].shape[0])],
                            [list_data[i][ipix][1] for ipix in xrange(list_data[i].shape[0])],
                            s=15, marker='+', zorder=1, color='black', alpha=1)
                plt.scatter([coord_dest2d[ipix][0] for ipix in xrange(coord_dest2d.shape[0])],
                            [coord_dest2d[ipix][1] for ipix in xrange(coord_dest2d.shape[0])],
                            s=15, marker='x', zorder=2, color='red', alpha=1)
                plt.scatter([coord_src2d[ipix][0] for ipix in xrange(coord_src2d.shape[0])],
                            [coord_src2d[ipix][1] for ipix in xrange(coord_src2d.shape[0])],
                            s=5, marker='o', zorder=2, color='blue', alpha=1)
                plt.scatter([coord_src2d_scaleX[ipix][0] for ipix in xrange(coord_src2d_scaleX.shape[0])],
                            [coord_src2d_scaleX[ipix][1] for ipix in xrange(coord_src2d_scaleX.shape[0])],
                            s=5, marker='o', zorder=2, color='green', alpha=1)
                plt.xlim(0, nx-1)
                plt.ylim(0, ny-1)
                plt.grid()
                plt.title(list_title[i])
                plt.xlabel('x')
                plt.ylabel('y')
            plt.show()

        # ============================================================
        # CALCULATE TRANSFORMATIONS
        # ============================================================
        # calculate forward transformation (in physical space)
        coord_init_phy_scaleX = np.array(im_dest.transfo_pix2phys(coord_init_pix_scaleX))
        coord_init_phy_scaleY = np.array(im_dest.transfo_pix2phys(coord_init_pix_scaleY))
        # calculate inverse transformation (in physical space)
        coord_init_phy_scaleXinv = np.array(im_src.transfo_pix2phys(coord_init_pix_scaleXinv))
        coord_init_phy_scaleYinv = np.array(im_src.transfo_pix2phys(coord_init_pix_scaleYinv))
        # compute displacement per pixel in destination space (for forward warping field)
        warp_x[:, :, iz] = np.array([coord_init_phy_scaleXinv[i, 0] - coord_init_phy[i, 0] for i in xrange(nx*ny)]).reshape((nx, ny))
        warp_y[:, :, iz] = np.array([coord_init_phy_scaleYinv[i, 1] - coord_init_phy[i, 1] for i in xrange(nx*ny)]).reshape((nx, ny))
        # compute displacement per pixel in source space (for inverse warping field)
        warp_inv_x[:, :, iz] = np.array([coord_init_phy_scaleX[i, 0] - coord_init_phy[i, 0] for i in xrange(nx*ny)]).reshape((nx, ny))
        warp_inv_y[:, :, iz] = np.array([coord_init_phy_scaleY[i, 1] - coord_init_phy[i, 1] for i in xrange(nx*ny)]).reshape((nx, ny))

    # Generate forward warping field (defined in destination space)
    data_warp = np.zeros(((((nx, ny, nz, 1, 3)))))
    data_warp[:, :, :, 0, 0] = -warp_x  # need to invert due to ITK conventions
    data_warp[:, :, :, 0, 1] = -warp_y
    im_dest = load(fname_dest)
    hdr_dest = im_dest.get_header()
    hdr_warp = hdr_dest.copy()
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = Nifti1Image(data_warp, None, hdr_warp)
    save(img, fname_warp)
    sct.printv('\nDone! Warping field generated: '+fname_warp, verbose)
    # generate inverse warping field (defined in source space)
    data_warp = np.zeros(((((nx, ny, nz, 1, 3)))))
    data_warp[:, :, :, 0, 0] = -warp_inv_x  # need to invert due to ITK conventions
    data_warp[:, :, :, 0, 1] = -warp_inv_y  # need
    im = load(fname_src)
    hdr = im.get_header()
    hdr_warp = hdr.copy()
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = Nifti1Image(data_warp, None, hdr_warp)
    save(img, fname_warp_inv)
    sct.printv('\nDone! Warping field generated: '+fname_warp_inv, verbose)

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
    # coord_diff_origin = (np.asarray(coord_origin_dest[0]) - np.asarray(coord_origin_input[0])).tolist()
    # [x_o, y_o, z_o] = [coord_diff_origin[0] * 1.0/px, coord_diff_origin[1] * 1.0/py, coord_diff_origin[2] * 1.0/pz]

    # initialization
    if paramreg.algo in ['Translation']:
        x_displacement = [0 for i in range(nz)]
        y_displacement = [0 for i in range(nz)]
        theta_rotation = [0 for i in range(nz)]
    if paramreg.algo in ['Rigid', 'Affine', 'BSplineSyN', 'SyN']:
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

            if paramreg.algo in ['Translation']:
                file_mat = prefix_warp2d+'0GenericAffine.mat'
                matfile = loadmat(file_mat, struct_as_record=True)
                array_transfo = matfile['AffineTransform_double_2_2']
                x_displacement[i] = array_transfo[4][0]  # Tx in ITK'S coordinate system
                y_displacement[i] = array_transfo[5][0]  # Ty  in ITK'S and fslview's coordinate systems
                theta_rotation[i] = asin(array_transfo[2]) # angle of rotation theta in ITK'S coordinate system (minus theta for fslview)

            if paramreg.algo in ['Rigid', 'Affine', 'BSplineSyN', 'SyN']:
                # List names of 2d warping fields for subsequent merge along Z
                file_warp2d = prefix_warp2d+'0Warp.nii.gz'
                file_warp2d_inv = prefix_warp2d+'0InverseWarp.nii.gz'
                list_warp.append(file_warp2d)
                list_warp_inv.append(file_warp2d_inv)

            if paramreg.algo in ['Rigid', 'Affine']:
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

    if paramreg.algo in ['Translation']:
        # convert to array
        x_disp_a = np.asarray(x_displacement)
        y_disp_a = np.asarray(y_displacement)
        theta_rot_a = np.asarray(theta_rotation)
        # Generate warping field
        generate_warping_field('dest.nii', x_disp_a, y_disp_a, theta_rot_a, fname=fname_warp)  #name_warp= 'step'+str(paramreg.step)
        # Inverse warping field
        generate_warping_field('src.nii', -x_disp_a, -y_disp_a, theta_rot_a, fname=fname_warp_inv)

    if paramreg.algo in ['Rigid', 'Affine', 'BSplineSyN', 'SyN']:
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
    sct.printv('\nCreating warping field for transformations along z...', verbose)

    file_dest = load(fname_dest)
    hdr_file_dest = file_dest.get_header()
    hdr_warp = hdr_file_dest.copy()

    # Get image dimensions
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
    sct.printv('.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
    sct.printv('.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)

    # Center of rotation
    if center_rotation == None:
        x_a = int(round(nx/2))
        y_a = int(round(ny/2))
    else:
        x_a = center_rotation[0]
        y_a = center_rotation[1]

    # Calculate displacement for each voxel
    data_warp = np.zeros(((((nx, ny, nz, 1, 3)))))
    vector_i = [[[i-x_a], [j-y_a]] for i in range(nx) for j in range(ny)]

    # if theta_rot == None:
    #     # for translations
    #     for k in range(nz):
    #         matrix_rot_a = np.asarray([[cos(theta_rot[k]), - sin(theta_rot[k])], [-sin(theta_rot[k]), -cos(theta_rot[k])]])
    #         tmp = matrix_rot_a + array(((-1, 0), (0, 1)))
    #         result = dot(tmp, array(vector_i).T[0]) + array([[x_trans[k]], [y_trans[k]]])
    #         for i in range(ny):
    #             data_warp[i, :, k, 0, 0] = result[0][i*nx:i*nx+ny]
    #             data_warp[i, :, k, 0, 1] = result[1][i*nx:i*nx+ny]

    # else:
        # For rigid transforms (not optimized)
        # if theta_rot != None:
    # TODO: this is not optimized! can do better!
    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                data_warp[i, j, k, 0, 0] = (cos(theta_rot[k]) - 1) * (i - x_a) - sin(theta_rot[k]) * (j - y_a) + x_trans[k]
                data_warp[i, j, k, 0, 1] = - sin(theta_rot[k]) * (i - x_a) - (cos(theta_rot[k]) - 1) * (j - y_a) + y_trans[k]
                data_warp[i, j, k, 0, 2] = 0

    # Generate warp file as a warping field
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = Nifti1Image(data_warp, None, hdr_warp)
    save(img, fname)
    sct.printv('\nDone! Warping field generated: '+fname, verbose)



def angle_between(a, b):
    """
    compute angle in radian between a and b. Throws an exception if a or b has zero magnitude.
    :param a:
    :param b:
    :return:
    """
    # TODO: check if extreme value that can make the function crash-- use "try"
    # from numpy.linalg import norm
    # from numpy import dot
    # import math
    arccosInput = np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b)
    # print arccosInput
    arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
    arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
    sign_angle = np.sign(np.cross(a, b))
    # print sign_angle
    return sign_angle * acos(arccosInput)

    # @xl_func("numpy_row v1, numpy_row v2: float")
    # def py_ang(v1, v2):
    # """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    # cosang = np.dot(a, b)
    # sinang = la.norm(np.cross(a, b))
    # return np.arctan2(sinang, cosang)


def compute_pca(data2d):
    """
    Compute PCA using sklearn
    :param data2d: 2d array. PCA will be computed on non-zeros values.
    :return:
        coordsrc: 2d array: centered non-zero coordinates
        pca: object: PCA result.
        centermass: 2x1 array: 2d coordinates of the center of mass
    """
    # round it and make it int (otherwise end up with values like 10-7)
    data2d = data2d.round().astype(int)
    # get non-zero coordinates, and transpose to obtain nx2 dimensions
    coordsrc = np.array(data2d.nonzero()).T
    # get center of mass
    centermass = coordsrc.mean(0)
    # center data
    coordsrc = coordsrc - centermass
    # normalize data
    coordsrc /= coordsrc.std()
    # Performs PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, copy=False, whiten=False)
    pca.fit(coordsrc)
    # pca_score = pca.explained_variance_ratio_
    # V = pca.components_
    return coordsrc, pca, centermass



def find_index_halfmax(data1d):
    """
    Find the two indices at half maximum for a bell-type curve (non-parametric). Uses center of mass calculation.
    :param data1d:
    :return: xmin, xmax
    """
    # normalize data between 0 and 1
    data1d = data1d / float(np.max(data1d))
    # loop across elements and stops when found 0.5
    for i in xrange(len(data1d)):
        if data1d[i] > 0.5:
            break
    # compute center of mass to get coordinate at 0.5
    xmin = i - 1 + (0.5 - data1d[i-1]) / float(data1d[i] - data1d[i-1])
    # continue for the descending slope
    for i in range(i, len(data1d)):
        if data1d[i] < 0.5:
            break
    # compute center of mass to get coordinate at 0.5
    xmax = i - 1 + (0.5 - data1d[i-1]) / float(data1d[i] - data1d[i-1])
    # display
    # plt.figure()
    # plt.plot(src1d)
    # plt.plot(xmin, 0.5, 'o')
    # plt.plot(xmax, 0.5, 'o')
    # plt.show()
    return xmin, xmax