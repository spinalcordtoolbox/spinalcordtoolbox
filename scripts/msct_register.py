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

import sys, os, shutil
from math import asin, cos, sin, acos
import numpy as np

from scipy import ndimage
from scipy.io import loadmat
from nibabel import load, Nifti1Image, save

from msct_image import Image
import sct_utils as sct
from sct_convert import convert
from sct_register_multimodal import Paramreg


def register_slicewise(fname_src,
                        fname_dest,
                        fname_mask='',
                        warp_forward_out='step0Warp.nii.gz',
                        warp_inverse_out='step0InverseWarp.nii.gz',
                        paramreg=None,
                        ants_registration_params=None,
                        path_qc='./',
                        verbose=0):

    # create temporary folder
    path_tmp = sct.tmp_create(basename="register", verbose=verbose)

    # copy data to temp folder
    sct.printv('\nCopy input data to temp folder...', verbose)
    convert(fname_src, os.path.join(path_tmp, "src.nii"))
    convert(fname_dest, os.path.join(path_tmp, "dest.nii"))
    if fname_mask != '':
        convert(fname_mask, os.path.join(path_tmp, "mask.nii.gz"))

    # go to temporary folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Calculate displacement
    if paramreg.algo == 'centermass':
        # translation of center of mass between source and destination in voxel space
        register2d_centermassrot('src.nii', 'dest.nii', fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out, rot=0, poly=int(paramreg.poly), path_qc=path_qc, verbose=verbose)
    elif paramreg.algo == 'centermassrot':
        # translation of center of mass and rotation based on source and destination first eigenvectors from PCA.
        register2d_centermassrot('src.nii', 'dest.nii', fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out, rot=1, poly=int(paramreg.poly), path_qc=path_qc, verbose=verbose, pca_eigenratio_th=float(paramreg.pca_eigenratio_th))
    elif paramreg.algo == 'columnwise':
        # scaling R-L, then column-wise center of mass alignment and scaling
        register2d_columnwise('src.nii', 'dest.nii', fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out, verbose=verbose, path_qc=path_qc, smoothWarpXY=int(paramreg.smoothWarpXY))
    else:
        # convert SCT flags into ANTs-compatible flags
        algo_dic = {'translation': 'Translation', 'rigid': 'Rigid', 'affine': 'Affine', 'syn': 'SyN', 'bsplinesyn': 'BSplineSyN', 'centermass': 'centermass'}
        paramreg.algo = algo_dic[paramreg.algo]
        # run slicewise registration
        register2d('src.nii', 'dest.nii', fname_mask=fname_mask, fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out, paramreg=paramreg, ants_registration_params=ants_registration_params, verbose=verbose)

    sct.printv('\nMove warping fields...', verbose)
    sct.copy(warp_forward_out, curdir)
    sct.copy(warp_inverse_out, curdir)

    # go back
    os.chdir(curdir)


def register2d_centermassrot(fname_src, fname_dest, fname_warp='warp_forward.nii.gz', fname_warp_inv='warp_inverse.nii.gz', rot=1, poly=0, path_qc='./', verbose=0, pca_eigenratio_th=1.6):
    """
    Rotate the source image to match the orientation of the destination image, using the first and second eigenvector
    of the PCA. This function should be used on segmentations (not images).
    This works for 2D and 3D images.  If 3D, it splits the image and performs the rotation slice-by-slice.
    input:
        fname_source: name of moving image (type: string)
        fname_dest: name of fixed image (type: string)
        fname_warp: name of output 3d forward warping field
        fname_warp_inv: name of output 3d inverse warping field
        rot: estimate rotation with PCA (type: int)
        poly: degree of polynomial regularization along z for rotation angle (type: int). 0: no regularization
        verbose:
    output:
        none
    """

    if verbose == 2:
        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt

    # Get image dimensions and retrieve nz
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
    sct.printv('  matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
    sct.printv('  voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', verbose)

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

    if len(data_src.shape) == 2:
        # reshape 2D data into pseudo 3D (only one slice)
        new_shape = list(data_src.shape)
        new_shape.append(1)
        new_shape = tuple(new_shape)
        data_src = data_src.reshape(new_shape)
        data_dest = data_dest.reshape(new_shape)

    # initialize displacement and rotation
    coord_src = [None] * nz
    pca_src = [None] * nz
    coord_dest = [None] * nz
    pca_dest = [None] * nz
    centermass_src = np.zeros([nz, 2])
    centermass_dest = np.zeros([nz, 2])
    # displacement_forward = np.zeros([nz, 2])
    # displacement_inverse = np.zeros([nz, 2])
    angle_src_dest = np.zeros(nz)
    z_nonzero = []
    # Loop across slices
    for iz in range(0, nz):
        try:
            # compute PCA and get center or mass
            coord_src[iz], pca_src[iz], centermass_src[iz, :] = compute_pca(data_src[:, :, iz])
            coord_dest[iz], pca_dest[iz], centermass_dest[iz, :] = compute_pca(data_dest[:, :, iz])
            # compute (src,dest) angle for first eigenvector
            if rot == 1:
                eigenv_src = pca_src[iz].components_.T[0][0], pca_src[iz].components_.T[1][0]  # pca_src.components_.T[0]
                eigenv_dest = pca_dest[iz].components_.T[0][0], pca_dest[iz].components_.T[1][0]  # pca_dest.components_.T[0]
                angle_src_dest[iz] = angle_between(eigenv_src, eigenv_dest)
                # check if ratio between the two eigenvectors is high enough to prevent poor robustness
                if pca_src[iz].explained_variance_ratio_[0] / pca_src[iz].explained_variance_ratio_[1] < pca_eigenratio_th:
                    angle_src_dest[iz] = 0
                if pca_dest[iz].explained_variance_ratio_[0] / pca_dest[iz].explained_variance_ratio_[1] < pca_eigenratio_th:
                    angle_src_dest[iz] = 0
            # append to list of z_nonzero
            z_nonzero.append(iz)
        # if one of the slice is empty, ignore it
        except ValueError:
            sct.printv('WARNING: Slice #' + str(iz) + ' is empty. It will be ignored.', verbose, 'warning')

    # regularize rotation
    if not poly == 0 and rot == 1:
        from msct_smooth import polynomial_fit
        angle_src_dest_regularized = polynomial_fit(z_nonzero, angle_src_dest[z_nonzero], poly)[0]
        # display
        if verbose == 2:
            plt.plot(180 * angle_src_dest[z_nonzero] / np.pi)
            plt.plot(180 * angle_src_dest_regularized / np.pi, 'r', linewidth=2)
            plt.grid()
            plt.xlabel('z')
            plt.ylabel('Angle (deg)')
            plt.savefig(os.path.join(path_qc, 'register2d_centermassrot_regularize_rotation.png'))
            plt.close()
        # update variable
        angle_src_dest[z_nonzero] = angle_src_dest_regularized

    # initialize warping fields
    # N.B. forward transfo is defined in destination space and inverse transfo is defined in the source space
    warp_x = np.zeros(data_dest.shape)
    warp_y = np.zeros(data_dest.shape)
    warp_inv_x = np.zeros(data_src.shape)
    warp_inv_y = np.zeros(data_src.shape)

    # construct 3D warping matrix
    for iz in z_nonzero:
        sct.printv(str(iz) + '/' + str(nz) + '..',)
        # get indices of x and y coordinates
        row, col = np.indices((nx, ny))
        # build 2xn array of coordinates in pixel space
        coord_init_pix = np.array([row.ravel(), col.ravel(), np.array(np.ones(len(row.ravel())) * iz)]).T
        # convert coordinates to physical space
        coord_init_phy = np.array(im_src.transfo_pix2phys(coord_init_pix))
        # get centermass coordinates in physical space
        centermass_src_phy = im_src.transfo_pix2phys([[centermass_src[iz, :].T[0], centermass_src[iz, :].T[1], iz]])[0]
        centermass_dest_phy = im_src.transfo_pix2phys([[centermass_dest[iz, :].T[0], centermass_dest[iz, :].T[1], iz]])[0]
        # build rotation matrix
        R = np.matrix(((cos(angle_src_dest[iz]), sin(angle_src_dest[iz])), (-sin(angle_src_dest[iz]), cos(angle_src_dest[iz]))))
        # build 3D rotation matrix
        R3d = np.eye(3)
        R3d[0:2, 0:2] = R
        # apply forward transformation (in physical space)
        coord_forward_phy = np.array(np.dot((coord_init_phy - np.transpose(centermass_dest_phy)), R3d) + np.transpose(centermass_src_phy))
        # apply inverse transformation (in physical space)
        coord_inverse_phy = np.array(np.dot((coord_init_phy - np.transpose(centermass_src_phy)), R3d.T) + np.transpose(centermass_dest_phy))
        # display rotations
        if verbose == 2 and not angle_src_dest[iz] == 0:
            # compute new coordinates
            coord_src_rot = coord_src[iz] * R
            coord_dest_rot = coord_dest[iz] * R.T
            # generate figure
            plt.figure('iz=' + str(iz) + ', angle_src_dest=' + str(angle_src_dest[iz]), figsize=(9, 9))
            # plt.ion()  # enables interactive mode (allows keyboard interruption)
            # plt.title('iz='+str(iz))
            for isub in [221, 222, 223, 224]:
                # plt.figure
                plt.subplot(isub)
                # ax = matplotlib.pyplot.axis()
                try:
                    if isub == 221:
                        plt.scatter(coord_src[iz][:, 0], coord_src[iz][:, 1], s=5, marker='o', zorder=10, color='steelblue',
                                    alpha=0.5)
                        pcaaxis = pca_src[iz].components_.T
                        pca_eigenratio = pca_src[iz].explained_variance_ratio_
                        plt.title('src')
                    elif isub == 222:
                        plt.scatter([coord_src_rot[i, 0] for i in range(len(coord_src_rot))], [coord_src_rot[i, 1] for i in range(len(coord_src_rot))], s=5, marker='o', zorder=10, color='steelblue', alpha=0.5)
                        pcaaxis = pca_dest[iz].components_.T
                        pca_eigenratio = pca_dest[iz].explained_variance_ratio_
                        plt.title('src_rot')
                    elif isub == 223:
                        plt.scatter(coord_dest[iz][:, 0], coord_dest[iz][:, 1], s=5, marker='o', zorder=10, color='red',
                                    alpha=0.5)
                        pcaaxis = pca_dest[iz].components_.T
                        pca_eigenratio = pca_dest[iz].explained_variance_ratio_
                        plt.title('dest')
                    elif isub == 224:
                        plt.scatter([coord_dest_rot[i, 0] for i in range(len(coord_dest_rot))], [coord_dest_rot[i, 1] for i in range(len(coord_dest_rot))], s=5, marker='o', zorder=10, color='red', alpha=0.5)
                        pcaaxis = pca_src[iz].components_.T
                        pca_eigenratio = pca_src[iz].explained_variance_ratio_
                        plt.title('dest_rot')
                    plt.text(-2.5, -2, 'eigenvectors:', horizontalalignment='left', verticalalignment='bottom')
                    plt.text(-2.5, -2.8, str(pcaaxis), horizontalalignment='left', verticalalignment='bottom')
                    plt.text(-2.5, 2.5, 'eigenval_ratio:', horizontalalignment='left', verticalalignment='bottom')
                    plt.text(-2.5, 2, str(pca_eigenratio), horizontalalignment='left', verticalalignment='bottom')
                    plt.plot([0, pcaaxis[0, 0]], [0, pcaaxis[1, 0]], linewidth=2, color='red')
                    plt.plot([0, pcaaxis[0, 1]], [0, pcaaxis[1, 1]], linewidth=2, color='orange')
                    plt.axis([-3, 3, -3, 3])
                    plt.gca().set_aspect('equal', adjustable='box')
                except Exception as e:
                    raise Exception
                    # sct.printv('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), 1, 'warning')
                    # sct.printv('WARNING: '+str(e), 1, 'warning')

                    # plt.axis('equal')
            plt.savefig(os.path.join(path_qc, 'register2d_centermassrot_pca_z' + str(iz) + '.png'))
            plt.close()

        # construct 3D warping matrix
        warp_x[:, :, iz] = np.array([coord_forward_phy[i, 0] - coord_init_phy[i, 0] for i in xrange(nx * ny)]).reshape((nx, ny))
        warp_y[:, :, iz] = np.array([coord_forward_phy[i, 1] - coord_init_phy[i, 1] for i in xrange(nx * ny)]).reshape((nx, ny))
        warp_inv_x[:, :, iz] = np.array([coord_inverse_phy[i, 0] - coord_init_phy[i, 0] for i in xrange(nx * ny)]).reshape((nx, ny))
        warp_inv_y[:, :, iz] = np.array([coord_inverse_phy[i, 1] - coord_init_phy[i, 1] for i in xrange(nx * ny)]).reshape((nx, ny))

    # Generate forward warping field (defined in destination space)
    generate_warping_field(fname_dest, warp_x, warp_y, fname_warp, verbose)
    generate_warping_field(fname_src, warp_inv_x, warp_inv_y, fname_warp_inv, verbose)


def register2d_columnwise(fname_src, fname_dest, fname_warp='warp_forward.nii.gz', fname_warp_inv='warp_inverse.nii.gz', verbose=0, path_qc='./', smoothWarpXY=1):
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
        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt

    # Get image dimensions and retrieve nz
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
    sct.printv('  matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
    sct.printv('  voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', verbose)

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

    if len(data_src.shape) == 2:
        # reshape 2D data into pseudo 3D (only one slice)
        new_shape = list(data_src.shape)
        new_shape.append(1)
        new_shape = tuple(new_shape)
        data_src = data_src.reshape(new_shape)
        data_dest = data_dest.reshape(new_shape)

    # initialize forward warping field (defined in destination space)
    warp_x = np.zeros(data_dest.shape)
    warp_y = np.zeros(data_dest.shape)

    # initialize inverse warping field (defined in source space)
    warp_inv_x = np.zeros(data_src.shape)
    warp_inv_y = np.zeros(data_src.shape)

    # Loop across slices
    sct.printv('\nEstimate columnwise transformation...', verbose)
    for iz in range(0, nz):
        sct.printv(str(iz) + '/' + str(nz) + '..',)

        # PREPARE COORDINATES
        # ============================================================
        # get indices of x and y coordinates
        row, col = np.indices((nx, ny))
        # build 2xn array of coordinates in pixel space
        # ordering of indices is as follows:
        # coord_init_pix[:, 0] = 0, 0, 0, ..., 1, 1, 1..., nx, nx, nx
        # coord_init_pix[:, 1] = 0, 1, 2, ..., 0, 1, 2..., 0, 1, 2
        coord_init_pix = np.array([row.ravel(), col.ravel(), np.array(np.ones(len(row.ravel())) * iz)]).T
        # convert coordinates to physical space
        coord_init_phy = np.array(im_src.transfo_pix2phys(coord_init_pix))
        # get 2d data from the selected slice
        src2d = data_src[:, :, iz]
        dest2d = data_dest[:, :, iz]
        # julien 20161105
        #<<<
        # threshold at 0.5
        src2d[src2d < th_nonzero] = 0
        dest2d[dest2d < th_nonzero] = 0
        # get non-zero coordinates, and transpose to obtain nx2 dimensions
        coord_src2d = np.array(np.where(src2d > 0)).T
        coord_dest2d = np.array(np.where(dest2d > 0)).T
        # here we use 0.5 as threshold for non-zero value
        # coord_src2d = np.array(np.where(src2d > th_nonzero)).T
        # coord_dest2d = np.array(np.where(dest2d > th_nonzero)).T
        #>>>

        # SCALING R-L (X dimension)
        # ============================================================
        # sum data across Y to obtain 1D signal: src_y and dest_y
        src1d = np.sum(src2d, 1)
        dest1d = np.sum(dest2d, 1)
        # make sure there are non-zero data in src or dest
        if np.any(src1d > th_nonzero) and np.any(dest1d > th_nonzero):
            # retrieve min/max of non-zeros elements (edge of the segmentation)
            # julien 20161105
            # <<<
            src1d_min, src1d_max = min(np.where(src1d != 0)[0]), max(np.where(src1d != 0)[0])
            dest1d_min, dest1d_max = min(np.where(dest1d != 0)[0]), max(np.where(dest1d != 0)[0])
            # for i in xrange(len(src1d)):
            #     if src1d[i] > 0.5:
            #         found index above 0.5, exit loop
                    # break
            # get indices (in continuous space) at half-maximum of upward and downward slope
            # src1d_min, src1d_max = find_index_halfmax(src1d)
            # dest1d_min, dest1d_max = find_index_halfmax(dest1d)
            # >>>
            # 1D matching between src_y and dest_y
            mean_dest_x = (dest1d_max + dest1d_min) / 2
            mean_src_x = (src1d_max + src1d_min) / 2
            # compute x-scaling factor
            Sx = (dest1d_max - dest1d_min + 1) / float(src1d_max - src1d_min + 1)
            # apply transformation to coordinates
            coord_src2d_scaleX = np.copy(coord_src2d)  # need to use np.copy to avoid copying pointer
            coord_src2d_scaleX[:, 0] = (coord_src2d[:, 0] - mean_src_x) * Sx + mean_dest_x
            coord_init_pix_scaleX = np.copy(coord_init_pix)
            coord_init_pix_scaleX[:, 0] = (coord_init_pix[:, 0] - mean_src_x) * Sx + mean_dest_x
            coord_init_pix_scaleXinv = np.copy(coord_init_pix)
            coord_init_pix_scaleXinv[:, 0] = (coord_init_pix[:, 0] - mean_dest_x) / float(Sx) + mean_src_x
            # apply transformation to image
            from skimage.transform import warp
            row_scaleXinv = np.reshape(coord_init_pix_scaleXinv[:, 0], [nx, ny])
            src2d_scaleX = warp(src2d, np.array([row_scaleXinv, col]), order=1)

            # ============================================================
            # COLUMN-WISE REGISTRATION (Y dimension for each Xi)
            # ============================================================
            coord_init_pix_scaleY = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
            coord_init_pix_scaleYinv = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
            # coord_src2d_scaleXY = np.copy(coord_src2d_scaleX)  # need to use np.copy to avoid copying pointer
            # loop across columns (X dimension)
            for ix in xrange(nx):
                # retrieve 1D signal along Y
                src1d = src2d_scaleX[ix, :]
                dest1d = dest2d[ix, :]
                # make sure there are non-zero data in src or dest
                if np.any(src1d > th_nonzero) and np.any(dest1d > th_nonzero):
                    # retrieve min/max of non-zeros elements (edge of the segmentation)
                    # src1d_min, src1d_max = min(np.nonzero(src1d)[0]), max(np.nonzero(src1d)[0])
                    # dest1d_min, dest1d_max = min(np.nonzero(dest1d)[0]), max(np.nonzero(dest1d)[0])
                    # 1D matching between src_y and dest_y
                    # Ty = (dest1d_max + dest1d_min)/2 - (src1d_max + src1d_min)/2
                    # Sy = (dest1d_max - dest1d_min) / float(src1d_max - src1d_min)
                    # apply translation and scaling to coordinates in column
                    # get indices (in continuous space) at half-maximum of upward and downward slope
                    # src1d_min, src1d_max = find_index_halfmax(src1d)
                    # dest1d_min, dest1d_max = find_index_halfmax(dest1d)
                    src1d_min, src1d_max = np.min(np.where(src1d > th_nonzero)), np.max(np.where(src1d > th_nonzero))
                    dest1d_min, dest1d_max = np.min(np.where(dest1d > th_nonzero)), np.max(np.where(dest1d > th_nonzero))
                    # 1D matching between src_y and dest_y
                    mean_dest_y = (dest1d_max + dest1d_min) / 2
                    mean_src_y = (src1d_max + src1d_min) / 2
                    # Tx = (dest1d_max + dest1d_min)/2 - (src1d_max + src1d_min)/2
                    Sy = (dest1d_max - dest1d_min + 1) / float(src1d_max - src1d_min + 1)
                    # apply forward transformation (in pixel space)
                    # below: only for debugging purpose
                    # coord_src2d_scaleX = np.copy(coord_src2d)  # need to use np.copy to avoid copying pointer
                    # coord_src2d_scaleX[:, 0] = (coord_src2d[:, 0] - mean_src) * Sx + mean_dest
                    # coord_init_pix_scaleY = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
                    # coord_init_pix_scaleY[:, 0] = (coord_init_pix[:, 0] - mean_src ) * Sx + mean_dest
                    range_x = range(ix * ny, ix * ny + nx)
                    coord_init_pix_scaleY[range_x, 1] = (coord_init_pix[range_x, 1] - mean_src_y) * Sy + mean_dest_y
                    coord_init_pix_scaleYinv[range_x, 1] = (coord_init_pix[range_x, 1] - mean_dest_y) / float(Sy) + mean_src_y
            # apply transformation to image
            col_scaleYinv = np.reshape(coord_init_pix_scaleYinv[:, 1], [nx, ny])
            src2d_scaleXY = warp(src2d, np.array([row_scaleXinv, col_scaleYinv]), order=1)
            # regularize Y warping fields
            from skimage.filters import gaussian
            col_scaleY = np.reshape(coord_init_pix_scaleY[:, 1], [nx, ny])
            col_scaleYsmooth = gaussian(col_scaleY, smoothWarpXY)
            col_scaleYinvsmooth = gaussian(col_scaleYinv, smoothWarpXY)
            # apply smoothed transformation to image
            src2d_scaleXYsmooth = warp(src2d, np.array([row_scaleXinv, col_scaleYinvsmooth]), order=1)
            # reshape warping field as 1d
            coord_init_pix_scaleY[:, 1] = col_scaleYsmooth.ravel()
            coord_init_pix_scaleYinv[:, 1] = col_scaleYinvsmooth.ravel()
            # display
            if verbose == 2:
                # FIG 1
                plt.figure(figsize=(15, 3))
                # plot #1
                ax = plt.subplot(141)
                plt.imshow(np.swapaxes(src2d, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # plot #2
                ax = plt.subplot(142)
                plt.imshow(np.swapaxes(src2d_scaleX, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src_scaleX')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # plot #3
                ax = plt.subplot(143)
                plt.imshow(np.swapaxes(src2d_scaleXY, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src_scaleXY')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # plot #4
                ax = plt.subplot(144)
                plt.imshow(np.swapaxes(src2d_scaleXYsmooth, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src_scaleXYsmooth (s=' + str(smoothWarpXY) + ')')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # save figure
                plt.savefig(os.path.join(path_qc, 'register2d_columnwise_image_z' + str(iz) + '.png'))
                plt.close()

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
            warp_x[:, :, iz] = np.array([coord_init_phy_scaleXinv[i, 0] - coord_init_phy[i, 0] for i in xrange(nx * ny)]).reshape((nx, ny))
            warp_y[:, :, iz] = np.array([coord_init_phy_scaleYinv[i, 1] - coord_init_phy[i, 1] for i in xrange(nx * ny)]).reshape((nx, ny))
            # compute displacement per pixel in source space (for inverse warping field)
            warp_inv_x[:, :, iz] = np.array([coord_init_phy_scaleX[i, 0] - coord_init_phy[i, 0] for i in xrange(nx * ny)]).reshape((nx, ny))
            warp_inv_y[:, :, iz] = np.array([coord_init_phy_scaleY[i, 1] - coord_init_phy[i, 1] for i in xrange(nx * ny)]).reshape((nx, ny))

    # Generate forward warping field (defined in destination space)
    generate_warping_field(fname_dest, warp_x, warp_y, fname_warp, verbose)
    # Generate inverse warping field (defined in source space)
    generate_warping_field(fname_src, warp_inv_x, warp_inv_y, fname_warp_inv, verbose)


def register2d(fname_src, fname_dest, fname_mask='', fname_warp='warp_forward.nii.gz', fname_warp_inv='warp_inverse.nii.gz', paramreg=Paramreg(step='0', type='im', algo='Translation', metric='MI', iter='5', shrink='1', smooth='0', gradStep='0.5'),
                    ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '', 'bspline': ',10', 'gaussiandisplacementfield': ',3,0',
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
    sct.printv('.. matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
    sct.printv('.. voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', verbose)

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
        sct.printv('Registering slice ' + str(i) + '/' + str(nz - 1) + '...', verbose)
        num = numerotation(i)
        prefix_warp2d = 'warp2d_' + num
        # if mask is used, prepare command for ANTs
        if fname_mask != '':
            masking = '-x mask_Z' + num + '.nii.gz'
        else:
            masking = ''
        # main command for registration
        cmd = ('isct_antsRegistration '
               '--dimensionality 2 '
               '--transform ' + paramreg.algo + '[' + str(paramreg.gradStep) +
               ants_registration_params[paramreg.algo.lower()] + '] '
               '--metric ' + paramreg.metric + '[dest_Z' + num + '.nii' + ',src_Z' + num + '.nii' + ',1,' + metricSize + '] '  #[fixedImage,movingImage,metricWeight +nb_of_bins (MI) or radius (other)
               '--convergence ' + str(paramreg.iter) + ' '
               '--shrink-factors ' + str(paramreg.shrink) + ' '
               '--smoothing-sigmas ' + str(paramreg.smooth) + 'mm '
               '--output [' + prefix_warp2d + ',src_Z' + num + '_reg.nii] '    #--> file.mat (contains Tx,Ty, theta)
               '--interpolation BSpline[3] '
               '--verbose 1 '
               + masking)
        # add init translation
        if not paramreg.init == '':
            init_dict = {'geometric': '0', 'centermass': '1', 'origin': '2'}
            cmd += ' -r [dest_Z' + num + '.nii' + ',src_Z' + num + '.nii,' + init_dict[paramreg.init] + ']'

        try:
            # run registration
            sct.run(cmd)

            if paramreg.algo in ['Translation']:
                file_mat = prefix_warp2d + '0GenericAffine.mat'
                matfile = loadmat(file_mat, struct_as_record=True)
                array_transfo = matfile['AffineTransform_double_2_2']
                x_displacement[i] = array_transfo[4][0]  # Tx in ITK'S coordinate system
                y_displacement[i] = array_transfo[5][0]  # Ty  in ITK'S and fslview's coordinate systems
                theta_rotation[i] = asin(array_transfo[2])  # angle of rotation theta in ITK'S coordinate system (minus theta for fslview)

            if paramreg.algo in ['Rigid', 'Affine', 'BSplineSyN', 'SyN']:
                # List names of 2d warping fields for subsequent merge along Z
                file_warp2d = prefix_warp2d + '0Warp.nii.gz'
                file_warp2d_inv = prefix_warp2d + '0InverseWarp.nii.gz'
                list_warp.append(file_warp2d)
                list_warp_inv.append(file_warp2d_inv)

            if paramreg.algo in ['Rigid', 'Affine']:
                # Generating null 2d warping field (for subsequent concatenation with affine transformation)
                sct.run('isct_antsRegistration -d 2 -t SyN[1, 1, 1] -c 0 -m MI[dest_Z' + num + '.nii, src_Z' + num + '.nii, 1, 32] -o warp2d_null -f 1 -s 0')
                # --> outputs: warp2d_null0Warp.nii.gz, warp2d_null0InverseWarp.nii.gz
                file_mat = prefix_warp2d + '0GenericAffine.mat'
                # Concatenating mat transfo and null 2d warping field to obtain 2d warping field of affine transformation
                sct.run('isct_ComposeMultiTransform 2 ' + file_warp2d + ' -R dest_Z' + num + '.nii warp2d_null0Warp.nii.gz ' + file_mat)
                sct.run('isct_ComposeMultiTransform 2 ' + file_warp2d_inv + ' -R src_Z' + num + '.nii warp2d_null0InverseWarp.nii.gz -i ' + file_mat)

        # if an exception occurs with ants, take the last value for the transformation
        # TODO: DO WE NEED TO DO THAT??? (julien 2016-03-01)
        except Exception, e:
            sct.printv('ERROR: Exception occurred.\n' + str(e), 1, 'error')

    # Merge warping field along z
    sct.printv('\nMerge warping fields along z...', verbose)

    if paramreg.algo in ['Translation']:
        # convert to array
        x_disp_a = np.asarray(x_displacement)
        y_disp_a = np.asarray(y_displacement)
        theta_rot_a = np.asarray(theta_rotation)
        # Generate warping field
        generate_warping_field('dest.nii', x_disp_a, y_disp_a, fname_warp=fname_warp)  #name_warp= 'step'+str(paramreg.step)
        # Inverse warping field
        generate_warping_field('src.nii', -x_disp_a, -y_disp_a, fname_warp=fname_warp_inv)

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
        sct.log.error('ERROR: the number is negative.')
        sys.exit(status=2)
    elif -1 < nb < 10:
        nb_output = '000' + str(nb)
    elif 9 < nb < 100:
        nb_output = '00' + str(nb)
    elif 99 < nb < 1000:
        nb_output = '0' + str(nb)
    elif 999 < nb < 10000:
        nb_output = str(nb)
    elif nb > 9999:
        sct.log.error('ERROR: the number is superior to 9999.')
        sys.exit(status = 2)
    return nb_output


def generate_warping_field(fname_dest, warp_x, warp_y, fname_warp='warping_field.nii.gz', verbose=1):
    """
    Generate an ITK warping field
    :param fname_dest:
    :param warp_x:
    :param warp_y:
    :param fname_warp:
    :param verbose:
    :return:
    """
    sct.printv('\nGenerate warping field...', verbose)

    # Get image dimensions
    # sct.printv('Get destination dimension', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
    # sct.printv('  matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
    # sct.printv('  voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)

    # initialize
    data_warp = np.zeros((nx, ny, nz, 1, 3))

    # fill matrix
    data_warp[:, :, :, 0, 0] = -warp_x  # need to invert due to ITK conventions
    data_warp[:, :, :, 0, 1] = -warp_y  # need to invert due to ITK conventions

    # save warping field
    im_dest = load(fname_dest)
    hdr_dest = im_dest.get_header()
    hdr_warp = hdr_dest.copy()
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = Nifti1Image(data_warp, None, hdr_warp)
    save(img, fname_warp)
    sct.printv(' --> ' + fname_warp, verbose)

    #
    # file_dest = load(fname_dest)
    # hdr_file_dest = file_dest.get_header()
    # hdr_warp = hdr_file_dest.copy()
    #
    #
    # # Center of rotation
    # if center_rotation == None:
    #     x_a = int(round(nx/2))
    #     y_a = int(round(ny/2))
    # else:
    #     x_a = center_rotation[0]
    #     y_a = center_rotation[1]
    #
    # # Calculate displacement for each voxel
    # data_warp = np.zeros(((((nx, ny, nz, 1, 3)))))
    # vector_i = [[[i-x_a], [j-y_a]] for i in range(nx) for j in range(ny)]
    #
    # # if theta_rot == None:
    # #     # for translations
    # #     for k in range(nz):
    # #         matrix_rot_a = np.asarray([[cos(theta_rot[k]), - sin(theta_rot[k])], [-sin(theta_rot[k]), -cos(theta_rot[k])]])
    # #         tmp = matrix_rot_a + array(((-1, 0), (0, 1)))
    # #         result = dot(tmp, array(vector_i).T[0]) + array([[x_trans[k]], [y_trans[k]]])
    # #         for i in range(ny):
    # #             data_warp[i, :, k, 0, 0] = result[0][i*nx:i*nx+ny]
    # #             data_warp[i, :, k, 0, 1] = result[1][i*nx:i*nx+ny]
    #
    # # else:
    #     # For rigid transforms (not optimized)
    #     # if theta_rot != None:
    # # TODO: this is not optimized! can do better!
    # for k in range(nz):
    #     for i in range(nx):
    #         for j in range(ny):
    #             data_warp[i, j, k, 0, 0] = (cos(theta_rot[k]) - 1) * (i - x_a) - sin(theta_rot[k]) * (j - y_a) + x_trans[k]
    #             data_warp[i, j, k, 0, 1] = - sin(theta_rot[k]) * (i - x_a) - (cos(theta_rot[k]) - 1) * (j - y_a) + y_trans[k]
    #             data_warp[i, j, k, 0, 2] = 0
    #
    # # Generate warp file as a warping field
    # hdr_warp.set_intent('vector', (), '')
    # hdr_warp.set_data_dtype('float32')
    # img = Nifti1Image(data_warp, None, hdr_warp)
    # save(img, fname)
    # sct.printv('\nDone! Warping field generated: '+fname, verbose)


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
    arccosInput = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    # sct.printv(arccosInput)
    arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
    arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
    sign_angle = np.sign(np.cross(a, b))
    # sct.printv(sign_angle)
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
    xmin = i - 1 + (0.5 - data1d[i - 1]) / float(data1d[i] - data1d[i - 1])
    # continue for the descending slope
    for i in range(i, len(data1d)):
        if data1d[i] < 0.5:
            break
    # compute center of mass to get coordinate at 0.5
    xmax = i - 1 + (0.5 - data1d[i - 1]) / float(data1d[i] - data1d[i - 1])
    # display
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(src1d)
    # plt.plot(xmin, 0.5, 'o')
    # plt.plot(xmax, 0.5, 'o')
    # plt.savefig('./normalize1d.png')
    return xmin, xmax
