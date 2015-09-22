#!/usr/bin/env python
########################################################################################################################
#
#
# Utility functions used for the segmentation of the gray matter
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# Modified: 2015-03-24
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

from math import sqrt

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# import time
# from scipy.optimize import minimize

from msct_image import Image
import sct_utils as sct
from msct_parser import Parser


class Slice:
    """
    Slice instance used in the model dictionary for the segmentation of the gray matter
    """
    def __init__(self, slice_id=None, im=None, sc_seg=None, gm_seg=None, wm_seg=None, reg_to_m=None, im_m=None, gm_seg_m=None, wm_seg_m=None, im_m_flat=None, gm_seg_m_flat=None, wm_seg_m_flat=None, level=None):
        """
        Slice constructor

        :param slice_id: slice ID number, type: int

        :param im: original image (a T2star 2D image croped around the spinal cord), type: numpy array

        :param gm_seg: manual gray matter segmentation of the original image, type: numpy array

        :param wm_seg: manual white matter segmentation of the original image, type: numpy array

        :param reg_to_m: name of the file containing the transformation for this slice to go from the image original space to the model space, type: string

        :param im_m: image in the model space, type: numpy array

        :param gm_seg_m: manual gray matter segmentation in the model space, type: numpy array

        :param wm_seg_m: manual white matter segmentation in the model space, type: numpy array

        :param im_m_flat: flatten image in the model space, type: numpy array

        :param gm_seg_m_flat: flatten manual gray matter  segmentation in the model space, type: numpy array

        :param wm_seg_m_flat: flatten manual white matter segmentation in the model space, type: numpy array

        :param level: vertebral level of the slice, type: int
        """
        self.id = slice_id
        self.im = im
        self.sc_seg = sc_seg
        self.gm_seg = gm_seg
        self.wm_seg = wm_seg
        self.reg_to_M = reg_to_m
        self.im_M = im_m
        self.gm_seg_M = gm_seg_m
        self.wm_seg_M = wm_seg_m
        self.im_M_flat = im_m_flat
        self.gm_seg_M_flat = gm_seg_m_flat
        self.wm_seg_M_flat = wm_seg_m_flat
        self.level = level

    def set(self, slice_id=None, im=None, sc_seg=None, gm_seg=None, wm_seg=None, reg_to_m=None, im_m=None, gm_seg_m=None, wm_seg_m=None, im_m_flat=None, gm_seg_m_flat=None, wm_seg_m_flat=None, level=None):
        """
        Slice setter, only the specified parameters are set

        :param slice_id: slice ID number, type: int

        :param im: original image (a T2star 2D image croped around the spinal cord), type: numpy array

        :param gm_seg: manual gray matter segmentation of the original image, type: numpy array

        :param wm_seg: manual white matter segmentation of the original image, type: numpy array

        :param reg_to_m: name of the file containing the transformation for this slice to go from the image original space to the model space, type: string

        :param im_m: image in the model space, type: numpy array

        :param gm_seg_m: manual gray matter segmentation in the model space, type: numpy array

        :param wm_seg_m: manual white matter segmentation in the model space, type: numpy array

        :param im_m_flat: flatten image in the model space, type: numpy array

        :param gm_seg_m_flat: flatten manual gray matter  segmentation in the model space, type: numpy array

        :param wm_seg_m_flat: flatten manual white matter segmentation in the model space, type: numpy array

        :param level: vertebral level of the slice, type: int
        """
        if slice_id is not None:
            self.id = slice_id
        if im is not None:
            self.im = im
        if sc_seg is not None:
            self.sc_seg = sc_seg
        if gm_seg is not None:
            self.gm_seg = gm_seg
        if wm_seg is not None:
            self.wm_seg = wm_seg
        if reg_to_m is not None:
            self.reg_to_M = reg_to_m
        if im_m is not None:
            self.im_M = im_m
        if gm_seg_m is not None:
            self.gm_seg_M = gm_seg_m
        if wm_seg_m is not None:
            self.wm_seg_M = wm_seg_m
        if im_m_flat is not None:
            self.im_M_flat = im_m_flat
        if gm_seg_m_flat is not None:
            self.gm_seg_M_flat = gm_seg_m_flat
        if wm_seg_m_flat is not None:
            self.wm_seg_M_flat = wm_seg_m_flat
        if level is not None:
            self.level = level

    def __repr__(self):
        s = '\nSlice #' + str(self.id)
        if self.level is not None:
            s += 'Level : ' + str(self.level)
        s += '\nAtlas : \n' + str(self.im) + '\nDecision : \n' + str(self.gm_seg) +\
             '\nTransformation to model space : ' + self.reg_to_M
        if self.im_M is not None:
            s += '\nAtlas in the common model space: \n' + str(self.im_M)
        if self.gm_seg_M is not None:
            s += '\nDecision in the common model space: \n' + str(self.gm_seg_M)
        return s


########################################################################################################################
# ---------------------------------------------------- FUNCTIONS ----------------------------------------------------- #
########################################################################################################################

'''
# ----------------------------------------------------------------------------------------------------------------------
def split(slice):
    """
    Split a given slice in two

    :param slice: slice to split, type: numpy array
    :return:
    """
    left_slice = []
    right_slice = []
    column_length = slice.shape[1]
    i = 0
    for column in slice:
        if i < column_length / 2:
            left_slice.append(column)
        else:
            right_slice.insert(0, column)
        i += 1
    left_slice = np.asarray(left_slice)
    right_slice = np.asarray(right_slice)
    assert (left_slice.shape == right_slice.shape), \
        str(left_slice.shape) + '==' + str(right_slice.shape) + \
        'You should check that the first dim of your image (or slice) is an odd number'
    return left_slice, right_slice
'''


# ----------------------------------------------------------------------------------------------------------------------
def show(coord_projected_img, pca, target):
    """
    Display an image projected in the reduced PCA space

    :param coord_projected_img: Coordinates of the target image in the PCA reduced space

    :param pca: PCA instance (used to get the mean image)

    :param target: image to display
    """
    import copy

    img_reduced_space = copy.copy(pca.mean_image)
    for i in range(0, coord_projected_img.shape[0]):
        img_reduced_space += int(coord_projected_img[i][0]) * pca.kept_modes.T[i].reshape(pca.N, 1)

    n = int(sqrt(pca.N))
    imgplot = plt.imshow(pca.mean_image)
    imgplot.set_interpolation('nearest')
    imgplot.set_cmap('gray')
    plt.title('Mean PCA Image')
    plt.show()

    imgplot = plt.imshow(target.reshape(n, n))
    imgplot.set_interpolation('nearest')
    # imgplot.set_cmap('gray')
    plt.title('Original Image')
    plt.show()

    imgplot = plt.imshow(img_reduced_space.reshape(n, n))
    imgplot.set_interpolation('nearest')
    # imgplot.set_cmap('gray')
    plt.title('Projected Image (projected in the PCA reduced space)')
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
def save_image(im_array, im_name, path='', im_type='', verbose=1):
    """
    Save an image from an array,
    if the array is a flatten image, the saved image will be square shaped

    :param im_array: image to save

    :param im_name: name of the image

    :param path: path where to save it

    :param im_type: type of image

    :param verbose:
    """
    if isinstance(im_array, list):
        n = int(sqrt(len(im_array)))
        im_data = np.asarray(im_array).reshape(n, n)
    else:
        im_data = np.asarray(im_array)
    im = Image(param=im_data, verbose=verbose)
    im.file_name = im_name
    im.ext = '.nii.gz'
    if path != '':
        im.path = path
    im.save(type=im_type)


# ----------------------------------------------------------------------------------------------------------------------
def apply_ants_transfo(fixed_im, moving_im, search_reg=True, transfo_type='Rigid', apply_transfo=True, transfo_name='', binary=True, path='./', inverse=0, verbose=0):
    """
    Compute and/or apply a registration using ANTs

    :param fixed_im: fixed image for the registration, type: numpy array

    :param moving_im: moving image for the registration, type: numpy array

    :param search_reg: compute (search) or load (from a file) the transformation to do, type: boolean

    :param transfo_type: type of transformation to apply, type: string

    :param apply_transfo: apply or not the transformation, type: boolean

    :param transfo_name: name of the file containing the transformation information (to load or save the transformation, type: string

    :param binary: if the image to register is binary, type: boolean

    :param path: path where to load/save the transformation

    :param inverse: apply inverse transformation

    :param verbose: verbose
    """
    import time
    res_im = None
    try:
        transfo_dir = transfo_type.lower() + '_transformations'
        if transfo_dir not in os.listdir(path):
            sct.run('mkdir ' + path + transfo_dir)
        dir_name = 'tmp_reg_' + time.strftime("%y%m%d%H%M%S") + '_' + str(time.time())
        sct.run('mkdir ' + dir_name, verbose=verbose)
        os.chdir('./' + dir_name)

        if binary:
            t = 'uint8'
        else:
            t = ''

        fixed_im_name = 'fixed_im'
        save_image(fixed_im, fixed_im_name, im_type=t, verbose=verbose)
        moving_im_name = 'moving_im'
        save_image(moving_im, moving_im_name, im_type=t, verbose=verbose)

        mat_name, inverse_mat_name = find_ants_transfo_name(transfo_type)

        if search_reg:
            reg_interpolation = 'BSpline'
            transfo_params = ''
            if transfo_type == 'BSpline':
                transfo_params = ',1'
            elif transfo_type == 'BSplineSyN':
                transfo_params = ',3,0'
                reg_interpolation = 'NearestNeighbor'
            elif transfo_type == 'SyN':
                transfo_params = ',1,1'
            gradientstep = 0.5  # 0.3
            metric = 'MeanSquares'
            metric_params = ',1,4'  # ',5'
            # metric = 'MI'
            # metric_params = ',1,2'
            niter = 5  # 20
            smooth = 0
            shrink = 1
            cmd_reg = 'antsRegistration -d 2 -n ' + reg_interpolation + ' -t ' + transfo_type + '[' + str(gradientstep) + transfo_params + '] ' \
                      '-m ' + metric + '[' + fixed_im_name + '.nii.gz,' + moving_im_name + '.nii.gz ' + metric_params + '] -o reg  -c ' + str(niter) + \
                      ' -s ' + str(smooth) + ' -f ' + str(shrink) + ' -v ' + str(verbose)  # + ' -r [' + fixed_im_name + '.nii.gz,' + moving_im_name + '.nii.gz ' + ',1]'

            sct.run(cmd_reg, verbose=verbose)

            sct.run('cp ' + mat_name + ' ../' + path + transfo_dir + '/'+transfo_name, verbose=verbose)
            if 'SyN' in transfo_type:
                sct.run('cp ' + inverse_mat_name + ' ../' + path + transfo_dir + '/'+transfo_name + '_inversed',
                        verbose=verbose)

        if apply_transfo:
            if not search_reg:
                sct.run('cp ../' + path + transfo_dir + '/' + transfo_name + ' ./' + mat_name, verbose=verbose)
                if 'SyN' in transfo_type:
                    sct.run('cp ../' + path + transfo_dir + '/' + transfo_name + '_inversed' + ' ./' + inverse_mat_name,
                            verbose=verbose)

            if binary or moving_im.max() == 1 or fixed_im.max() == 1:
                apply_transfo_interpolation = 'NearestNeighbor'
            else:
                apply_transfo_interpolation = 'BSpline'

            if 'SyN' in transfo_type and inverse:
                cmd_apply = 'antsApplyTransforms -d 2 -i ' + moving_im_name + '.nii.gz -o ' + moving_im_name + '_moved.nii.gz ' \
                            '-n ' + apply_transfo_interpolation + ' -t [' + inverse_mat_name + '] ' \
                            '-r ' + fixed_im_name + '.nii.gz -v ' + str(verbose)

            else:
                cmd_apply = 'antsApplyTransforms -d 2 -i ' + moving_im_name + '.nii.gz -o ' + moving_im_name + '_moved.nii.gz ' \
                            '-n ' + apply_transfo_interpolation + ' -t [' + mat_name + ',' + str(inverse) + '] ' \
                            '-r ' + fixed_im_name + '.nii.gz -v ' + str(verbose)

            sct.run(cmd_apply, verbose=verbose)

            res_im = Image(moving_im_name + '_moved.nii.gz')
    except Exception, e:
        sct.printv('WARNING: AN ERROR OCCURRED WHEN DOING RIGID REGISTRATION USING ANTs', 1, 'warning')
        print e
    else:
        sct.printv('Removing temporary files ...', verbose=verbose, type='normal')
        os.chdir('..')
        sct.run('rm -rf ' + dir_name + '/', verbose=verbose)

    if apply_transfo and res_im is not None:
        return res_im.data


# ----------------------------------------------------------------------------------------------------------------------
def find_ants_transfo_name(transfo_type):
    """
    find the name of the transformation file automatically saved by ANTs for a type of transformation

    :param transfo_type: type of transformation

    :return transfo_name, inverse_transfo_name:
    """
    transfo_name = ''
    inverse_transfo_name = ''
    if transfo_type == 'Rigid' or transfo_type == 'Affine':
        transfo_name = 'reg0GenericAffine.mat'
    elif transfo_type == 'BSpline':
        transfo_name = 'reg0BSpline.txt'
    elif transfo_type == 'BSplineSyN' or transfo_type == 'SyN':
        transfo_name = 'reg0Warp.nii.gz'
        inverse_transfo_name = 'reg0InverseWarp.nii.gz'
    return transfo_name, inverse_transfo_name


# ######################  DEVELOPING GROUPWISE  #######################
# ----------------------------------------------------------------------------------------------------------------------
def apply_2d_transformation(matrix, tx=0, ty=0, theta=0, s=1, transfo=None, inverse=False):
    """
    Apply an Affine transformation defined a translation, a rotation and a scaling

    :param matrix: image to apply transformation on, type: numpy ndarray

    :param tx: translation along the x-axis, type: float

    :param ty: translation along the y-axis, type: float

    :param theta: rotation angle in counter-clockwise direction as radians, type: float

    :param s: scaling factor, type: float

    :return transformed_mat, transfo: transformed matrix, transformation used
    """
    from skimage import transform as tf
    if transfo is None:
        # transfo = tf.AffineTransform(scale=[scx, scy], rotation=theta, translation=(tx, ty))
        transfo = tf.SimilarityTransform(scale=s, rotation=theta, translation=(tx, ty))

    if inverse:
        transformed_mat = tf.warp(matrix.astype('uint32'), transfo.inverse, preserve_range=True)
    else:
        transformed_mat = tf.warp(matrix.astype('uint32'), transfo, preserve_range=True)

    return transformed_mat, transfo
# ######################################################################

'''
# ----------------------------------------------------------------------------------------------------------------------
def kronecker_delta(x, y):
    """
    Kronecker delta function
    :param x: float
    :param y: float
    :return: 0 if x=y, 1 if x and y are different
    """
    if x == y:
        return 1
    else:
        return 0
'''

'''
# ------------------------------------------------------------------------------------------------------------------
def compute_mutual_information_seg(seg_data_set):
    """
    Compute the mean segmentation image for a given segmentation data set
    keeping only the pixels that are present in ALL the data set
    :param seg_data_set:
    :return:
    """
    seg_sum = np.asarray(seg_data_set[0])
    n_seg = len(seg_data_set)

    for dic_slice in seg_data_set[1:]:
        seg_sum += np.asarray(dic_slice)

    index_to_keep = seg_sum == n_seg
    mean_seg = index_to_keep.astype(int)

    return mean_seg
'''


# ------------------------------------------------------------------------------------------------------------------
def l0_norm(x, y):
    """
    L0 norm of two images x and y
    :param x:
    :param y:
    :return: l0 norm
    """
    return np.linalg.norm(x.flatten() - y.flatten(), 0)


# ------------------------------------------------------------------------------------------------------------------
def inverse_wmseg_to_gmseg(wm_seg, sc_seg, name_wm_seg, hdr):
    """
    Inverse a white matter segmentation array image to get a gray matter segmentation image and save it

    :param wm_seg: white matter segmentation to inverse, type: Image

    :param original_im: original image croped around the spinal cord

    :param name_wm_seg: name of the white matter segmentation (to save the associated gray matter segmentation),
     type: string

    :return inverted_seg: gray matter segmentation image
    """
    import time
    import skimage.morphology as skm

    inverted_seg = []
    if len(sc_seg.data.shape) == 3 and sc_seg.data.shape[2] != 1:
        dim = 3
    else:
        dim = 2

    tmp_dir = 'tmp_' + time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir ' + tmp_dir)
    os.chdir(tmp_dir)

    if dim == 3:
        for seg_slice, sc_slice in zip(wm_seg.data, sc_seg.data):
            inverted_seg.append(sc_slice-seg_slice)
    elif dim == 2:
        inverted_seg = sc_seg.data - wm_seg.data

    inverted_seg = skm.remove_small_objects(np.asarray(inverted_seg).astype('bool'), 4).astype(int)

    os.chdir('..')

    inverted_seg_im = Image(param=inverted_seg)
    inverted_seg_im.file_name = name_wm_seg + '_inv_to_gm'
    inverted_seg_im.ext = '.nii.gz'
    inverted_seg_im.hdr = hdr
    inverted_seg_im.save()

    sct.run('rm -rf ' + tmp_dir)

    return inverted_seg_im

'''
# ------------------------------------------------------------------------------------------------------------------
def inverse_wmseg_to_gmseg_version3(wm_seg, original_im, name_wm_seg): # WORKS ALMOST OK BUT NOT PERFECT
    """
    Inverse a white matter segmentation array image to get a gray matter segmentation image and save it

    :param wm_seg: white matter segmentation to inverse, type: Image

    :param original_im: original image croped around the spinal cord

    :param name_wm_seg: name of the white matter segmentation (to save the associated gray matter segmentation),
     type: string

    :return inverted_seg: gray matter segmentation image
    """
    import time
    import skimage.morphology as skm

    inverted_seg = []
    sc = Image(param=original_im.data)
    if len(sc.data.shape) == 3 and sc.data.shape[2] != 1:
        dim = 3
    else:
        dim = 2

    tmp_dir = 'tmp_' + str(time.time())
    sct.run('mkdir ' + tmp_dir)
    os.chdir(tmp_dir)

    nz_coord_sc = sc.getNonZeroCoordinates()
    for coord in nz_coord_sc:
        if dim == 3:
            sc.data[coord.x, coord.y, coord.z] = 1
        elif dim == 2:
            sc.data[coord.x, coord.y] = 1
    sc.file_name = 'original_target_sc'
    sc.ext = '.nii.gz'
    sc.save()

    sct.run(' cp ../' + wm_seg.file_name + '.nii.gz ./res.nii.gz')


    sct.run('fslmaths res.nii.gz -kernel box 3 -dilM dilated_res.nii.gz')
    sct.run('fslmaths dilated_res.nii.gz -kernel box 4 -ero closed_res.nii.gz')

    sct.run('fslmaths original_target_sc.nii.gz -kernel box 3 -ero eroded_sc.nii.gz')
    # sct.run('fslmaths eroded_sc.nii.gz -ero eroded_sc.nii.gz')

    eroded_sc = Image('eroded_sc.nii.gz')

    new_sc = Image('closed_res.nii.gz')

    res = Image('res.nii.gz')

    new_sc.data = ((res.data + new_sc.data + eroded_sc.data) > 0).astype(int)  # ((eroded_sc.data + new_sc.data) > 0).astype(int)
    new_sc.file_name = 'new_sc'
    new_sc.save()

    if dim == 3:
        for seg_slice, sc_slice in zip(wm_seg.data, new_sc.data):
            inverted_seg.append(sc_slice-seg_slice)
    elif dim == 2:
        inverted_seg = new_sc.data - wm_seg.data

    inverted_seg = skm.remove_small_objects(np.asarray(inverted_seg).astype('bool'), 7).astype(int)

    # inverted_seg_pos = (np.asarray(inverted_seg) > 0).astype(int)

    os.chdir('..')

    inverted_seg_im = Image(param=inverted_seg)
    inverted_seg_im.file_name = name_wm_seg + '_inv_to_gm'
    inverted_seg_im.ext = '.nii.gz'
    inverted_seg_im.save()

    sct.run('rm -rf ' + tmp_dir)

    return inverted_seg_im


# ------------------------------------------------------------------------------------------------------------------
def inverse_wmseg_to_gmseg_version2(wm_seg, original_im, name_wm_seg):
    """
    Inverse a white matter segmentation array image to get a gray matter segmentation image and save it

    :param wm_seg: white matter segmentation to inverse, type: Image

    :param original_im: original image croped around the spinal cord

    :param name_wm_seg: name of the white matter segmentation (to save the associated gray matter segmentation),
     type: string

    :return inverted_seg: gray matter segmentation image
    """
    import time

    inverted_seg = []
    sc = Image(param=original_im.data)
    if len(sc.data.shape) == 3 and sc.data.shape[2] != 1:
        dim = 3
    else:
        dim = 2

    tmp_dir = 'tmp_' + str(time.time())
    sct.run('mkdir ' + tmp_dir)
    os.chdir(tmp_dir)

    nz_coord_sc = sc.getNonZeroCoordinates()
    for coord in nz_coord_sc:
        if dim == 3:
            sc.data[coord.x, coord.y, coord.z] = 1
        elif dim == 2:
            sc.data[coord.x, coord.y] = 1
    sc.file_name = 'original_target_sc'
    sc.ext = '.nii.gz'
    sc.save()

    sct.run(' cp ../' + wm_seg.file_name + '.nii.gz ./res.nii.gz')


    sct.run('fslmaths res.nii.gz -binv inverted_res.nii.gz')
    sct.run('fslmaths inverted_res.nii.gz -kernel  2D -dilD dilated_res.nii.gz')

    dilated_res = Image('dilated_res.nii.gz')

    inv_res = Image('inverted_res.nii.gz')

    inverted_seg = ((dilated_res.data - inv_res.data) > 0).astype(int)

    os.chdir('..')

    inverted_seg_pos_image = Image(param=np.asarray(inverted_seg))
    inverted_seg_pos_image.file_name = name_wm_seg + '_inv_to_gm'
    inverted_seg_pos_image.ext = '.nii.gz'
    inverted_seg_pos_image.save()

    # sct.run('rm -rf ' + tmp_dir)

    return inverted_seg_pos_image
'''

# ------------------------------------------------------------------------------------------------------------------
def inverse_gmseg_to_wmseg(gm_seg, original_im, name_gm_seg):
    """
    Inverse a gray matter segmentation array image to get a white matter segmentation image and save it

    :param gm_seg: gray matter segmentation to inverse, type: Image

    :param original_im: original image croped around the spinal cord

    :param name_gm_seg: name of the gray matter segmentation (to save the associated white matter segmentation),
     type: string

    :return inverted_seg: white matter segmentation image
    """
    assert gm_seg.data.shape == original_im.data.shape

    original_hdr = gm_seg.hdr

    gm_seg_copy = gm_seg.copy()
    sc = original_im.copy()
    nz_coord_sc = sc.getNonZeroCoordinates()
    nz_coord_seg = gm_seg.getNonZeroCoordinates()
    if len(gm_seg_copy.data.shape) == 3:
        for coord in nz_coord_sc:
            sc.data[coord.x, coord.y, coord.z] = 1
        for coord in nz_coord_seg:
            gm_seg_copy.data[coord.x, coord.y, coord.z] = 1
    elif len(gm_seg_copy.data.shape) == 2:
        for coord in nz_coord_sc:
            sc.data[coord.x, coord.y] = 1
        for coord in nz_coord_seg:
            gm_seg_copy.data[coord.x, coord.y] = 1

    # cast of the -1 values (-> GM pixel at the exterior of the SC pixels) to +1 --> WM pixel
    res_wm_seg = np.absolute(sc.data - gm_seg_copy.data).astype(int)

    res_wm_seg_im = Image(param=np.asarray(res_wm_seg), absolutepath=name_gm_seg + '_inv_to_wm.nii.gz')
    res_wm_seg_im.hdr = original_hdr
    res_wm_seg_im.save()

    return res_wm_seg_im


# ------------------------------------------------------------------------------------------------------------------
def correct_wmseg(res_gmseg, original_im, name_wm_seg, hdr):
    wmseg_dat = (original_im.data > 0).astype(int) - res_gmseg.data

    # corrected_wm_seg = Image(param=(wmseg_dat > 0).astype(int))
    corrected_wm_seg = Image(param=wmseg_dat)

    corrected_wm_seg.file_name = name_wm_seg + '_corrected'
    corrected_wm_seg.ext = '.nii.gz'
    corrected_wm_seg.hdr = hdr
    corrected_wm_seg.save()

    sct.run('fslmaths ' + corrected_wm_seg.file_name + '.nii.gz -thr 0 ' + corrected_wm_seg.file_name + '.nii.gz')

    return corrected_wm_seg


# ------------------------------------------------------------------------------------------------------------------
def compute_majority_vote_mean_seg(seg_data_set, threshold=0.5, type='binary'):
    """
    Compute the mean segmentation image for a given segmentation data set seg_data_set by Majority Vote

    :param seg_data_set: data set of segmentation slices (2D)

    :param threshold: threshold to select the value of a pixel
    :return:
    """
    if type == 'binary':
        return (np.sum(seg_data_set, axis=0) / float(len(seg_data_set)) >= threshold).astype(int)
    else:
        return (np.sum(seg_data_set, axis=0) / float(len(seg_data_set))).astype(float)

# ------------------------------------------------------------------------------------------------------------------
def get_key_from_val(dic, val):
    """
    inversed dictionary getter

    :param dic: dictionary
    :param val: value
    :return k: associated key
    """
    for k, v in dic.items():
        if v == val:
            return k


# ------------------------------------------------------------------------------------------------------------------
def check_file_to_niigz(file_name, verbose=1):
    if os.path.isfile(file_name):
        if sct.extract_fname(file_name)[2] != '.nii.gz':
            sct.run('fslchfiletype NIFTI_GZ ' + file_name)
            file_name = sct.extract_fname(file_name)[1] + '.nii.gz'
        return file_name
    else:
        sct.printv('WARNING: ' + file_name + ' is not a file ...', verbose, 'warning')
        return False
########################################################################################################################
# -------------------------------------------------- PRETREATMENTS --------------------------------------------------- #
########################################################################################################################


# ------------------------------------------------------------------------------------------------------------------
def crop_t2_star_pipeline(path, box_size=45):
    """
    Pretreatment pipeline croping the t2star file around the spinal cord and both croped t2star and gray matter manual
    segmentation to a 45*45 squared image

    :param path: path to the data
    """

    for subject_dir in os.listdir(path):
        if os.path.isdir(path + subject_dir + '/'):
            t2star = ''
            sc_seg = ''
            seg_in = ''
            seg_in_name = ''
            manual_seg = ''
            manual_seg_name = ''
            mask_box = ''
            seg_in_croped = ''
            manual_seg_croped = ''

            '''
            # VERSION 1 OF THE PRE TREATMENTS

            mask_centerline = ''
            centerline = ''
            mask_box = ''
            croped = ''
            print subject_dir
            for file in os.listdir(dir + '/' + subject_dir):
                if 't2star.nii' in file and 'mask' not in file:
                    t2star = file
                elif '-mask_centerline.nii' in file:
                    mask_centerline = file
                elif 'mask_' in file:
                    mask_box = file
                elif '_centerline.nii' in file:
                    centerline = file
                elif '_croped' in file :
                    croped = file
            if t2star != '' and mask_centerline != '':
                path = dir + '/' + subject_dir + '/'
                print 'path : ', path
                os.chdir(path)
                t2star_path,t2star_name,ext = sct.extract_fname(t2star)
                if centerline == '':
                    sct.run('sct_get_centerline -i ' + t2star + ' -p '  + mask_centerline)
                if mask_box == '':
                    sct.run('sct_create_mask -i '  + t2star + ' -m centerline,'  + centerline +' -s 40 -f box' )
                if croped == '':
                    sct.run('sct_crop_image -i '  + t2star + ' -o '  + t2star_name + '_croped' + ext+
                    ' -m mask_' + t2star)
                os.chdir('..')
            '''

            '''
            # VERSION 2 OF THE PRE TREATMENTS



            for file in os.listdir(dir + '/' + subject_dir):
                if 't2star.nii' in file and 'mask' not in file and 'seg' not in file and 'IRP' not in file:
                    t2star = file
                    t2star_path,t2star_name,ext = sct.extract_fname(t2star)
                elif 'square_mask' in file and 'IRP' not in file:
                    mask_box = file
                elif '_seg' in file and 'in' not in file and 'croped' not in file and 'IRP' not in file:
                    sc_seg = file
                elif '_seg_in' in file and 'croped' not in file and 'IRP' not in file:
                    seg_in = file
                    seg_in_name = sct.extract_fname(seg_in)[1]
                elif '_croped' in file and 'IRP' not in file:
                    seg_in_croped = file
            if t2star != '' and sc_seg != '':
                path = dir + '/' + subject_dir + '/'
                print 'path : ', path
                os.chdir(path)

                try:

                    if seg_in == '':
                        sct.run('sct_crop_over_mask.py -i ' + t2star + ' -mask ' + sc_seg + ' -square 0 -o '
                        + t2star_name + '_seg_in')
                        seg_in = t2star_name + '_seg_in.nii.gz'
                        seg_in_name = t2star_name + '_seg_in'
                    if mask_box == '':
                        sct.run('sct_create_mask -i ' + t2star + ' -m center -s 70 -o '
                         + t2star_name + '_square_mask.nii.gz -f box' )
                        mask_box = t2star_name + '_square_mask.nii.gz'
                    if seg_in_croped == '':
                        sct.run('sct_crop_over_mask.py -i ' + seg_in + ' -mask ' + mask_box + ' -square 1 -o '
                        + seg_in_name + '_croped')
                    #os.chdir('..')

                except Exception,e:
                    sct.printv('WARNING: an error occured ... \n ' + str(e) ,1, 'warning')
                else:
                    print 'Done !'
                    #sct.run('rm -rf ./tmp_' + now)
                os.chdir('..')
            '''

            # VERSION 3 OF THE PRE TREATMENTS
            for subject_file in os.listdir(path + '/' + subject_dir):
                file_low = subject_file.lower()
                #todo change back im to t2star
                if 't2star' in file_low  or 'im' in file_low  in file_low and 'mask' not in file_low and 'seg' not in file_low and 'IRP' not in file_low:
                    t2star = subject_file
                    t2star_path, t2star_name, ext = sct.extract_fname(t2star)
                elif 'square' in file_low and 'mask' in file_low and 'IRP' not in file_low:
                    mask_box = subject_file
                elif 'seg' in file_low and 'in' not in file_low and 'croped' not in file_low and 'gm' not in file_low and 'IRP' not in file_low:
                    sc_seg = subject_file
                    '''
                    elif '_seg_in' in file_low and 'croped' not in file_low and 'IRP' not in file_low:
                        seg_in = subject_file
                        seg_in_name = sct.extract_fname(seg_in)[1]
                    '''
                elif 'gm' in file_low and 'croped.nii' not in file_low and 'IRP' not in file_low:
                    manual_seg = subject_file
                    manual_seg_name = sct.extract_fname(manual_seg)[1]
                    '''
                    elif '_croped.nii' in file_low and 'IRP' not in file_low and 'gm' not in file_low:
                        seg_in_croped = subject_file
                    '''
                elif '_croped.nii' in file_low and 'gm' in file_low and 'IRP' not in file_low:
                    manual_seg_croped = subject_file

            print 't2star', t2star
            print sc_seg

            if t2star != '' and sc_seg != '':
                subject_path = path + '/' + subject_dir + '/'
                print 'path : ', subject_path
                os.chdir(subject_path)
                '''
                now = str(time.time())
                sct.run('mkdir tmp_'+ now)
                sct.run('cp ./' + t2star + ' ./tmp_'+now+'/'+t2star)
                sct.run('cp ./' + sc_seg + ' ./tmp_'+now+'/'+sc_seg)
                os.chdir('./tmp_'+now)
                '''
                try:

                    if seg_in == '':
                        sct.run('sct_crop_over_mask.py -i ' + t2star + ' -mask ' + sc_seg + ' -square 0 '
                                '-o ' + t2star_name + '_seg_in')
                        seg_in = t2star_name + '_seg_in.nii.gz'
                        seg_in_name = t2star_name + '_seg_in'

                    if mask_box == '':

                        sct.run('sct_create_mask -i ' + seg_in + ' -m centerline,' + sc_seg + ' -s ' + str(box_size - 2) + ' -o ' + t2star_name + '_square_mask_from_sc_seg.nii.gz -f box')
                        mask_box = t2star_name + '_square_mask_from_sc_seg.nii.gz'

                    if seg_in_croped == '':
                        sct.run('sct_crop_over_mask.py -i ' + seg_in + ' -mask ' + mask_box + ' -square 1 '
                                '-o ' + seg_in_name + '_croped')

                    if manual_seg_croped == '':
                        sct.run('sct_crop_over_mask.py -i ' + manual_seg + ' -mask ' + mask_box + ' -square 1'
                                ' -o ' + manual_seg_name + '_croped')

                    # os.chdir('..')

                except Exception, e:
                    sct.printv('WARNING: an error occured ... \n ' + str(e), 1, 'warning')
                else:
                    print 'Done !'
                    # sct.run('rm -rf ./tmp_' + now)
                os.chdir('..')


# ------------------------------------------------------------------------------------------------------------------
def crop_t2_star_by_slice(path, box_size=45):
    """
    ###################################################################################################################################################
    DOES NOT WORK FOR NOW
    Pretreatment pipeline croping the t2star file around the spinal cord and both croped t2star and gray matter manual
    segmentation to a 45*45 squared image for AMU data

    :param path: path to the data
    """

    for subject_dir in os.listdir(path):
        if os.path.isdir(path + '/' + subject_dir):
            os.chdir(path)
            for subject_file in os.listdir(subject_dir):
                file_low = subject_file.lower()
                t2star = ''
                sc_seg = ''
                seg_in = ''
                seg_in_name = ''
                manual_seg = ''
                manual_seg_name = ''
                mask_box = ''
                seg_in_croped = ''
                manual_seg_croped = ''
                ext = ''

                if 'manual' not in file_low and 'seg' not in file_low and 'mask' not in file_low \
                        and 'ds_store' not in file_low:
                    t2star = subject_file
                    t2star_path, t2star_name, ext = sct.extract_fname(t2star)

                    sc_seg = t2star_name + '_manual_sc_seg.nii.gz'
                    manual_seg = t2star_name + '_manual_gm_seg.nii.gz'
                    manual_seg_name = t2star_name + '_manual_gm_seg'
                if t2star != '' and sc_seg != '':
                    subject_path = path + '/' + subject_dir + '/'
                    print 'path : ', subject_path
                    os.chdir(subject_dir)

                    try:

                        if seg_in == '':
                            sct.run('sct_crop_over_mask.py -i ' + t2star + ' -mask ' + sc_seg + ' -square 0 '
                                    '-o ' + t2star_name + '_seg_in')
                            seg_in = t2star_name + '_seg_in' + ext
                            seg_in_name = t2star_name + '_seg_in'

                        if mask_box == '':
                            '''
                            status, seg_in_orientation = sct.run('sct_orientation -i ' + seg_in)
                            seg_in_orientation = seg_in_orientation[4:7]
                            if seg_in_orientation != 'RPI':

                            sct.run('sct_orientation -i ' + seg_in + ' -s RAI ')
                            seg_in = seg_in_name + '_RAI' + ext

                            status, sc_seg_orientation = sct.run('sct_orientation -i ' + sc_seg)
                            sc_seg_orientation = sc_seg_orientation[4:7]
                            if sc_seg_orientation != 'RPI':

                            sct.run('sct_orientation -i ' + sc_seg + ' -s RPI ')
                            sc_seg = sct.extract_fname(sc_seg)[1] + '_RPI.nii.gz'
                            sct.run('sct_orientation -i ' + manual_seg + ' -s RPI ')
                            manual_seg = manual_seg_name + '_RPI.nii.gz'
                            '''
                            sct.run('sct_create_mask -i ' + manual_seg + ' -m centerline,' + sc_seg + ' -s ' + str(box_size - 2) + ' '
                                    '-o ' + t2star_name + '_square_mask_from_sc_seg.nii.gz -f box')
                            mask_box = t2star_name + '_square_mask_from_sc_seg.nii.gz'

                        if seg_in_croped == '':
                            sct.run('sct_crop_over_mask.py -i ' + seg_in + ' -mask ' + mask_box + ' -square 1 '
                                    '-o ' + seg_in_name + '_croped')

                        if manual_seg_croped == '':
                            sct.run('sct_crop_over_mask.py -i ' + manual_seg + ' -mask ' + mask_box + ' -square 1'
                                    ' -o ' + manual_seg_name + '_croped')
                        os.chdir('..')
                    except Exception, e:
                        sct.printv('WARNING: an error occured ... \n ' + str(e), 1, 'warning')
                    else:
                        print 'Done !'
                        # sct.run('rm -rf ./tmp_' + now)
            os.chdir('..')


# ------------------------------------------------------------------------------------------------------------------
def crop_t2_star(t2star, sc_seg, box_size=75):
    """
    Pretreatment function croping the t2star file around the spinal cord and to a 45*45 squared image

    :param t2star: t2star image to be croped

    :param sc_seg: segmentation of the spinal cord
    """
    t2star_name = sct.extract_fname(t2star)[1]
    sc_seg_name = sct.extract_fname(sc_seg)[1]
    mask_box = None

    try:

        sct.run('sct_crop_over_mask.py -i ' + t2star + ' -mask ' + sc_seg + ' -square 0 -o ' + t2star_name + '_seg_in')
        seg_in = t2star_name + '_seg_in.nii.gz'
        seg_in_name = t2star_name + '_seg_in'

        sct.run('sct_create_mask -i ' + seg_in + ' -m centerline,' + sc_seg + ' -s ' + str(box_size - 2) + ' -o ' + t2star_name + '_square_mask_from_sc_seg.nii.gz -f box')
        mask_box = t2star_name + '_square_mask_from_sc_seg.nii.gz'

        status, output = sct.run('sct_crop_over_mask.py -i ' + seg_in + ' -mask ' + mask_box + ' -square 1 -o ' + seg_in_name + '_croped')
        if t2star_name + '_square_mask_from_sc_seg_IRP.nii.gz' in os.listdir('.'):
            mask_box = t2star_name + '_square_mask_from_sc_seg_IRP.nii.gz'

    except Exception, e:
        sct.printv('WARNING: an error occured when croping ' + t2star_name + '... \n ' + str(e), 1, 'warning')
    return mask_box


# ------------------------------------------------------------------------------------------------------------------
def save_by_slice(dic_dir):
    """
    from a dictionary containing for each subject a 3D image crop around the spinal cord,
     a graymatter segmentation 3D image, and a level image (from the registration of the template to the T2star image)

     save an image per slice including the level in the image name

    :param dic_dir: dictionary directory
    """
    if dic_dir[-1] == '/':
        dic_by_slice_dir = './' + dic_dir[:-1] + '_by_slice/'
    else:
        dic_by_slice_dir = './' + dic_dir + '_by_slice/'

    sct.run('mkdir ' + dic_by_slice_dir)

    for subject_dir in os.listdir(dic_dir):
        subject_path = dic_dir + '/' + subject_dir
        if os.path.isdir(subject_path):
            sct.run('mkdir ' + dic_by_slice_dir + subject_dir)
            # getting the level file
            path_file_levels = None
            label_by_slice = {}
            level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2',
                           10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}
            for file_name in os.listdir(subject_path):
                if 'level' in file_name:
                    path_file_levels = subject_path + '/' + file_name
                    if 'IRP' not in file_name:
                        sct.run('sct_orientation -i ' + subject_path + '/' + file_name + ' -s IRP')
                        path_file_levels = subject_path + '/' + file_name[:-7] + '_IRP.nii.gz'

            if path_file_levels is None and 'label' in os.listdir(subject_path):
                if 'MNI-Poly-AMU_level_IRP.nii.gz' not in sct.run('ls ' + subject_path + '/label/template')[1]:
                    sct.run('sct_orientation -i ' + subject_path + '/label/template/MNI-Poly-AMU_level.nii.gz -s IRP')
                path_file_levels = subject_path + '/label/template/MNI-Poly-AMU_level_IRP.nii.gz'
            elif path_file_levels is not None:
                im_levels = Image(path_file_levels)
                nz_coord = im_levels.getNonZeroCoordinates()
                for i_level_slice, level_slice in enumerate(im_levels.data):
                    nz_val = []
                    for coord in nz_coord:
                        if coord.x == i_level_slice:
                            nz_val.append(level_slice[coord.y, coord.z])
                    try:
                        label_by_slice[i_level_slice] = int(round(sum(nz_val)/len(nz_val)))
                    except ZeroDivisionError:
                        sct.printv('No level label for slice ' + str(i_level_slice) + ' of subject ' + subject_dir)
                        label_by_slice[i_level_slice] = 0

            for file_name in os.listdir(subject_path):
                if 'seg_in' in file_name and 'croped' in file_name:
                    im = Image(subject_path + '/' + file_name)
                    im_zooms = im.hdr.get_zooms()
                    slice_zoom = (im_zooms[1], im_zooms[2], im_zooms[0])
                    if path_file_levels is None:
                        for i_slice, im_slice in enumerate(im.data):
                            if i_slice < 10:
                                i_slice_str = str(i_slice)
                                i_slice_str = '0' + i_slice_str
                            else:
                                i_slice_str = str(i_slice)
                            im_slice = Image(param=im_slice, absolutepath=dic_by_slice_dir + subject_dir + '/' + subject_dir +
                                  '_slice' + i_slice_str + '_im.nii.gz', hdr=im.hdr)

                            if len(im_slice.hdr.get_zooms()) == 3:
                                im_slice.hdr.set_zooms(slice_zoom)
                            im_slice.save()

                    else:
                        for i_slice, im_slice in enumerate(im.data):
                            if i_slice < 10:
                                i_slice_str = str(i_slice)
                                i_slice_str = '0' + i_slice_str
                            else:
                                i_slice_str = str(i_slice)
                            im_slice = Image(param=im_slice, absolutepath=dic_by_slice_dir + subject_dir + '/' + subject_dir +
                                  '_slice' + i_slice_str + '_' + level_label[label_by_slice[i_slice]] + '_im.nii.gz', hdr=im.hdr)

                            if len(im_slice.hdr.get_zooms()) == 3:
                                im_slice.hdr.set_zooms(slice_zoom)
                            im_slice.save()

                if 'manual_gmseg' in file_name and 'croped' in file_name:
                    seg = Image(dic_dir + subject_dir + '/' + file_name)
                    seg_zooms = seg.hdr.get_zooms()
                    slice_zoom = (seg_zooms[1], seg_zooms[2], seg_zooms[0])
                    if path_file_levels is None:
                        for i_slice, seg_slice in enumerate(seg.data):
                            if i_slice < 10:
                                i_slice_str = str(i_slice)
                                i_slice_str = '0' + i_slice_str
                            else:
                                i_slice_str = str(i_slice)
                            seg_slice = Image(param=seg_slice, absolutepath=dic_by_slice_dir + subject_dir + '/' + subject_dir +
                                  '_slice' + i_slice_str + '_' + level_label[label_by_slice[i_slice]] + '_manual_gmseg.nii.gz', hdr=seg.hdr)

                            if len(seg_slice.hdr.get_zooms()) == 3:
                                seg_slice.hdr.set_zooms(slice_zoom)
                            seg_slice.save()

                    else:
                        for i_slice, seg_slice in enumerate(seg.data):
                            if i_slice < 10:
                                i_slice_str = str(i_slice)
                                i_slice_str = '0' + i_slice_str
                            else:
                                i_slice_str = str(i_slice)

                            seg_slice = Image(param=seg_slice, absolutepath=dic_by_slice_dir + subject_dir + '/' + subject_dir +
                                  '_slice' + i_slice_str + '_' + level_label[label_by_slice[i_slice]] + '_seg.nii.gz', hdr=seg.hdr)

                            if len(seg_slice.hdr.get_zooms()) == 3:
                                seg_slice.hdr.set_zooms(slice_zoom)
                            seg_slice.save()


# ------------------------------------------------------------------------------------------------------------------
def amu_treatments(data_path):
    """
    get a segmentation image of the spinal cord an of the graymatter from a three level mask

    :param data_path: path to the data

    :return:
    """
    im_ext = '.nii'
    if data_path[-1] == '/':
        data_path = data_path[:-1]
    original_path = os.path.abspath('.')
    os.chdir(data_path)
    for subject_dir in os.listdir('.'):
        subject_path = data_path + '/' + subject_dir
        if os.path.isdir(subject_dir):
            os.chdir(subject_dir)
            sc_seg_list = []
            gm_seg_list = []
            im_list = []
            for file_name in os.listdir('.'):
                ext = sct.extract_fname(file_name)[2]
                if 'mask' in file_name and ext != '.hdr':
                    mask_im = Image(file_name)

                    sc_seg_im = mask_im.copy()
                    sc_seg_im.file_name = sct.extract_fname(file_name)[1][:-5] + '_manual_sc_seg'
                    sc_seg_im.ext = '.nii.gz'
                    sc_seg_im.data = (sc_seg_im.data > 1).astype(int)
                    # sc_seg_im = Image(param=sc_seg, absolutepath=subject_path + '/' + sct.extract_fname(file_name)[1][:-5] + '_manual_sc_seg.nii.gz')
                    # sc_seg_im.orientation = 'RPI'
                    sc_seg_im.save()
                    sc_seg_list.append(sc_seg_im.file_name + sc_seg_im.ext)

                    gm_seg_im = mask_im.copy()
                    gm_seg_im.file_name = sct.extract_fname(file_name)[1][:-5] + '_manual_gm_seg'
                    gm_seg_im.ext = '.nii.gz'
                    gm_seg_im.data = (gm_seg_im.data > 2).astype(int)
                    # gm_seg_im = Image(param=gm_seg, absolutepath=subject_path + '/' + sct.extract_fname(file_name)[1][:-5] + '_manual_gm_seg.nii.gz')
                    # gm_seg_im.orientation = 'RPI'
                    gm_seg_im.save()
                    gm_seg_list.append(gm_seg_im.file_name + gm_seg_im.ext)

                    im_list.append(file_name[:2] + im_ext)

            # merging the slice images into a 3D image
            im_list.reverse()
            gm_seg_list.reverse()
            sc_seg_list.reverse()
            cmd_merge = 'fslmerge -z '
            im_name = subject_dir + '_im.nii.gz '
            cmd_merge_im = cmd_merge + im_name
            gmseg_name = subject_dir + '_manual_gmseg.nii.gz '
            cmd_merge_gm_seg = cmd_merge + gmseg_name
            scseg_name = subject_dir + '_manual_scseg.nii.gz '
            cmd_merge_sc_seg = cmd_merge + scseg_name

            for im_i, gm_i, sc_i in zip(im_list, gm_seg_list, sc_seg_list):
                cmd_merge_im += im_i + ' '
                cmd_merge_gm_seg += gm_i + ' '
                cmd_merge_sc_seg += sc_i + ' '

            sct.run(cmd_merge_im)
            sct.run(cmd_merge_gm_seg)
            sct.run(cmd_merge_sc_seg)

            # creating a level image
            level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}
            level_dat = np.zeros((mask_im.data.shape[0], mask_im.data.shape[1], len(im_list)))
            for i, im_i_name in enumerate(im_list):
                level_dat.T[:][:][i] = get_key_from_val(level_label, im_i_name[:2].upper())
            Image(param=level_dat, absolutepath=subject_dir + '_levels.nii.gz').save()

            # resampling

            resample_image(im_name)

            resample_image(gmseg_name, binary=True, thr=0.45)

            resample_image(scseg_name, binary=True, thr=0.55)

            # organizing data
            sct.run('mkdir original_data/')
            sct.run('mkdir extracted_data/')
            sct.run('mkdir 3d_data/')
            sct.run('mkdir 3d_resampled_data/')

            for file_name in os.listdir('.'):
                if 'mask' in file_name:
                    sct.run('mv ' + file_name + ' original_data/')
                elif 'manual' in file_name and 'G1' not in file_name:
                    sct.run('mv ' + file_name + ' extracted_data/')
                elif 'resampled.nii' in file_name:
                    sct.run('mv ' + file_name + ' 3d_resampled_data/')
                elif 'G1' in file_name:
                    sct.run('mv ' + file_name + ' 3d_data/')
                elif not os.path.isdir(os.path.abspath('.') + '/' + file_name):
                    sct.run('mv ' + file_name + ' original_data/')

            os.chdir('..')
    os.chdir(original_path)

# ------------------------------------------------------------------------------------------------------------------
def vanderbilt_treatments(data_path):
    """
    get a segmentation image of the spinal cord an of the graymatter from a three level mask

    :param data_path: path to the data

    :return:
    """
    im_ext = '.nii.gz'
    if data_path[-1] == '/':
        data_path = data_path[:-1]
    original_path = os.path.abspath('.')
    os.chdir(data_path)
    for subject_dir in os.listdir('.'):
        if os.path.isdir(subject_dir):
            os.chdir(subject_dir)
            sc_seg_list = []
            gm_seg_list = []
            im_list = []
            for file_name in os.listdir('.'):
                if 'seg' in file_name:
                    mask_im = Image(file_name)

                    sc_seg_im = mask_im.copy()
                    sc_seg_im.file_name = sct.extract_fname(file_name)[1][:-4] + '_manual_sc_seg'
                    sc_seg_im.ext = '.nii.gz'
                    sc_seg_im.data = (sc_seg_im.data > 0).astype(int)
                    # sc_seg_im = Image(param=sc_seg, absolutepath=subject_path + '/' + sct.extract_fname(file_name)[1][:-5] + '_manual_sc_seg.nii.gz')
                    # sc_seg_im.orientation = 'RPI'
                    sc_seg_im.save()
                    sc_seg_list.append(sc_seg_im.file_name + sc_seg_im.ext)

                    gm_seg_im = mask_im.copy()
                    gm_seg_im.file_name = sct.extract_fname(file_name)[1][:-4] + '_manual_gm_seg'
                    gm_seg_im.ext = '.nii.gz'
                    gm_seg_im.data = (gm_seg_im.data > 1).astype(int)
                    # gm_seg_im = Image(param=gm_seg, absolutepath=subject_path + '/' + sct.extract_fname(file_name)[1][:-5] + '_manual_gm_seg.nii.gz')
                    # gm_seg_im.orientation = 'RPI'
                    gm_seg_im.save()
                    gm_seg_list.append(gm_seg_im.file_name + gm_seg_im.ext)

                    im_list.append(file_name[:17] + im_ext)

            # merging the slice images into a 3D image
            '''
            im_list.reverse()
            gm_seg_list.reverse()
            sc_seg_list.reverse()
            '''

            cmd_merge = 'fslmerge -z '
            im_name = subject_dir + '_im.nii.gz'
            cmd_merge_im = cmd_merge + im_name
            gmseg_name = subject_dir + '_manual_gmseg.nii.gz'
            cmd_merge_gm_seg = cmd_merge + gmseg_name
            scseg_name = subject_dir + '_manual_scseg.nii.gz'
            cmd_merge_sc_seg = cmd_merge + scseg_name

            for im_i, gm_i, sc_i in zip(im_list, gm_seg_list, sc_seg_list):
                cmd_merge_im += ' ' + im_i
                cmd_merge_gm_seg += ' ' + gm_i
                cmd_merge_sc_seg += ' ' + sc_i

            sct.run(cmd_merge_im)
            sct.run(cmd_merge_gm_seg)
            sct.run(cmd_merge_sc_seg)

            label_slices = [im_slice.split('_')[-1][2:4] for im_slice in im_list]
            i_slice_to_level = {0: 6, 1: 6, 2: 6, 3: 6, 4: 6, 5: 5, 6: 5, 7: 5, 8: 5, 9: 5, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 3, 21: 2, 22: 2, 23: 2, 24: 2, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1}

            level_dat = np.zeros((mask_im.data.shape[0], mask_im.data.shape[1], len(im_list)))
            for i, l_slice in enumerate(label_slices):
                i_slice = int(l_slice) - 1
                level_dat.T[:][:][i] = i_slice_to_level[i_slice]
            Image(param=level_dat, absolutepath=subject_dir + '_levels.nii.gz').save()

            '''
            # creating a level image
            level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}
            level_dat = np.zeros((mask_im.data.shape[0], mask_im.data.shape[1], len(im_list)))
            for i, im_i_name in enumerate(im_list):
                level_dat.T[:][:][i] = get_key_from_val(level_label, im_i_name[:2].upper())
            Image(param=level_dat, absolutepath=subject_dir + '_levels.nii.gz').save()
            '''

            # resampling

            resample_image(im_name)

            resample_image(gmseg_name, binary=True)

            resample_image(scseg_name, binary=True)

            # organizing data
            sct.run('mkdir original_data/')
            sct.run('mkdir extracted_data/')
            sct.run('mkdir 3d_data/')
            sct.run('mkdir 3d_resampled_data/')
            sct.run('mkdir dic_data/')

            for file_name in os.listdir('.'):
                if '_manual_gm_seg' in file_name and 'sl' in file_name:
                    sct.run('cp ' + file_name + ' dic_data/')
                    sct.run('cp ' + file_name[:-21] + '.nii.gz dic_data/')
            for file_name in os.listdir('.'):
                if 'sl' in file_name and 'manual' not in file_name:
                    sct.run('mv ' + file_name + ' original_data/')
                elif 'manual' in file_name and 'sl' in file_name:
                    sct.run('mv ' + file_name + ' extracted_data/')
                elif 'resampled.nii' in file_name:
                    sct.run('mv ' + file_name + ' 3d_resampled_data/')
                elif '_sl' not in file_name and not os.path.isdir(os.path.abspath('.') + '/' + file_name) or 'level' in file_name:
                    sct.run('mv ' + file_name + ' 3d_data/')
                elif not os.path.isdir(os.path.abspath('.') + '/' + file_name):
                    sct.run('mv ' + file_name + ' original_data/')

            os.chdir('..')
    os.chdir(original_path)


# ------------------------------------------------------------------------------------------------------------------
def resample_image(fname, suffix='_resampled.nii.gz', binary=False, npx=0.3, npy=0.3, thr=0.0, interpolation='Cubic'):
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname)
    if round(px, 2) != round(npx, 2) or round(py, 2) != round(npy, 2):
        name_pad = fname[:-7] + '_pad.nii.gz'
        name_resample = name_pad[:-7] + suffix
        # name_croped = name_resample[:-7] + '_croped.nii.gz'
        if binary:
            interpolation = 'NearestNeighbor'
        sct.run('c3d ' + fname + ' -pad 0x0x1vox 0x0x0vox 0 -o ' + name_pad)
        fx = px/npx
        fy = py/npy
        sct.run('sct_resample -i ' + name_pad + ' -f ' + str(fx) + 'x' + str(fy) + 'x1 -o ' + name_resample + ' -x ' + interpolation)
        sct.run('sct_crop_image -i ' + name_resample + ' -o ' + name_resample + ' -dim 2 -start 1 -end ' + str(nz))

        if binary:
            sct.run('fslmaths ' + name_resample + ' -thr ' + str(thr) + ' ' + name_resample)
            sct.run('fslmaths ' + name_resample + ' -bin ' + name_resample)

        return name_resample
    else:
        sct.printv('Image resolution already ' + str(npx) + 'x' + str(npy) + 'xpz')
        return fname


# ------------------------------------------------------------------------------------------------------------------
def dataset_pretreatments(path_to_dataset):
    """
    pretreatment function for a dataset of 3D images to be integrated to the model
    the dataset should contain for each subject :
        - a T2*-w image containing 'im' in its name
        - a segmentation of the spinal cord containing 'seg' in its name
        - a manual segmentation of the gray matter containing 'gm' in its name
        - a 'level image' containing 'level' in its name : the level image is an image containing a level label per slice indicating at wich vertebral level correspond this slice
    :param path:
    :return:
    """

    axial_pix_dim = 0.3
    model_image_size = 75
    interpolation = 'Cubic'  # 'Sinc'
    original_path = os.path.abspath('.')
    for subject_dir in os.listdir(path_to_dataset):
        if os.path.isdir(path_to_dataset + '/' + subject_dir):
            os.chdir(path_to_dataset + '/' + subject_dir)
            # getting the subject images
            t2star = ''
            scseg = ''
            gmseg = ''
            for file_name in os.listdir('.'):
                if 'im' in file_name:
                    t2star = file_name
                elif 'gm' in file_name:
                    gmseg = file_name
                elif 'seg' in file_name and 'gm' not in file_name:
                    scseg = file_name

            t2star = resample_image(t2star, npx=axial_pix_dim, npy=axial_pix_dim, interpolation=interpolation)
            scseg = resample_image(scseg, npx=axial_pix_dim, npy=axial_pix_dim, binary=True)
            gmseg = resample_image(gmseg, npx=axial_pix_dim, npy=axial_pix_dim, binary=True)

            new_names = []
            for f_name in [t2star, scseg, gmseg]:
                status, output = sct.run('sct_orientation -i ' + f_name)
                if output != 'RPI':
                    status, output = sct.run('sct_orientation -i ' + f_name + ' -s RPI')
                    new_names.append(output.split(':')[1][1:-1])
            t2star = new_names[0]
            scseg = new_names[1]
            gmseg = new_names[2]


            mask_box = crop_t2_star(t2star, scseg, box_size=model_image_size)
            sct.run('sct_crop_over_mask.py -i ' + gmseg + ' -mask ' + mask_box + ' -square 1 -o ' + sct.extract_fname(gmseg)[1] + '_croped')

            os.chdir(original_path)
    save_by_slice(path_to_dataset)


# ------------------------------------------------------------------------------------------------------------------
def compute_level_file(t2star_fname, t2star_sc_seg_fname , t2_fname, t2_seg_fname, landmarks_fname):
    # Registration to template
    cmd_register_template = 'sct_register_to_template -i ' + t2_fname + ' -s ' + t2_seg_fname + ' -l ' + landmarks_fname
    sct.run(cmd_register_template)

    cmd_warp_template = 'sct_warp_template -d ' + t2_fname + ' -w warp_template2anat.nii.gz'
    sct.run(cmd_warp_template)

    # Registration template to t2star
    cmd_register_multimodal = 'sct_register_multimodal -i template2anat.nii.gz -d ' + t2star_fname + ' -iseg ./label/template/MNI-Poly-AMU_cord.nii.gz -dseg ' + t2star_sc_seg_fname + ' -p step=1,type=seg,algo=syn,metric=MeanSquares,iter=5:step=2,type=im,algo=slicereg,metric=MeanSquares,iter=5'
    sct.run(cmd_register_multimodal)

    multimodal_warp_name = 'warp_template2anat2' + t2star_fname
    total_warp_name = 'warp_template2t2star.nii.gz'
    cmd_concat = 'sct_concat_transfo -w warp_template2anat.nii.gz,' + multimodal_warp_name + ' -d ' + t2star_fname + ' -o ' + total_warp_name
    sct.run(cmd_concat)

    cmd_warp = 'sct_warp_template -d ' + t2star_fname + ' -w ' + total_warp_name
    sct.run(cmd_warp)

    sct.run('sct_orientation -i ./label/template/MNI-Poly-AMU_level.nii.gz -s IRP')

    return './label/template/MNI-Poly-AMU_level_IRP.nii.gz'


########################################################################################################################
# --------------------------------------------------- VALIDATION ----------------------------------------------------- #
########################################################################################################################

# ------------------------------------------------------------------------------------------------------------------
def leave_one_out_old(dic_path, reg=None, target_reg='pairwise'):
    """
    Leave one out cross validation taking 1 SUBJECT out of the dictionary at each step
    and computing the resulting dice coefficient, the time of computation and an error map

    :param dic_path: path to the dictionary to use to do the model validation

    :param reg: type of registration to apply to the data (only in pairwise)

    :param target_reg: type of registration of the target to the model

    """
    import time
    dice_file = open('dice_coeff.txt', 'w')
    dice_sum = 0
    time_file = open('computation_time.txt', 'w')
    time_sum = 0
    n_subject = 0
    n_slices = 0
    e = None
    error_map_sum = None
    error_map_abs_sum = None
    first = True

    for subject_dir in os.listdir(dic_path):
        subject_path = dic_path + '/' + subject_dir
        if os.path.isdir(subject_path):
            try:
                tmp_dir = 'tmp_' + subject_dir + '_as_target'
                sct.run('mkdir ' + tmp_dir)

                tmp_dic_name = 'dic'
                sct.run('cp -r ' + dic_path + ' ./' + tmp_dir + '/' + tmp_dic_name + '/')
                sct.run('mv ./' + tmp_dir + '/' + tmp_dic_name + '/' + subject_dir + ' ./' + tmp_dir)

                # Gray matter segmentation using this subject as target
                os.chdir(tmp_dir)
                target = ''
                ref_gm_seg = ''
                res = ''

                for file_name in os.listdir(subject_dir):
                    if 'seg_in' in file_name:
                        target = subject_dir + '/' + file_name
                    elif 'manual' in file_name:
                        ref_gm_seg = subject_dir + '/' + file_name

                cmd_gm_seg = 'sct_asman -i ' + target + ' -dic ' + tmp_dic_name + ' -model compute  -target-reg ' + target_reg
                if reg is not None:
                    cmd_gm_seg += ' -reg ' + reg

                before_seg = time.time()
                sct.run(cmd_gm_seg)
                seg_time = time.time() - before_seg

                for file_name in os.listdir('.'):
                    if 'graymatterseg' in file_name and 'manual' not in file_name:
                        res = file_name

                # Validation
                ref_gm_seg_im = Image(ref_gm_seg)
                target_im = Image(target)

                inverse_gmseg_to_wmseg(ref_gm_seg_im, target_im, ref_gm_seg_im.file_name)

                ref_wm_seg = ref_gm_seg_im.file_name + '_inv_to_wm.nii.gz'
                ref_wm_seg_im = Image(ref_wm_seg)

                res_im = Image(res)

                # Dice coefficient
                status, dice_output = sct.run('sct_dice_coefficient ' + res + ' ' + ref_wm_seg)
                dice = dice_output[-9:]
                os.chdir('..')

                dice_sum += float(dice)
                n_subject += 1
                dice_file.write(subject_dir + ': ' + dice)

                # Error map
                if first:
                    error_map_sum = np.zeros(ref_wm_seg_im.data[0].shape)
                    error_map_abs_sum = np.zeros(ref_wm_seg_im.data[0].shape)
                    first = False

                error_3d = (ref_wm_seg_im.data - res_im.data) + 1
                error_3d_abs = abs(ref_wm_seg_im.data - res_im.data)

                error_map_sum += np.sum(error_3d, axis=0)
                error_map_abs_sum += np.sum(error_3d_abs, axis=0)

                n_slices += ref_wm_seg_im.data.shape[0]

                # Time of computation
                time_file.write(subject_dir + ' as target: ' + str(seg_time) + ' sec '
                                '- ' + str(seg_time / ref_wm_seg_im.data.shape[0]) + ' sec/target_slice\n')
                time_sum += seg_time

            except Exception, e:
                sct.printv('WARNING: an error occurred ...', 1, 'warning')
                print e
            # else:
            #    sct.run('rm -rf ' + tmp_dir)
    if e is None:
        dice_file.write('\nmean dice: ' + str(dice_sum/n_subject))
        dice_file.close()

        Image(param=(error_map_sum/n_slices) - 1, absolutepath='error_map.nii.gz').save()
        Image(param=error_map_abs_sum/n_slices, absolutepath='error_map_abs.nii.gz').save()

        time_file.write('\nmean computation time: ' + str(time_sum/n_subject) + ' sec')
        time_file.write('\nmean computation time per subject slice: ' + str(time_sum/n_slices) + ' sec')
        time_file.close()

'''
# ------------------------------------------------------------------------------------------------------------------
def leave_one_out_by_subject(dic_path, use_levels=True, weight=1.2):
    """
    Leave one out cross validation taking 1 SUBJECT out of the dictionary at each step
    and computing the resulting dice coefficient, the time of computation and an error map

    :param dic_path: path to the dictionary to use to do the model validation

    """
    import time
    from sct_asman import Model, Param, GMsegSupervisedMethod
    init = time.time()
    wm_dice_file = open('wm_dice_coeff.txt', 'w')
    gm_dice_file = open('gm_dice_coeff.txt', 'w')
    n_subject = 0
    n_slices = 0
    e = None
    error_map_abs_sum = None
    first = True

    level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}

    for subject_dir in os.listdir(dic_path):
        subject_path = dic_path + '/' + subject_dir
        if os.path.isdir(subject_path):
            try:
                tmp_dir = 'tmp_' + subject_dir + '_as_target'
                sct.run('mkdir ' + tmp_dir)

                tmp_dic_name = 'dic'
                sct.run('cp -r ' + dic_path + ' ./' + tmp_dir + '/' + tmp_dic_name + '/')
                sct.run('mv ./' + tmp_dir + '/' + tmp_dic_name + '/' + subject_dir + ' ./' + tmp_dir)

                # Gray matter segmentation using this subject as target
                os.chdir(tmp_dir)
                model_param = Param()
                model_param.path_dictionary = tmp_dic_name
                model_param.todo_model = 'compute'
                model_param.weight_beta = weight
                model_param.use_levels = use_levels
                model_param.res_type = 'prob'

                model = Model(model_param=model_param, k=0.8)
                # model_dir = 'gm_seg_model_data/'

                n_slices_model = model.dictionary.J

                # cmd_compute_model = 'sct_asman -dic ' + tmp_dic_name + ' -model compute -weight ' +  str(weight) + ' -use-levels ' + str(int(use_levels))
                # sct.run(cmd_compute_model)

                for file_name in os.listdir(subject_dir):
                    target = ''
                    ref_gm_seg = ''
                    level = ''

                    if 'im' in file_name:
                        target = subject_dir + '/' + file_name
                        ref_gm_seg = subject_dir + '/' + file_name[:-10] + '_seg.nii.gz'
                        slice_level = None
                        target_slice = ''

                        # cmd_segment_gm = 'sct_asman.py -i ' + target + ' -model load -dic ' + model.param.model_dir + ' -weight ' + str(weight)


                        name_list = file_name.split('_')
                        for word in name_list:
                            if word.upper() in level_label.values():
                                slice_level = word.upper()
                            elif 'slice' in word:
                                target_slice = word

                        # sct.run(cmd_segment_gm)
                        gm_seg_param = Param()
                        gm_seg_param.path_dictionary = model.param.model_dir
                        gm_seg_param.todo_model = 'load'
                        gm_seg_param.weight_beta = weight
                        gm_seg_param.use_levels = use_levels
                        gm_seg_param.res_type = 'prob'

                        gm_seg_method = GMsegSupervisedMethod(target, slice_level, model, gm_seg_param=gm_seg_param)


                        # Validation
                        ref_gm_seg_im = Image(ref_gm_seg)
                        target_im = Image(target)

                        inverse_gmseg_to_wmseg(ref_gm_seg_im, target_im, ref_gm_seg_im.file_name)

                        ref_wm_seg = ref_gm_seg_im.file_name + '_inv_to_wm.nii.gz'
                        ref_wm_seg_im = Image(ref_wm_seg)


                        res_wmseg_prob_im = gm_seg_method.res_wm_seg  # Image(res_wmseg)
                        res_gmseg_prob_im = gm_seg_method.res_gm_seg  # Image(res_gmseg)

                        res_wmseg_prob = res_wmseg_prob_im.file_name + res_wmseg_prob_im.ext
                        res_gmseg_prob = res_gmseg_prob_im.file_name + res_gmseg_prob_im.ext

                        res_wmseg_bin =  res_wmseg_prob_im.file_name + '_binary' + res_wmseg_prob_im.ext
                        res_gmseg_bin =  res_gmseg_prob_im.file_name + '_binary' + res_gmseg_prob_im.ext
                        sct.run('fslmaths ' + res_wmseg_prob + ' -thr 0.50001 ' + res_wmseg_bin)
                        sct.run('fslmaths ' + res_gmseg_prob + ' -thr 0.50001 ' + res_gmseg_bin)

                        res_wmseg_bin_im = Image(res_wmseg_bin)


                        # Dice coefficient
                        status, wm_dice_output = sct.run('sct_dice_coefficient ' + res_wmseg_bin + ' ' + ref_wm_seg)
                        wm_dice = wm_dice_output[22:-1]
                        wm_dice_file.write(subject_dir + ' ' + target_slice + ' ' + slice_level + ': ' + wm_dice + ' ; nslices: ' + str(n_slices_model) + '\n')


                        status, gm_dice_output = sct.run('sct_dice_coefficient ' + res_gmseg_bin + ' ' + ref_gm_seg)
                        gm_dice = gm_dice_output[22:-1]
                        gm_dice_file.write(subject_dir + ' ' + target_slice + ' ' + slice_level + ': ' + gm_dice + ' ; nslices: ' + str(n_slices_model) + '\n')


                        # Error map
                        if first:
                            error_map_abs_sum = np.zeros(ref_wm_seg_im.data.shape)
                            first = False

                        error_3d_abs = abs(ref_wm_seg_im.data - res_wmseg_bin_im.data)

                        error_map_abs_sum += sum(error_3d_abs)
                        n_slices += 1

                        sct.run('mkdir ' + subject_dir + '_' + file_name[:-10] + '/')
                        sct.run('mv ./' + file_name[:-10] + '* ./' + subject_dir + '_' + file_name[:-10] + '/')

                os.chdir('..')

            except Exception, e:
                sct.printv('WARNING: an error occurred ...', 1, 'warning')
                print e
            # else:
            #    sct.run('rm -rf ' + tmp_dir)
    if e is None:
        wm_dice_file.close()
        # gm_dice_file.close()

        Image(param=error_map_abs_sum/n_slices, absolutepath='error_map_abs.nii.gz').save()
        t = time.time() - init
        print 'Done in ' + str(t) + ' sec'
'''

# ------------------------------------------------------------------------------------------------------------------
def leave_one_out_by_subject(param):
    """
    Leave one out cross validation taking 1 SUBJECT out of the dictionary at each step
    and computing the resulting dice coefficient, the time of computation and an error map

    :param dic_path: path to the dictionary to use to do the model validation

    """
    import time
    from sct_asman import Model, Param, GMsegSupervisedMethod
    from sct_segment_graymatter import FullGmSegmentation
    init = time.time()

    dic_path, dic_3d, use_levels, weight, reg, z_reg = param
    use_levels = bool(int(use_levels))
    print 'use-levels: ', use_levels
    z_reg = bool(int(z_reg))
    weight = float(weight)
    print 'weight gamma: ', weight

    wm_dice_file = open('wm_dice_coeff.txt', 'w')
    gm_dice_file = open('gm_dice_coeff.txt', 'w')
    wm_csa_file = open('wm_csa.txt', 'w')
    gm_csa_file = open('gm_csa.txt', 'w')
    n_subject = 0
    n_slices = 0
    e = None
    # error_map_abs_sum = None
    # first = True

    level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}

    for subject_dir in os.listdir(dic_path):
        subject_path = dic_path + '/' + subject_dir
        if os.path.isdir(subject_path):
            try:
                tmp_dir = 'tmp_' + subject_dir + '_as_target'
                sct.run('mkdir ' + tmp_dir)

                tmp_dic_name = 'dic'
                sct.run('cp -r ' + dic_path + ' ./' + tmp_dir + '/' + tmp_dic_name + '/')
                sct.run('cp -r ' + dic_3d + '/' + subject_dir +' ./' + tmp_dir + '/' )
                sct.run('mv ./' + tmp_dir + '/' + tmp_dic_name + '/' + subject_dir + ' ./' + tmp_dir + '/' + subject_dir + '_by_slice')

                # Gray matter segmentation using this subject as target
                os.chdir(tmp_dir)
                model_param = Param()
                model_param.path_dictionary = tmp_dic_name
                model_param.todo_model = 'compute'
                model_param.weight_beta = weight
                model_param.use_levels = use_levels
                model_param.res_type = 'prob'
                model_param.reg = reg.split(':')
                model_param.z_regularisation = z_reg

                model = Model(model_param=model_param, k=0.8)
                # model_dir = 'gm_seg_model_data/'

                n_slices_model = model.dictionary.J

                target = ''
                sc_seg = ''
                ref_gm_seg = ''
                level = ''
                for file_name in os.listdir(subject_dir): # 3d files

                    if 'im' in file_name:
                        target = subject_dir + '/' + file_name
                    elif 'level' in file_name:
                        level = subject_dir + '/' + file_name
                    elif 'gm' in file_name:
                        ref_gm_seg = subject_dir + '/' + file_name
                    elif 'seg' in file_name:
                        sc_seg = subject_dir + '/' + file_name

                # sct.run('sct_segment_graymatter.py -i ' + target + ' -s ' + sc_seg + ' -l ' + level + ' -ref ' + ref_gm_seg + ' -dic ' + model.param.model_dir + ' -res-type prob')

                full_gmseg = FullGmSegmentation(target, sc_seg, None, level, ref_gm_seg=ref_gm_seg, model=model, param=model_param)

                # ## VALIDATION ##
                # Dice coeff
                subject_dice = open(full_gmseg.dice_name, 'r')
                dice_lines_list = subject_dice.readlines()
                subject_dice.close()

                gm_dices = None
                wm_dices = None
                n_subject_slices = len(full_gmseg.gm_seg.target_seg_methods.target)
                n_slices += n_subject_slices

                for i,line in enumerate(dice_lines_list):
                    if 'Gray Matter' in line:
                        gm_dices = dice_lines_list[i+10:i+10+n_subject_slices]

                    if 'White Matter' in line:
                        wm_dices = dice_lines_list[i+10:i+10+n_subject_slices]

                subject_slices_levels = {}
                for slice_dice in wm_dices:
                    target_slice, wm_dice = slice_dice[:-1].split(' ')

                    if int(target_slice) < 10:
                        target_slice = 'slice0' + target_slice
                    else:
                        target_slice = 'slice' + target_slice

                    slice_level = ''
                    for file_name in os.listdir('./' + subject_dir + '_by_slice'):
                        if target_slice in file_name:
                            slice_level = file_name[file_name.find(target_slice)+8:file_name.find(target_slice)+10]
                            subject_slices_levels[target_slice] = slice_level
                    wm_dice_file.write(subject_dir + ' ' + target_slice + ' ' + slice_level + ': ' + wm_dice + ' ; nslices: ' + str(n_slices_model) + '\n')

                for slice_dice in gm_dices:
                    target_slice, gm_dice = slice_dice[:-1].split(' ')

                    if int(target_slice) < 10:
                        target_slice = 'slice0' + target_slice
                    else:
                        target_slice = 'slice' + target_slice
                    slice_level = subject_slices_levels[target_slice]

                    gm_dice_file.write(subject_dir + ' ' + target_slice + ' ' + slice_level + ': ' + gm_dice + ' ; nslices: ' + str(n_slices_model) + '\n')

                # error map
                # TODO : error map by level

                sct.run('sct_process_segmentation -i ' + full_gmseg.res_names['corrected_wm_seg'] + ' -p csa')
                tmp_csa_file = open('csa.txt')
                csa_lines = tmp_csa_file.readlines()
                tmp_csa_file.close()
                for slice_csa in csa_lines:
                    target_slice, wm_csa = slice_csa.split(',')
                    if int(target_slice) < 10:
                        target_slice = 'slice0' + target_slice
                    else:
                        target_slice = 'slice' + target_slice
                    slice_level = subject_slices_levels[target_slice]
                    wm_csa_file.write(subject_dir + ' ' + target_slice + ' ' + slice_level + ': ' + wm_csa[:-1] + '\n')
                sct.run('mv csa.txt csa_corrected_wm_seg.txt')

                sct.run('sct_process_segmentation -i ' + full_gmseg.res_names['gm_seg'] + ' -p csa')
                tmp_csa_file = open('csa.txt')
                csa_lines = tmp_csa_file.readlines()
                tmp_csa_file.close()
                for slice_csa in csa_lines:
                    target_slice, gm_csa = slice_csa.split(',')
                    if int(target_slice) < 10:
                        target_slice = 'slice0' + target_slice
                    else:
                        target_slice = 'slice' + target_slice
                    slice_level = subject_slices_levels[target_slice]
                    gm_csa_file.write(subject_dir + ' ' + target_slice + ' ' + slice_level + ': ' + gm_csa[:-1] + '\n')
                sct.run('mv csa.txt csa_gm_seg.txt')

                os.chdir('..')

            except Exception, e:
                sct.printv('WARNING: an error occurred ...', 1, 'warning')
                print e
            # else:
            #    sct.run('rm -rf ' + tmp_dir)
    if e is None:
        wm_dice_file.close()
        gm_dice_file.close()
        wm_csa_file.close()
        gm_csa_file.close()

        # Image(param=error_map_abs_sum/n_slices, absolutepath='error_map_abs.nii.gz').save()
        t = time.time() - init
        print 'Done in ' + str(t) + ' sec'

# ------------------------------------------------------------------------------------------------------------------
def leave_one_out_by_slice(dic_path, reg=None, target_reg='pairwise', use_levels=True, weight=None):
    """
    Leave one out cross validation taking 1 SLICE out of the dictionary at each step
    and computing the resulting dice coefficient, the time of computation and an error map

    :param dic_path: path to the dictionary to use to do the model validation

    :param reg: type of registration to apply to the data (only in pairwise)

    :param target_reg: type of registration of the target to the model

    """
    import time
    wm_dice_file = open('wm_dice_coeff.txt', 'w')
    wm_dice_sum = 0
    gm_dice_file = open('gm_dice_coeff.txt', 'w')
    gm_dice_sum = 0
    time_file = open('computation_time.txt', 'w')
    level_file = open('levels_similarity.txt', 'w')
    similarity_sum = 0
    time_sum = 0
    n_slices = 0
    e = None
    error_map_sum = None
    error_map_abs_sum = None
    first = True

    for subject_dir in os.listdir(dic_path):
        subject_path = dic_path + '/' + subject_dir
        level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}
        if os.path.isdir(subject_path):
            for file_name in os.listdir(subject_path):
                if "im" in file_name:
                    try:
                        # Gray matter segmentation for this slice as target
                        target = file_name
                        target_name = subject_dir + '_' + sct.extract_fname(target)[1][:-3]
                        '''
                        if 'errsm' in target or 'pilot' in target:
                            ref_gm_seg = sct.extract_fname(target)[1][:-3] + '_seg.nii.gz'
                        else:
                        '''
                        ref_gm_seg = sct.extract_fname(target)[1][:-3] + '_manual_gmseg.nii.gz'
                        # slice_level = target_name[-2:]
                        slice_level = ''
                        target_slice = ''
                        name_list = file_name.split('_')
                        for word in name_list:
                            if word.upper() in level_label.values():
                                slice_level = word.upper()
                            elif 'slice' in word:
                                target_slice = word

                        if target_slice == '':
                            target_slice = slice_level

                        tmp_dir = 'tmp_' + subject_dir + '_' + target_slice + '_as_target'
                        sct.run('mkdir ./' + tmp_dir)

                        tmp_dic_name = 'dic'

                        sct.run('cp -r ' + dic_path + ' ./' + tmp_dir + '/' + tmp_dic_name + '/')

                        sct.run('mv ./' + tmp_dir + '/' + tmp_dic_name + '/' + subject_dir + '/' + target + ' ./' + tmp_dir)
                        sct.run('mv ./' + tmp_dir + '/' + tmp_dic_name + '/' + subject_dir + '/' + ref_gm_seg + ' ./' + tmp_dir)

                        # beginning of the gm seg
                        os.chdir(tmp_dir)

                        gm_res = ''
                        wm_res = ''
                        cmd_gm_seg = 'sct_asman -i ' + target + ' -dic ' + tmp_dic_name + ' -model compute  -target-reg ' + target_reg

                        if use_levels:
                            cmd_gm_seg += ' -l ' + slice_level
                        if reg is not None:
                            cmd_gm_seg += ' -reg ' + reg
                        if weight is not None:
                            cmd_gm_seg += ' -weight ' + str(weight)

                        before_seg = time.time()
                        sct.run(cmd_gm_seg)
                        seg_time = time.time() - before_seg

                        for file_name_tmp_dir in os.listdir('.'):
                            if 'wmseg' in file_name_tmp_dir and 'inv_to_gm' not in file_name_tmp_dir and '_corrected' not in file_name_tmp_dir:
                                wm_res = file_name_tmp_dir
                                wm_res_im = Image(wm_res)
                            elif 'inv_to_gm' in file_name_tmp_dir:
                                gm_res = file_name_tmp_dir
                                gm_res_im = Image(gm_res)

                        # Validation
                        ref_gm_seg_im = Image(ref_gm_seg)
                        target_im = Image(target)

                        inverse_gmseg_to_wmseg(ref_gm_seg_im, target_im, ref_gm_seg_im.file_name)

                        ref_wm_seg = ref_gm_seg_im.file_name + '_inv_to_wm.nii.gz'
                        ref_wm_seg_im = Image(ref_wm_seg)

                        # Levels similarity
                        selected_slices_levels = open('selected_slices.txt', 'r')
                        line_list = selected_slices_levels.read().split("'")
                        selected_slices_levels.close()
                        levels = []
                        similar_levels = 0
                        for s in line_list:
                            if s in level_label.values():
                                levels.append(s)
                        for l in levels:
                            if l == slice_level:
                                similar_levels += 1
                        slice_similarity = float(similar_levels)/len(levels)
                        similarity_sum += slice_similarity
                        level_file.write(subject_dir + ' slice ' + target_slice + ': ' + str(100*slice_similarity) + '% (' + str(slice_level) + ')\n')

                        # Dice coefficient
                        status, wm_dice_output = sct.run('sct_dice_coefficient ' + wm_res + ' ' + ref_wm_seg)
                        wm_dice = wm_dice_output.split(' ')[-1]

                        status, gm_dice_output = sct.run('sct_dice_coefficient ' + gm_res + ' ' + ref_gm_seg)
                        gm_dice = gm_dice_output.split(' ')[-1]
                        os.chdir('..')

                        wm_dice_sum += float(wm_dice)
                        gm_dice_sum += float(gm_dice)
                        n_slices += 1
                        wm_dice_file.write(subject_dir + ' slice ' + target_slice + ': ' + wm_dice[:-1] + ' (' + str(slice_level) + ')\n')
                        gm_dice_file.write(subject_dir + ' slice ' + target_slice + ': ' + gm_dice[:-1] + ' (' + str(slice_level) + ')\n')

                        # Error map
                        if first:
                            error_map_sum = np.zeros(ref_wm_seg_im.data.shape)
                            error_map_abs_sum = np.zeros(ref_wm_seg_im.data.shape)
                            first = False

                        error_3d = (ref_wm_seg_im.data - wm_res_im.data) + 1
                        error_3d_abs = abs(ref_wm_seg_im.data - wm_res_im.data)

                        error_map_sum += error_3d
                        error_map_abs_sum += error_3d_abs

                        # n_slices += ref_wm_seg_im.data.shape[0]

                        # Time of computation
                        time_file.write(subject_dir + ' slice ' + target_slice + ' as target: ' + str(seg_time) + ' sec\n')
                        time_sum += seg_time

                    except Exception, e:
                        sct.printv('WARNING: an error occurred ...', 1, 'warning')
                        print e
                    # else:
                    #    sct.run('rm -rf ' + tmp_dir)
    # if e is None:
    wm_dice_file.write('\nmean dice: ' + str(wm_dice_sum/n_slices))
    wm_dice_file.close()

    gm_dice_file.write('\nmean dice: ' + str(gm_dice_sum/n_slices))
    gm_dice_file.close()

    level_file.write('\nmean similarity: ' + str(100*similarity_sum/n_slices) + '% ')
    level_file.close()

    Image(param=(error_map_sum/n_slices) - 1, absolutepath='error_map.nii.gz').save()
    Image(param=error_map_abs_sum/n_slices, absolutepath='error_map_abs.nii.gz').save()

    time_file.write('\nmean computation time: ' + str(time_sum/n_slices) + ' sec')
    time_file.close()

########################################################################################################################
# ------------------------------------------------- POST-TREATMENTS -------------------------------------------------- #
########################################################################################################################


# ------------------------------------------------------------------------------------------------------------------
def inverse_square_crop(croped_image, square_mask):
    nz_coord = square_mask.getNonZeroCoordinates()

    assert len(nz_coord) == croped_image.data.size
    dim = len(croped_image.data.shape)

    inverse_croped = square_mask.copy()
    if dim == 3 :
        done_slices = []
        for coord in nz_coord:
            if coord.x not in done_slices:
                inverse_croped.data[coord.x, coord.y: coord.y + croped_image.data.shape[1], coord.z: coord.z + croped_image.data.shape[2]] = croped_image.data[coord.x]
                done_slices.append(coord.x)
    elif dim == 2:
        coord = nz_coord[0]
        inverse_croped.data[coord.x: coord.x + croped_image.data.shape[0], coord.y: coord.y + croped_image.data.shape[1]] = croped_image.data

    inverse_croped.file_name = croped_image.file_name + '_original_dimension'

    return inverse_croped


# ------------------------------------------------------------------------------------------------------------------
def compute_error_map(data_path):
    error_map_sum = None
    error_map_abs_sum = None
    first = True
    n_slices = 0
    os.chdir(data_path)
    for file_name in os.listdir('.'):
        if os.path.isdir(file_name) and file_name != 'dictionary':
            os.chdir(file_name)

            res = ''
            ref_wm_seg = ''

            for slice_file in os.listdir('.'):
                if 'graymatterseg' in slice_file:
                    res = slice_file
                elif 'inv_to_wm' in slice_file:
                    ref_wm_seg = slice_file
            res_im = Image(res)
            ref_wm_seg_im = Image(ref_wm_seg)

            if first:
                error_map_sum = np.zeros(ref_wm_seg_im.data.shape)
                error_map_abs_sum = np.zeros(ref_wm_seg_im.data.shape)
                first = False

            error_3d = (ref_wm_seg_im.data - res_im.data) + 1
            error_3d_abs = abs(ref_wm_seg_im.data - res_im.data)

            error_map_sum += error_3d
            error_map_abs_sum += error_3d_abs
            n_slices += 1
            os.chdir('..')

    Image(param=(error_map_sum/n_slices) - 1, absolutepath='error_map.nii.gz').save()
    Image(param=error_map_abs_sum/n_slices, absolutepath='error_map_abs.nii.gz').save()


# ------------------------------------------------------------------------------------------------------------------
def compute_error_map_by_level_loocv_by_slice(data_path):
    diff_by_level = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}
    error_map_abs_sum = None
    first = True
    n_slices = 0
    os.chdir(data_path)
    for file_name in os.listdir('.'):
        if os.path.isdir(file_name) and file_name != 'dictionary':
            os.chdir(file_name)

            for slice_dir in os.listdir('.'):
                if 'slice' in slice_dir:
                    os.chdir(slice_dir)
                    res = ''
                    ref_wm_seg = ''

                    for slice_file in os.listdir('.'):
                        if 'res' in slice_file and 'corrected' not in slice_file and 'gm' not in slice_file:
                            res = slice_file
                        elif 'inv_to_wm' in slice_file:
                            ref_wm_seg = slice_file
                    res_im = Image(res)
                    ref_wm_seg_im = Image(ref_wm_seg)
                    level = ''
                    for word in ref_wm_seg.split('_'):
                        if word.upper() in diff_by_level.keys():
                            level = word.upper()

                    if first:
                        error_map_abs_sum = np.zeros(ref_wm_seg_im.data.shape)
                        first = False


                    error_3d_abs = abs(ref_wm_seg_im.data - res_im.data)
                    diff_by_level[level].append(error_3d_abs)
                    error_map_abs_sum += error_3d_abs
                    n_slices += 1
                    os.chdir('..')
            os.chdir('..')


    for l, errors in diff_by_level.items():
        n = len(errors)

        Image(param=sum(errors)/n, absolutepath='error_map_' + str(l) + '.nii.gz').save()

    Image(param=error_map_abs_sum/n_slices, absolutepath='error_map_abs.nii.gz').save()


# ------------------------------------------------------------------------------------------------------------------
'''
import os
os.chdir('/Volumes/folder_shared/greymattersegmentation/supervised_gmseg_method/res/2015-06-23-LOOCV_CSA')
from msct_gmseg_utils import compute_error_map_by_level
compute_error_map_by_level('no_levels_gamma0_Affine')
'''


def compute_error_map_by_level(data_path):
    original_path = os.path.abspath('.')
    diff_by_level = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': [], '': []}
    error_map_abs_sum = None
    first = True
    n_slices = 0
    os.chdir(data_path)
    for subject_dir in os.listdir('.'):
        if os.path.isdir(subject_dir) and subject_dir != 'dictionary' and subject_dir != 'dic':
            os.chdir(subject_dir)

            subject = '_'.join(subject_dir.split('_')[1:3])
            if 'pilot' in subject:
                subject = subject.split('_')[0]

            # ref_gm_seg = ''
            # ref_sc_seg = ''
            sq_mask = ''
            level = ''
            res_wm_seg = ''
            res_dir = None
            for fname in os.listdir('.'):
                if 'tmp_' + subject in fname and os.path.isdir(fname):
                    res_dir = fname
            """
            if 'manual_gmseg' in fname:
                ref_gm_seg  = './' + subject + '/' + fname
                status, res_orientation = sct.run('sct_orientation -i ' + ref_gm_seg)
                gm_or = res_orientation[4:7]
                if gm_or != 'IRP':
                    sct.run('sct_orientation -i ' + ref_gm_seg + ' -s IRP')
                    ref_gm_seg = sct.extract_fname(ref_gm_seg)[0] + sct.extract_fname(ref_gm_seg)[1] + '_IRP' + sct.extract_fname(ref_gm_seg)[2]

            elif 'seg' in fname and 'gm' not in fname:
                ref_sc_seg  = './' + subject + '/' + fname
                status, res_orientation = sct.run('sct_orientation -i ' + ref_sc_seg)
                sc_or = res_orientation[4:7]
                if sc_or != 'IRP':
                    sct.run('sct_orientation -i ' + ref_sc_seg + ' -s IRP')
                    ref_sc_seg = sct.extract_fname(ref_sc_seg)[0] + sct.extract_fname(ref_sc_seg)[1] + '_IRP' + sct.extract_fname(ref_sc_seg)[2]

            elif 'level' in fname:
                level = './' + subject + '/' + fname
                status, level_orientation = sct.run('sct_orientation -i ' + level)
                level_orientation = level_orientation[4:7]
                if level_orientation != 'IRP':
                    sct.run('sct_orientation -i ' + level + ' -s IRP')
                    level = sct.extract_fname(level)[0] + sct.extract_fname(level)[1] + '_IRP' + sct.extract_fname(level)[2]
            """
            os.chdir(res_dir)
            for fname in os.listdir('.'):
                if 'level' in fname and 'MNI-Poly-AMU' in fname and 'IRP' in fname:
                    level = fname
                    '''
                    status, level_orientation = sct.run('sct_orientation -i ' + level)
                    level_orientation = level_orientation[4:7]
                    if level_orientation != 'IRP':
                        sct.run('sct_orientation -i ' + level + ' -s IRP')
                        level = sct.extract_fname(level)[0] + sct.extract_fname(level)[1] + '_IRP' + sct.extract_fname(level)[2]
                    '''
                elif 'square_mask' in fname and 'IRP' in fname:
                    sq_mask = fname
                elif 'res_wmseg' in fname and 'corrected' in fname and 'original_dimension' not in fname:
                    res_wm_seg = fname

            '''
            res_wm_seg = 'res_wm_seg_bin.nii.gz'
            status, res_ori = sct.run('sct_orientation -i ' + res_wm_seg)
            res_ori = res_ori[4:7]
            if res_ori != 'IRP':
                sct.run('sct_orientation -i ' + res_wm_seg + ' -s IRP')
                res_wm_seg = sct.extract_fname(res_wm_seg)[0] + sct.extract_fname(res_wm_seg)[1] + '_IRP' + sct.extract_fname(res_wm_seg)[2]
            '''
            ref_wm_seg = 'ref_wm_seg.nii.gz'
            status, ref_ori = sct.run('sct_orientation -i ' + ref_wm_seg)
            ref_ori = ref_ori[4:7]
            if ref_ori != 'IRP':
                sct.run('sct_orientation -i ' + ref_wm_seg + ' -s IRP')
                ref_wm_seg = sct.extract_fname(ref_wm_seg)[0] + sct.extract_fname(ref_wm_seg)[1] + '_IRP' + sct.extract_fname(ref_wm_seg)[2]

            level_im = Image(level)

            sct.run('sct_crop_over_mask.py -i ' + ref_wm_seg + ' -mask ' + sq_mask + ' -square 1 -o ' + sct.extract_fname(ref_wm_seg)[1] + '_croped')
            ref_wm_seg = sct.extract_fname(ref_wm_seg)[0] + sct.extract_fname(ref_wm_seg)[1] + '_croped' + sct.extract_fname(ref_wm_seg)[2]

            '''
            sct.run('sct_crop_over_mask.py -i ' + res_wm_seg + ' -mask ' + sq_mask + ' -square 1 -o ' + sct.extract_fname(res_wm_seg)[1] + '_croped')
            res_wm_seg = sct.extract_fname(res_wm_seg)[0] + sct.extract_fname(res_wm_seg)[1] + '_croped' + sct.extract_fname(res_wm_seg)[2]
            '''
            ref_wm_seg_im = Image(ref_wm_seg)
            res_wm_seg_im = Image(res_wm_seg)


            '''
            ref_gm_seg_im = Image(ref_gm_seg)
            ref_sc_seg_im = Image(ref_sc_seg)

            ref_wm_seg_im = inverse_gmseg_to_wmseg(ref_gm_seg_im, ref_sc_seg_im, 'ref_gm_seg')

            res_wm_seg = ''
            for fname in os.listdir('.'):
                if 'wmseg' in fname and 'corrected' in fname:
                    res_wm_seg = fname

            status, res_orientation = sct.run('sct_orientation -i ' + res_wm_seg)
            orientation = res_orientation[4:7]
            if orientation != 'IRP':
                sct.run('sct_orientation -i ' + res_wm_seg + ' -s IRP')
                res_wm_seg = sct.extract_fname(res_wm_seg)[1] + '_IRP' + sct.extract_fname(res_wm_seg)[2]
            res_wm_seg_im = Image(res_wm_seg)
            '''
            label_by_slice = {}
            level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2',
                           10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}
            nz_coord = level_im.getNonZeroCoordinates()
            for i_level_slice, level_slice in enumerate(level_im.data):
                nz_val = []
                for coord in nz_coord:
                    if coord.x == i_level_slice:
                        nz_val.append(level_slice[coord.y, coord.z])
                try:
                    label_by_slice[i_level_slice] = int(round(sum(nz_val)/len(nz_val)))
                except ZeroDivisionError:
                    sct.printv('No level label for slice ' + str(i_level_slice) + ' of subject ' + subject_dir)
                    label_by_slice[i_level_slice] = 0

            for i_slice in range(len(ref_wm_seg_im.data)):
                if first:
                    error_map_abs_sum = np.zeros(ref_wm_seg_im.data[i_slice].shape)
                    first = False

                error_3d_abs = abs(ref_wm_seg_im.data[i_slice] - res_wm_seg_im.data[i_slice])
                slice_level = level_label[label_by_slice[i_slice]]
                diff_by_level[slice_level].append(error_3d_abs)
                error_map_abs_sum += error_3d_abs
                n_slices += 1
            os.chdir('../..')
    for l, errors in diff_by_level.items():
        try:
            n = len(errors)
            if n != 0:
                Image(param=sum(errors)/n, absolutepath='error_map_' + str(l) + '.nii.gz').save()
        except ZeroDivisionError:
            sct.printv('WARNING: no data for level ' + str(l), 1, 'warning')
    Image(param=error_map_abs_sum/n_slices, absolutepath='error_map_abs.nii.gz').save()
    os.chdir('..')
    os.chdir(original_path)
'''
from msct_gmseg_utils import compute_error_map_by_level
compute_error_map_by_level('.')

'''


# ------------------------------------------------------------------------------------------------------------------
def compute_similarities(data_path):
    similarity_sum = 0
    n_slices = 0
    os.chdir(data_path)
    level_file = open('levels_similarity.txt', 'w')
    for file_name in os.listdir('.'):
        if os.path.isdir(file_name) and file_name != 'dictionary':
            os.chdir(file_name)

            for im_file in os.listdir('.'):
                if 'seg.nii.gz' in im_file:
                    ref_seg = im_file
            subject = ref_seg[:8]
            n_slice = ref_seg[14:16]
            slice_level = ref_seg[-13:-11]

            selected_slices_levels = open('selected_slices.txt', 'r')
            line_list = selected_slices_levels.read().split("'")
            selected_slices_levels.close()
            levels = []
            similar_levels = 0
            for s in line_list:
                if s in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6']:
                    levels.append(s)
            for l in levels:
                if l == slice_level:
                    similar_levels += 1
            slice_similarity = float(similar_levels)/len(levels)
            similarity_sum += slice_similarity
            level_file.write(subject + ' slice ' + n_slice + ': ' + str(slice_similarity*100) + '% (' + str(slice_level) + ')\n')
            n_slices += 1
            os.chdir('..')
    level_file.write('\nmean similarity: ' + str(similarity_sum/n_slices) + '% ')
    level_file.close()
    os.chdir('..')


# ------------------------------------------------------------------------------------------------------------------
def save_data_by_slice(path_to_model):
    import pickle
    model_slices_list = pickle.load(open(path_to_model + '/dictionary_slices.pkl', 'rb'))
    sct.run('mkdir ' + path_to_model + '/model_slices')
    model_slices = [Slice(slice_id=i_slice, level=dic_slice[2], im_m=dic_slice[0], wm_seg_m=dic_slice[1], im_m_flat=dic_slice[0].flatten(),  wm_seg_m_flat=dic_slice[1].flatten()) for i_slice, dic_slice in enumerate(model_slices_list)]  # type: list of slices

    for mod_slice in model_slices:
        Image(param=mod_slice.im_M, absolutepath= path_to_model + '/model_slices/slice' + str(mod_slice.id) + '_level' + str(mod_slice.level) + '_im_m.nii.gz').save()
        Image(param=mod_slice.wm_seg_M, absolutepath= path_to_model + '/model_slices/slice' + str(mod_slice.id) + '_level' + str(mod_slice.level) + '_wmseg_m.nii.gz').save()


if __name__ == "__main__":
        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Utility functions for the gray matter segmentation')
        parser.add_option(name="-crop",
                          type_value="folder",
                          description="Path to the folder containing all your subjects' data "
                                      "to be croped as pretreatment",
                          mandatory=False,
                          example='dictionary/')
        parser.add_option(name="-loocv",
                          type_value=[[','], 'str'],# "folder",
                          description="Path to a dictionary folder to do 'Leave One Out Validation' on. If you want to do several registrations, separate them by : without white space"
                                      "dictionary_by_slice/,dictionary_3d/,use_levels,weight_param,registration_type,z_regularisation",
                          mandatory=False,
                          example='dic_by_slice/,dic_3d/,1,1.2,Rigid:Affine,1') # 'dictionary/')
        parser.add_option(name="-error-map",
                          type_value="folder",
                          description="Path to a dictionary folder to compute the error map on",
                          mandatory=False,
                          example='dictionary/')
        parser.add_option(name="-save-dic-by-slice",
                          type_value="folder",
                          description="Path to a dictionary folder to be saved by slice",
                          mandatory=False,
                          example='dictionary/')
        parser.add_option(name="-treat-AMU",
                          type_value="folder",
                          description="Path to a dictionary folder with images in the AMU format to be treated",
                          mandatory=False,
                          example='dictionary/')
        parser.add_option(name="-treat-vanderbilt",
                          type_value="folder",
                          description="Path to a dictionary folder with images in the vanderbilt format to be treated",
                          mandatory=False,
                          example='dictionary/')
        parser.add_option(name="-pretreat",
                          type_value="folder",
                          description="Path to a dictionary folder of data to be pre-treated. Each subject folder should contain a t2star image, a GM manual segmentation, a spinal cord segmentationand and a level label image ",
                          mandatory=False,
                          example='dictionary/')
        parser.add_option(name="-gmseg-to-wmseg",
                          type_value=[[','], 'file'],
                          description="Gray matter segmentation image and spinal cord segmentation image",
                          mandatory=False,
                          example='manual_gmseg.nii.gz,sc_seg.nii.gz')

        arguments = parser.parse(sys.argv[1:])

        if "-crop" in arguments:
            crop_t2_star_pipeline(arguments['-crop'])
        if "-loocv" in arguments:
            leave_one_out_by_subject(arguments['-loocv'])
        if "-error-map" in arguments:
            compute_error_map(arguments['-error-map'])
        if "-save-dic-by-slice" in arguments:
            save_by_slice(arguments['-save-dic-by-slice'])
        if "-treat-AMU" in arguments:
            amu_treatments(arguments['-treat-AMU'])
        if "-treat-vanderbilt" in arguments:
            vanderbilt_treatments(arguments['-treat-vanderbilt'])
        if "-pretreat" in arguments:
            dataset_pretreatments(arguments['-pretreat'])
        if "-gmseg-to-wmseg" in arguments:
            gmseg = arguments['-gmseg-to-wmseg'][0]
            gmseg_im = Image(gmseg)
            scseg = arguments['-gmseg-to-wmseg'][1]
            scseg_im = Image(scseg)
            inverse_gmseg_to_wmseg(gmseg_im, scseg_im, sct.extract_fname(gmseg)[1])


