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

from msct_image import Image, get_dimension
import sct_utils as sct
from msct_parser import Parser
from sct_image import set_orientation, get_orientation_3d

def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Utility functions for the gray matter segmentation')
    parser.add_option(name="-crop",
                      type_value="folder",
                      description="Path to the folder containing all your subjects' data "
                                  "to be croped as preprocessing",
                      mandatory=False,
                      example='dictionary/')
    parser.add_option(name="-loocv",
                      type_value=[[','], 'str'],# "folder",
                      description="Path to a dictionary folder to do 'Leave One Out Validation' on. If you want to do several registrations, separate them by \":\" without white space "
                                  "dictionary-by-slice/,dictionary3d/,denoising,registration_type,metric,use_levels,weight,eq_id,mode_weighted_similarity,weighted_label_fusion"
                                  "If you use denoising, the di-by-slice should be denoised, no need to change the 3d-dic.",
                      # dic_path_original, dic_3d, denoising, reg, metric, use_levels, weight, eq, mode_weighted_sim, weighted_label_fusion
                      mandatory=False,
                      example='dic_by_slice/,dic_3d/,1,Rigid:Affine,MI,1,2.5,1,0,0')
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
    parser.add_option(name="-hausdorff",
                      type_value="folder",
                      description="Path to a folder with various loocv results",
                      mandatory=False,
                      example='dictionary/')
    parser.add_option(name="-preprocess",
                      type_value="folder",
                      description="Path to a dictionary folder of data to be pre-processed. Each subject folder should contain a t2star image, a GM manual segmentation, a spinal cord segmentationand and a level label image ",
                      mandatory=False,
                      example='dictionary/')
    parser.add_option(name="-gmseg-to-wmseg",
                      type_value=[[','], 'file'],
                      description="Gray matter segmentation image and spinal cord segmentation image",
                      mandatory=False,
                      example='manual_gmseg.nii.gz,sc_seg.nii.gz')
    return parser


class Slice:
    """
    Slice instance used in the model dictionary for the segmentation of the gray matter
    """
    def __init__(self, slice_id=None, im=None, sc_seg=None, list_gm_seg=None, list_wm_seg=None, reg_to_m=None, im_m=None, list_gm_seg_m=None, list_wm_seg_m=None, im_m_flat=None, list_gm_seg_m_flat=None, list_wm_seg_m_flat=None, level=None):
        """
        Slice constructor

        :param slice_id: slice ID number, type: int

        :param im: original image (a T2star 2D image croped around the spinal cord), type: numpy array

        :param list_gm_seg: list of manual gray matter segmentation of the original image, type: numpy array

        :param list_wm_seg: list of manual white matter segmentation of the original image, type: numpy array

        :param reg_to_m: name of the file containing the transformation for this slice to go from the image original space to the model space, type: string

        :param im_m: image in the model space, type: numpy array

        :param list_gm_seg_m: list of manual gray matter segmentation in the model space, type: numpy array

        :param list_wm_seg_m: list of manual white matter segmentation in the model space, type: numpy array

        :param im_m_flat: flatten image in the model space, type: numpy array

        :param list_gm_seg_m_flat: list of flatten manual gray matter  segmentation in the model space, type: numpy array

        :param list_wm_seg_m_flat: list of flatten manual white matter segmentation in the model space, type: numpy array

        :param level: vertebral level of the slice, type: int
        """
        self.id = slice_id
        self.im = np.asarray(im)
        self.sc_seg = np.asarray(sc_seg)
        self.gm_seg = np.asarray(list_gm_seg)
        self.wm_seg = np.asarray(list_wm_seg)
        self.reg_to_M = reg_to_m
        self.im_M = np.asarray(im_m)
        self.gm_seg_M = np.asarray(list_gm_seg_m)
        self.wm_seg_M = np.asarray(list_wm_seg_m)
        self.im_M_flat = im_m_flat
        self.gm_seg_M_flat = list_gm_seg_m_flat
        self.wm_seg_M_flat = list_wm_seg_m_flat
        self.level = level

    def set(self, slice_id=None, im=None, sc_seg=None, list_gm_seg=None, list_wm_seg=None, reg_to_m=None, im_m=None, list_gm_seg_m=None, list_wm_seg_m=None, im_m_flat=None, list_gm_seg_m_flat=None, list_wm_seg_m_flat=None, level=None):
        """
        Slice setter, only the specified parameters are set

        :param slice_id: slice ID number, type: int

        :param im: original image (a T2star 2D image croped around the spinal cord), type: numpy array

        :param list_gm_seg: list of manual gray matter segmentation of the original image, type: numpy array

        :param list_wm_seg: list of manual white matter segmentation of the original image, type: numpy array

        :param reg_to_m: name of the file containing the transformation for this slice to go from the image original space to the model space, type: string

        :param im_m: image in the model space, type: numpy array

        :param list_gm_seg_m: list of manual gray matter segmentation in the model space, type: numpy array

        :param list_wm_seg_m: list of manual white matter segmentation in the model space, type: numpy array

        :param im_m_flat: flatten image in the model space, type: numpy array

        :param list_gm_seg_m_flat: list of flatten manual gray matter  segmentation in the model space, type: numpy array

        :param list_wm_seg_m_flat: list of flatten manual white matter segmentation in the model space, type: numpy array

        :param level: vertebral level of the slice, type: int
        """
        if slice_id is not None:
            self.id = slice_id
        if im is not None:
            self.im = im
        if sc_seg is not None:
            self.sc_seg = sc_seg
        if list_gm_seg is not None:
            self.gm_seg = list_gm_seg
        if list_wm_seg is not None:
            self.wm_seg = list_wm_seg
        if reg_to_m is not None:
            self.reg_to_M = reg_to_m
        if im_m is not None:
            self.im_M = im_m
        if list_gm_seg_m is not None:
            self.gm_seg_M = list_gm_seg_m
        if list_wm_seg_m is not None:
            self.wm_seg_M = list_wm_seg_m
        if im_m_flat is not None:
            self.im_M_flat = im_m_flat
        if list_gm_seg_m_flat is not None:
            self.gm_seg_M_flat = list_gm_seg_m_flat
        if list_wm_seg_m_flat is not None:
            self.wm_seg_M_flat = list_wm_seg_m_flat
        if level is not None:
            self.level = level

    def __repr__(self):
        s = '\nSlice #' + str(self.id)
        if self.level is not None:
            s += 'Level : ' + str(self.level)
        s += '\nImage : \n' + str(self.im) + '\nGray matter segmentation : \n' + str(self.gm_seg) +\
             '\nTransformation to model space : ' + self.reg_to_M
        if self.im_M is not None:
            s += '\nImage in the common model space: \n' + str(self.im_M)
        if self.gm_seg_M is not None:
            s += '\nGray matter segmentation : \n' + str(self.gm_seg_M)
        return s


########################################################################################################################
# ---------------------------------------------------- FUNCTIONS ----------------------------------------------------- #
########################################################################################################################

########################################################################################################################
# ----------------------------------------- ONLY USED BY MSCT_MULTIATLAS_SEG ----------------------------------------- #
########################################################################################################################

# ------------------------------------------------------------------------------------------------------------------
def inverse_gmseg_to_wmseg(gm_seg, original_im, name_gm_seg='gmseg', save=True, verbose=1):
    """
    Inverse a gray matter segmentation array image to get a white matter segmentation image and save it

    :param gm_seg: gray matter segmentation to inverse, type: Image or np.ndarray

    :param original_im: original image croped around the spinal cord, type: Image or np.ndarray

    :param name_gm_seg: name of the gray matter segmentation (to save the associated white matter segmentation),
     type: string

    :return inverted_seg: white matter segmentation image, if an image is inputted, returns an image, if only np.ndarrays are inputted, return a np.nodarray
    """
    # getting the inputs correctly
    type_res = ''
    if isinstance(gm_seg, Image):
        original_hdr = gm_seg.hdr
        gm_seg_dat = gm_seg.data
        type_res += 'image'
    elif isinstance(gm_seg, np.ndarray):
        gm_seg_dat = gm_seg
        type_res += 'array'
    else:
        sct.printv('WARNING: gm_seg is instance of ' + type(gm_seg), verbose, 'warning')
        gm_seg_dat = None

    if isinstance(original_im, Image):
        original_dat = original_im.data
        type_res += 'image'

    elif isinstance(original_im, np.ndarray):
        original_dat = original_im
        type_res += 'array'
    else:
        sct.printv('WARNING: original_im is instance of ' + type(original_im), verbose, 'warning')
        original_dat = None

    # check that they are of the same shape
    assert gm_seg_dat.shape == original_dat.shape

    # inverse arrays
    binary_gm_seg_dat = (gm_seg_dat > 0).astype(int)
    sc_dat = (original_dat > 0).astype(int)
    # cast of the -1 values (-> GM pixel at the exterior of the SC pixels) to +1 --> WM pixel
    res_wm_seg = np.asarray(np.absolute(sc_dat - binary_gm_seg_dat).astype(int))

    if 'image' in type_res:
        res_wm_seg_im = Image(param=res_wm_seg, absolutepath=name_gm_seg + '_inv_to_wm.nii.gz')
        res_wm_seg_im.hdr = original_hdr
        if save:
            res_wm_seg_im.save()
        res_wm_seg_im.orientation = gm_seg.orientation
        return res_wm_seg_im
    else:
        return res_wm_seg


# ----------------------------------------------------------------------------------------------------------------------
def apply_ants_transfo(fixed_im, moving_im, search_reg=True, transfo_type='Affine', metric='MI', apply_transfo=True, transfo_name='', binary=True, path='./', inverse=0, verbose=0):
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
        dir_name = 'tmp_reg_' + time.strftime("%y%m%d%H%M%S") + '_' + str(time.time())+'/'
        sct.run('mkdir ' + dir_name, verbose=verbose)
        os.chdir('./' + dir_name)

        if binary:
            t = 'uint8'
        else:
            t = ''

        fixed_im_name = 'fixed_im'
        Image(param=np.asarray(fixed_im), absolutepath='./'+fixed_im_name+'.nii.gz').save(type=t)
        moving_im_name = 'moving_im'
        Image(param=np.asarray(moving_im), absolutepath='./'+moving_im_name+'.nii.gz').save(type=t)

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
            # metric_params = ',1,4'  # for MeanSquares
            metric_params = ',1,2'  # for MI
            niter = 5
            smooth = 0
            shrink = 1
            cmd_reg = 'isct_antsRegistration -d 2 -n ' + reg_interpolation + ' -t ' + transfo_type + '[' + str(gradientstep) + transfo_params + '] ' \
                      '-m ' + metric + '[' + fixed_im_name + '.nii.gz,' + moving_im_name + '.nii.gz ' + metric_params + '] -o reg  -c ' + str(niter) + \
                      ' -s ' + str(smooth) + ' -f ' + str(shrink) + ' -v ' + str(verbose)  # + ' -r [' + fixed_im_name + '.nii.gz,' + moving_im_name + '.nii.gz ' + ',1]'

            sct.run(cmd_reg, verbose=verbose)

            sct.run('cp ' + mat_name + ' ../' + path + transfo_dir + '/'+transfo_name, verbose=verbose)
            if 'SyN' in transfo_type:
                sct.run('cp ' + inverse_mat_name + ' ../' + path + transfo_dir + '/'+transfo_name + '_inversed',
                        verbose=verbose)

        if apply_transfo:
            if not search_reg:
                os.chdir('..')
                sct.run('cp ' + path + transfo_dir + '/' + transfo_name + ' ./' + dir_name + mat_name, verbose=verbose)
                if 'SyN' in transfo_type:
                    sct.run('cp ' + path + transfo_dir + '/' + transfo_name + '_inversed' + ' ./' + dir_name + inverse_mat_name,
                            verbose=verbose)
                os.chdir('./' + dir_name)

            if binary or moving_im.max() == 1 or fixed_im.max() == 1:
                apply_transfo_interpolation = 'NearestNeighbor'
            else:
                apply_transfo_interpolation = 'BSpline'

            if 'SyN' in transfo_type and inverse:
                cmd_apply = 'isct_antsApplyTransforms -d 2 -i ' + moving_im_name + '.nii.gz -o ' + moving_im_name + '_moved.nii.gz ' \
                            '-n ' + apply_transfo_interpolation + ' -t [' + inverse_mat_name + '] ' \
                            '-r ' + fixed_im_name + '.nii.gz -v ' + str(verbose)

            else:
                cmd_apply = 'isct_antsApplyTransforms -d 2 -i ' + moving_im_name + '.nii.gz -o ' + moving_im_name + '_moved.nii.gz ' \
                            '-n ' + apply_transfo_interpolation + ' -t [' + mat_name + ',' + str(inverse) + '] ' \
                            '-r ' + fixed_im_name + '.nii.gz -v ' + str(verbose)

            sct.run(cmd_apply, verbose=verbose)

            res_im = Image(moving_im_name + '_moved.nii.gz')
        os.chdir('..')
    except Exception, e:
        sct.printv('WARNING: AN ERROR OCCURRED WHEN DOING RIGID REGISTRATION USING ANTs', 1, 'warning')
        print e
    else:
        sct.printv('Removing temporary files ...', verbose=verbose, type='normal')
        #os.chdir('..')
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


# ------------------------------------------------------------------------------------------------------------------
def l0_norm(x, y):
    """
    L0 norm of two images x and y (used to compute tau)
    :param x:
    :param y:
    :return: l0 norm
    """
    return np.linalg.norm(x.flatten() - y.flatten(), 0)


# ------------------------------------------------------------------------------------------------------------------
def compute_majority_vote_mean_seg(seg_data_set, threshold=0.5, weights=None, type='binary'):
    """
    Compute the mean segmentation image for a given segmentation data set seg_data_set by Majority Vote

    :param seg_data_set: data set of segmentation slices (2D)

    :param threshold: threshold to select the value of a pixel
    :return:
    """
    if weights is None:
        average = np.sum(seg_data_set, axis=0) / float(len(seg_data_set))
    else:
        average = np.sum(np.einsum('ijk,i->ijk', seg_data_set, weights), axis=0)
    # average[average > 1] = 1
    if type == 'binary':
        return (average >= threshold).astype(int)
    else:
        return average.astype(float)


# ------------------------------------------------------------------------------------------------------------------
def correct_wmseg(res_gmseg, original_im, name_wm_seg, hdr):
    '''
    correct WM seg edges using input SC seg
    *** NOT USED ANYMORE***
    '''
    wmseg_dat = (original_im.data > 0).astype(int) - res_gmseg.data

    # corrected_wm_seg = Image(param=(wmseg_dat > 0).astype(int))
    corrected_wm_seg = Image(param=wmseg_dat)

    corrected_wm_seg.file_name = name_wm_seg + '_corrected'
    corrected_wm_seg.ext = '.nii.gz'
    corrected_wm_seg.hdr = hdr
    corrected_wm_seg.save()

    # sct.run('fslmaths ' + corrected_wm_seg.file_name + '.nii.gz -thr 0 ' + corrected_wm_seg.file_name + '.nii.gz')
    sct.run('sct_maths -i ' + corrected_wm_seg.file_name + '.nii.gz -thr 0 -o ' + corrected_wm_seg.file_name + '.nii.gz')

    return corrected_wm_seg

# ------------------------------------------------------------------------------------------------------------------
def get_all_seg_from_dic(slices, type='wm'):
    """
    get a list of all manual segmentations in a list od slices that have multiple manual segmentations per slice
    :return:
    """
    if type  == 'wm':
        list_seg_by_slice = [dic_slice.wm_seg for dic_slice in slices]
    elif type == 'gm':
        list_seg_by_slice = [dic_slice.gm_seg for dic_slice in slices]
    elif type.lower() == 'wm_m':
        list_seg_by_slice = [dic_slice.wm_seg_M for dic_slice in slices]
    elif type.lower() == 'gm_m':
        list_seg_by_slice = [dic_slice.gm_seg_M for dic_slice in slices]

    list_all_seg = []
    for list_seg in list_seg_by_slice:
        for seg in list_seg:
            list_all_seg.append(seg)

    return np.asarray(list_all_seg)


# ------------------------------------------------------------------------------------------------------------------
def load_level(level_file, type='int', verbose=1):
    """
    Find the vertebral level of the target image slice(s) for a level image (or a string if the target is 2D)
    :param level_file: image or text file containing level information
    :param target_slices: list of slices in which the level needs to be set
    :return None: the target level is set in the function
    """
    dic_level_by_i = {}
    if isinstance(level_file, Image):
        subject_levels = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: []}
        for i_level_slice, level_slice in enumerate(level_file.data):
            try:
                l = int(round(np.mean(level_slice[level_slice > 0])))
                dic_level_by_i[i_level_slice] = l
                subject_levels[l].append(i_level_slice)
            except Exception, e:
                sct.printv('WARNING: ' + str(e) + '\nNo level label for slice ' + str(i_level_slice) + ' of target', verbose, 'warning')
                dic_level_by_i[i_level_slice] = 0

        if type == 'float':
            for int_level, slices_list in subject_levels.items():
                n_slices_by_level = len(slices_list)
                if n_slices_by_level == 1:
                    index = slices_list[0]
                    if index == 0:
                        dic_level_by_i[index] = int_level+0.1
                    elif index == len(level_file.data)-1:
                        dic_level_by_i[index] = int_level+0.9
                    else:
                        dic_level_by_i[index] = int_level+0.5
                elif n_slices_by_level > 1:
                    gap = 1.0/(n_slices_by_level + 1)
                    for i, index in enumerate(slices_list):
                        dic_level_by_i[index] = int_level+((n_slices_by_level-i)*gap)

    elif isinstance(level_file, str):
        if os.path.isfile(level_file):
            assert sct.extract_fname(level_file)[2] == '.txt', 'ERROR: the level file is nor an image nor a text file ...'
            level_txt_file = open(level_file, 'r')
            lines = level_txt_file.readlines()
            level_txt_file.close()

            for line in lines[1:]:
                i_slice, level = line.split(',')
                level = int(level[:-1])
                i_slice = int(i_slice)
                dic_level_by_i[i_slice] = level
        else:
            # WARNING: This should be the same dictionary as in the class ModelDictionary.level_label
            level_label_dic = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7', 15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5'}
            dic_level_by_i[0] = get_key_from_val(level_label_dic, level_file.upper())

    list_level_by_slice = [dic_level_by_i[i] for i in sorted(dic_level_by_i.keys())]

    return list_level_by_slice



########################################################################################################################
# ----------------------------------------------- OTHER UTILS FUNCTIONS ---------------------------------------------- #
########################################################################################################################

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
    """
    Check if the input is really a file and change the type to nii.gz if different

    :param file_name: name to check

    :param verbose:

    :return: file_name if the input is really a file, False otherwise
    """
    ext = '.nii.gz'
    if os.path.isfile(file_name):
        if sct.extract_fname(file_name)[2] != ext:
            # sct.run('fslchfiletype NIFTI_GZ ' + file_name)
            new_file_name = sct.extract_fname(file_name)[1] + ext
            sct.run('sct_convert -i ' + file_name + ' -o ' + new_file_name)
        else:
            new_file_name = file_name
        return new_file_name
    else:
        sct.printv('WARNING: ' + file_name + ' is not a file ...', verbose, 'warning')
        return False


# ------------------------------------------------------------------------------------------------------------------
def save_dic_slices(path_to_model):
    """
    Save all the dictionary slices of a computed model

    :param path_to_model: path to the compute model used to load the dictionary
    """
    import pickle, gzip
    level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}
    model_slices_list = pickle.load(gzip.open(path_to_model + '/dictionary_slices.pklz', 'rb'))
    sct.run('mkdir ' + path_to_model + '/model_slices')
    model_slices = [Slice(slice_id=i_slice, level=dic_slice[3], im_m=dic_slice[0], wm_seg_m=dic_slice[1], gm_seg_m=dic_slice[2], im_m_flat=dic_slice[0].flatten(),  wm_seg_m_flat=dic_slice[1].flatten()) for i_slice, dic_slice in enumerate(model_slices_list)]  # type: list of slices

    for mod_slice in model_slices:
        Image(param=mod_slice.im_M, absolutepath= path_to_model + '/model_slices/slice' + str(mod_slice.id) + '_' + level_label[mod_slice.level] + '_im_m.nii.gz').save()
        Image(param=mod_slice.wm_seg_M, absolutepath= path_to_model + '/model_slices/slice' + str(mod_slice.id) + '_' + level_label[mod_slice.level] + '_wmseg_m.nii.gz').save()


# ------------------------------------------------------------------------------------------------------------------
def extract_metric_from_slice_set(slices_set, seg_to_use=None, metric='Mean', gm_percentile=0.03, wm_percentile=0.05, save=False, output='metric_in_dictionary.txt'):
    """
    uses the registereg images and GM segmentation (or another segmentation)to extract mean intensity values in the WM dn GM
    :param slices_set:
    :param seg_to_use: must be of GM not WM
    :param gm_percentile:
    :param wm_percentile:
    :param save:
    :return:
    """
    import copy
    level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}
    if save:
        f = open(output, 'w')
        f.write('Slice id - Slice level - '+metric+' in WM - '+metric+' in GM - Std in WM - Std in GM\n')
    else:
        f = None
    slice_set_metric = {}
    for i, slice_i in enumerate(slices_set):
        gm_dat = copy.deepcopy(slice_i.im_M)
        wm_dat = copy.deepcopy(slice_i.im_M)

        # mask with the gray matter segmentation
        if seg_to_use is None:
            mean_gm_seg_M = compute_majority_vote_mean_seg(slice_i.gm_seg_M)
            gm_dat[mean_gm_seg_M == 0] = 0
            wm_dat[mean_gm_seg_M == 1] = 0
        else:
            gm_dat[seg_to_use[i] == 0] = 0
            wm_dat[seg_to_use[i] == 1] = 0

        # threshold to 0.1 to get rid of the (almost) null values
        gm_dat = gm_dat[gm_dat > 0.1]
        wm_dat = wm_dat[wm_dat > 0.1]

        # metric in GM
        if gm_percentile != 0:
            # removing outliers with a percentile
            gm_dat = sorted(gm_dat.flatten())
            n_gm_outliers = int(round(gm_percentile*len(gm_dat)/2.0))
            if n_gm_outliers != 0 and n_gm_outliers*2 < len(gm_dat):
                gm_dat = gm_dat[n_gm_outliers:-n_gm_outliers]

        if gm_dat == []:
            gm_met = 0
            gm_std = 0
        else:
            if metric.lower() == 'mean':
                gm_met = np.mean(gm_dat)
            elif metric.lower() == 'median':
                gm_met = np.median(gm_dat)
            gm_std = np.std(gm_dat)

        # metric in WM
        if wm_percentile != 0:
            # removing outliers with a percentile
            wm_dat = sorted(wm_dat.flatten())
            n_wm_outliers = int(round(wm_percentile*len(wm_dat)/2.0))
            if n_wm_outliers != 0 and n_wm_outliers*2 < len(wm_dat):
                wm_dat = wm_dat[n_wm_outliers:-n_wm_outliers]
        if wm_dat == []:
            wm_met = 0
            wm_std = 0
        else:
            if metric.lower() == 'mean':
                wm_met = np.mean(wm_dat)
            elif metric.lower() == 'median':
                wm_met = np.median(wm_dat)
            wm_std = np.std(wm_dat)

        slice_set_metric[slice_i.id] = (wm_met, gm_met, wm_std, gm_std)
        if save:
            f.write(str(slice_i.id) + ' - ' + level_label[int(slice_i.level)] + ' - ' + str(wm_met) + ' - ' + str(gm_met) + ' - ' + str(wm_std) + ' - ' + str(gm_std) + '\n')
    if save:
        f.close()
    return slice_set_metric



########################################################################################################################
# -------------------------------------------------- PRETREATMENTS --------------------------------------------------- #
########################################################################################################################

# ------------------------------------------------------------------------------------------------------------------
def crop_t2_star_pipeline(path, box_size=75):
    """
    Pretreatment pipeline croping the t2star files from a dataset around the spinal cord and both croped t2star and gray matter manual segmentation to a 75*75 squared image

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

            for subject_file in os.listdir(path + '/' + subject_dir):
                file_low = subject_file.lower()
                if 't2star' in file_low or 'im' in file_low  in file_low and 'mask' not in file_low and 'seg' not in file_low and 'irp' not in file_low:
                    t2star = subject_file
                    t2star_path, t2star_name, ext = sct.extract_fname(t2star)
                elif 'square' in file_low and 'mask' in file_low and 'irp' not in file_low:
                    mask_box = subject_file
                elif 'seg' in file_low and 'in' not in file_low and 'croped' not in file_low and 'gm' not in file_low and 'irp' not in file_low:
                    sc_seg = subject_file
                elif 'gm' in file_low and 'croped.nii' not in file_low and 'irp' not in file_low:
                    manual_seg = subject_file
                    manual_seg_name = sct.extract_fname(manual_seg)[1]
                elif '_croped.nii' in file_low and 'gm' in file_low and 'irp' not in file_low:
                    manual_seg_croped = subject_file

            if t2star != '' and sc_seg != '':
                subject_path = path + '/' + subject_dir + '/'
                os.chdir(subject_path)
                ext = '.nii.gz'
                try:
                    if seg_in == '':
                        seg_in_name = t2star_name + '_seg_in'
                        seg_in = seg_in_name + ext
                        sct.run('sct_crop_image -i ' + t2star + ' -m ' + sc_seg + ' -b 0 -o ' + seg_in)

                    if mask_box == '':
                        mask_box = t2star_name + '_square_mask_from_sc_seg'+ext
                        sct.run('sct_create_mask -i ' + seg_in + ' -p centerline,' + sc_seg + ' -size ' + str(box_size) + ' -o ' + mask_box + ' -f box')

                    if seg_in_croped == '':
                        seg_in_im = Image(seg_in)
                        mask_im = Image(mask_box)
                        seg_in_im.crop_and_stack(mask_im, suffix='_croped', save=True)
                        seg_in_name += '_croped'
                        seg_in_im.setFileName(seg_in_name +ext)
                        seg_in_im = set_orientation(seg_in_im, 'IRP')

                    if manual_seg_croped == '':
                        manual_seg_im = Image(manual_seg)
                        mask_im = Image(mask_box)
                        manual_seg_im.crop_and_stack(mask_im, suffix='_croped', save=True)
                        manual_seg_name += '_croped'
                        manual_seg_im.setFileName(manual_seg_name+ext)
                        manual_seg_im = set_orientation(manual_seg_im, 'IRP')

                except Exception, e:
                    sct.printv('WARNING: an error occured ... \n ' + str(e), 1, 'warning')
                else:
                    print 'Done !'
                os.chdir('..')


# ------------------------------------------------------------------------------------------------------------------
def crop_t2_star(t2star, sc_seg, box_size=75):
    """
    Pretreatment function croping the t2star file around the spinal cord and to a 75*75 squared image

    :param t2star: t2star image to be croped

    :param sc_seg: segmentation of the spinal cord
    """
    t2star_name = sct.extract_fname(t2star)[1]
    sc_seg_name = sct.extract_fname(sc_seg)[1]
    mask_box = None
    fname_seg_in_IRP = None

    try:
        ext = '.nii.gz'
        seg_in_name = t2star_name + '_seg_in'
        seg_in = seg_in_name + ext
        sct.run('sct_crop_image -i ' + t2star + ' -m ' + sc_seg + ' -b 0 -o ' + seg_in)

        mask_box = t2star_name + '_square_mask_from_sc_seg'+ext
        sct.run('sct_create_mask -i ' + seg_in + ' -p centerline,' + sc_seg + ' -size ' + str(box_size) + ' -o ' + mask_box + ' -f box')

        seg_in_im = Image(seg_in)
        mask_im = Image(mask_box)
        seg_in_im.crop_and_stack(mask_im, suffix='_croped', save=True)
        seg_in_name += '_croped'
        seg_in_im.setFileName(seg_in_name +ext)
        seg_in_im = set_orientation(seg_in_im, 'IRP')

        fname_seg_in_IRP = seg_in_name+'_IRP'+ext

        if t2star_name + '_square_mask_from_sc_seg_IRP.nii.gz' in os.listdir('.'):
            mask_box = t2star_name + '_square_mask_from_sc_seg_IRP.nii.gz'

    except Exception, e:
        sct.printv('WARNING: an error occured when croping ' + t2star_name + '... \n ' + str(e), 1, 'warning')
    return mask_box, fname_seg_in_IRP


# ------------------------------------------------------------------------------------------------------------------
def save_by_slice(dic_dir):
    """
    from a dictionary containing for each subject a 3D image crop around the spinal cord,
     a graymatter segmentation 3D image, and a level image (from the registration of the template to the T2star image)

     save an image per slice including the level in the image name

    :param dic_dir: dictionary directory
    """
    if dic_dir[-1] == '/':
        dic_by_slice_dir = dic_dir[:-1] + '_by_slice/'
    else:
        dic_by_slice_dir = dic_dir + '_by_slice/'

    sct.run('mkdir ' + dic_by_slice_dir)

    for subject_dir in os.listdir(dic_dir):
        subject_path = dic_dir + '/' + subject_dir
        if os.path.isdir(subject_path):
            sct.run('mkdir ' + dic_by_slice_dir + subject_dir)
            i_manual_gm = 0
            # getting the level file
            path_file_levels = None
            level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7', 15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5'}
            for file_name in os.listdir(subject_path):
                if 'level' in file_name:
                    path_file_levels = subject_path + '/' + file_name
                    '''
                    if 'IRP' not in file_name:
                        path_file_levels_IRP = sct.add_suffix(path_file_levels, '_IRP')
                        sct.run('sct_image -i ' + subject_path + '/' + file_name + ' -setorient IRP -o '+path_file_levels_IRP)
                        path_file_levels = path_file_levels_IRP # subject_path + '/' + sct.extract_fname(file_name)[1] + '_IRP.nii.gz'
                    '''
            if path_file_levels is None and 'label' in os.listdir(subject_path):
                '''
                if 'MNI-Poly-AMU_level_IRP.nii.gz' not in sct.run('ls ' + subject_path + '/label/template')[1]:
                    sct.run('sct_image -i ' + subject_path + '/label/template/MNI-Poly-AMU_level.nii.gz -setorient IRP')
                '''
                path_file_levels = subject_path + '/label/template/MNI-Poly-AMU_level.nii.gz'

            elif path_file_levels is not None:
                path_level, file_level, ext_level = sct.extract_fname(path_file_levels)
                if ext_level == '.nii.gz' or ext_level == '.nii':
                    level_info_file = Image(path_file_levels)
                    level_info_file.change_orientation('IRP')
                elif ext_level == '.txt':
                    level_info_file = path_file_levels
                else:
                    level_info_file = None
                    sct.printv('WARNING: no level file for subject '+subject_dir, 1, 'warning')

                list_level_by_slice = load_level(level_info_file)
                '''
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
                '''

            for file_name in os.listdir(subject_path):
                if 'seg_in' in file_name and 'croped' in file_name and 'IRP' in file_name:
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
                            fname_slice = dic_by_slice_dir + subject_dir + '/' + subject_dir + '_slice' + i_slice_str + '_im.nii.gz'
                            im_slice = Image(param=im_slice, absolutepath=fname_slice, hdr=im.hdr)

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
                            fname_slice = dic_by_slice_dir + subject_dir + '/' + subject_dir + '_slice' + i_slice_str + '_' + level_label[list_level_by_slice[i_slice]] + '_im.nii.gz'
                            im_slice = Image(param=im_slice, absolutepath=fname_slice, hdr=im.hdr)

                            if len(im_slice.hdr.get_zooms()) == 3:
                                im_slice.hdr.set_zooms(slice_zoom)
                            im_slice.save()

                if 'gm' in file_name and 'croped' in file_name and 'IRP' in file_name:
                    seg = Image(dic_dir + '/' + subject_dir + '/' + file_name)
                    seg_zooms = seg.hdr.get_zooms()
                    slice_zoom = (seg_zooms[1], seg_zooms[2], seg_zooms[0])
                    if path_file_levels is None:
                        for i_slice, seg_slice in enumerate(seg.data):
                            if i_slice < 10:
                                i_slice_str = str(i_slice)
                                i_slice_str = '0' + i_slice_str
                            else:
                                i_slice_str = str(i_slice)
                            seg_slice = Image(param=seg_slice, absolutepath=dic_by_slice_dir + subject_dir + '/' + subject_dir + '_slice' + i_slice_str + '_' + level_label[list_level_by_slice[i_slice]] + '_manual_gmseg_rater'+str(i_manual_gm)+'.nii.gz', hdr=seg.hdr)

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
                                  '_slice' + i_slice_str + '_' + level_label[list_level_by_slice[i_slice]] + '_manual_gmseg_rater'+str(i_manual_gm)+'.nii.gz', hdr=seg.hdr)

                            if len(seg_slice.hdr.get_zooms()) == 3:
                                seg_slice.hdr.set_zooms(slice_zoom)
                            seg_slice.save()
                    i_manual_gm += 1


# ------------------------------------------------------------------------------------------------------------------
def resample_image(fname, suffix='_resampled.nii.gz', binary=False, npx=0.3, npy=0.3, thr=0.0, interpolation='spline'):
    """
    Resampling function: add a padding, resample, crop the padding
    :param fname: name of the image file to be resampled
    :param suffix: suffix added to the original fname after resampling
    :param binary: boolean, image is binary or not
    :param npx: new pixel size in the x direction
    :param npy: new pixel size in the y direction
    :param thr: if the image is binary, it will be thresholded at thr (default=0) after the resampling
    :param interpolation: type of interpolation used for the resampling
    :return: file name after resampling (or original fname if it was already in the correct resolution)
    """
    im_in = Image(fname)
    orientation = get_orientation_3d(im_in)
    if orientation != 'RPI':
        im_in = set_orientation(im_in, 'RPI')
        im_in.save()
        fname = im_in.absolutepath
    nx, ny, nz, nt, px, py, pz, pt = im_in.dim

    if round(px, 2) != round(npx, 2) or round(py, 2) != round(npy, 2):
        name_resample = sct.extract_fname(fname)[1] + suffix
        if binary:
            interpolation = 'nn'

        sct.run('sct_resample -i '+fname+' -mm '+str(npx)+'x'+str(npy)+'x'+str(pz)+' -o '+name_resample+' -x '+interpolation)

        if binary:
            sct.run('sct_maths -i ' + name_resample + ' -thr ' + str(thr) + ' -o ' + name_resample)
            sct.run('sct_maths -i ' + name_resample + ' -bin -o ' + name_resample)

        if orientation != 'RPI':
            im_resample = Image(name_resample)
            im_resample = set_orientation(im_resample, orientation)
            im_resample.save()
            name_resample = im_resample.absolutepath
        return name_resample
    else:
        if orientation != 'RPI':
            im_in = set_orientation(im_in, orientation)
            im_in.save()
            fname = im_in.absolutepath
        sct.printv('Image resolution already ' + str(npx) + 'x' + str(npy) + 'xpz')
        return fname


# ------------------------------------------------------------------------------------------------------------------
def dataset_preprocessing(path_to_dataset, denoise=True):
    """
    preprocessing function for a dataset of 3D images to be integrated to the model
    the dataset should contain for each subject :
        - a T2*-w image containing 'im' in its name
        - a segmentation of the spinal cord containing 'seg' in its name
        - a manual segmentation of the gray matter containing 'gm' in its name (or several manual segmentations from different raters)
        - a 'level image' containing 'level' in its name : the level image is an image containing a level label per slice indicating at wich vertebral level correspond this slice
    :param path:
    """
    from copy import deepcopy
    axial_pix_dim = 0.3
    model_image_size = 75
    interpolation = 'spline'  # 'Cubic'
    original_path = os.path.abspath('.')
    for subject_dir in os.listdir(path_to_dataset):
        if os.path.isdir(path_to_dataset + '/' + subject_dir):
            os.chdir(path_to_dataset + '/' + subject_dir)
            # getting the subject images
            fname_t2star = ''
            fname_scseg = ''
            list_fname_gmseg = []
            for file_name in os.listdir('.'):
                if 'im' in file_name:
                    fname_t2star = file_name
                elif 'gm' in file_name:
                    list_fname_gmseg.append(file_name)
                elif 'seg' in file_name and 'gm' not in file_name:
                    fname_scseg = file_name

            list_new_names = []
            list_fnames = deepcopy(list_fname_gmseg)
            list_fnames.append(fname_scseg)
            list_fnames.append(fname_t2star)

            for f_name in list_fnames:
                im = Image(f_name)
                orientation = get_orientation_3d(im)
                if orientation != 'RPI':
                    im = set_orientation(im, 'RPI')
                    list_new_names.append(im.absolutepath)
                    im.save()
                    # new_names.append(output.split(':')[1][1:-1])
                else:
                    list_new_names.append(f_name)

            fname_t2star = list_new_names[-1]
            fname_scseg = list_new_names[-2]
            list_fname_gmseg = list_new_names[:-2]

            fname_t2star = resample_image(fname_t2star, npx=axial_pix_dim, npy=axial_pix_dim, interpolation=interpolation)
            fname_scseg = resample_image(fname_scseg, npx=axial_pix_dim, npy=axial_pix_dim, binary=True, interpolation='nn')
            list_fname_gmseg = [resample_image(fname_gmseg, npx=axial_pix_dim, npy=axial_pix_dim, binary=True, interpolation='nn') for fname_gmseg in list_fname_gmseg]



            if denoise:
                from sct_maths import denoise_ornlm
                t2star_im = Image(fname_t2star)
                t2star_im.data = denoise_ornlm(t2star_im.data)
                t2star_im.save()

            mask_box, fname_seg_in_IRP = crop_t2_star(fname_t2star, fname_scseg, box_size=model_image_size)

            for fname_gmseg in list_fname_gmseg:
                im_gmseg = Image(fname_gmseg)
                im_mask = Image(mask_box)
                im_gmseg.crop_and_stack(im_mask, suffix='_croped', save=True)
                sct.run('sct_image -i '+sct.extract_fname(fname_gmseg)[1] + '_croped.nii.gz -setorient IRP')

            os.chdir(original_path)
    save_by_slice(path_to_dataset)


# ------------------------------------------------------------------------------------------------------------------
def compute_level_file(t2star_fname, t2star_sc_seg_fname , t2_fname, t2_seg_fname, landmarks_fname):
    """
    mini registration pipeline to get the vertebral levels for a T2 star using the anatomic image (T2) and a segmentation of the spinal cord
    :param t2star_fname:
    :param t2star_sc_seg_fname:
    :param t2_fname:
    :param t2_seg_fname:
    :param landmarks_fname:
    :return: path ofr the level image
    """
    # Registration to template
    cmd_register_template = 'sct_register_to_template -i ' + t2_fname + ' -s ' + t2_seg_fname + ' -l ' + landmarks_fname
    sct.run(cmd_register_template)

    cmd_warp_template = 'sct_warp_template -d ' + t2_fname + ' -w warp_template2anat.nii.gz -a 0'
    sct.run(cmd_warp_template)

    # Registration template to t2star
    cmd_register_multimodal = 'sct_register_multimodal -i template2anat.nii.gz -d ' + t2star_fname + ' -iseg ./label/template/MNI-Poly-AMU_cord.nii.gz -dseg ' + t2star_sc_seg_fname + ' -param step=1,type=seg,algo=syn,metric=MeanSquares,iter=5:step=2,type=im,algo=slicereg,metric=MeanSquares,iter=5'
    sct.run(cmd_register_multimodal)

    multimodal_warp_name = 'warp_template2anat2' + t2star_fname
    total_warp_name = 'warp_template2t2star.nii.gz'
    cmd_concat = 'sct_concat_transfo -w warp_template2anat.nii.gz,' + multimodal_warp_name + ' -d ' + t2star_fname + ' -o ' + total_warp_name
    sct.run(cmd_concat)

    cmd_warp = 'sct_warp_template -d ' + t2star_fname + ' -w ' + total_warp_name + ' -a 0 '
    sct.run(cmd_warp)

    sct.run('sct_image -i ./label/template/MNI-Poly-AMU_level.nii.gz -setorient IRP')

    return 'MNI-Poly-AMU_level_IRP.nii.gz'



########################################################################################################################
# ------------------------------------------------- POST-TREATMENTS -------------------------------------------------- #
########################################################################################################################

# ------------------------------------------------------------------------------------------------------------------
def inverse_square_crop(im_croped, im_square_mask):
    if im_square_mask.orientation != 'IRP':
        im_square_mask = set_orientation(im_square_mask, 'IRP')
    assert len(im_croped.data) == len(im_square_mask.data)

    im_inverse_croped = Image(np.zeros(im_square_mask.data.shape), hdr=im_square_mask.hdr)
    for i_slice, slice_croped in enumerate(im_croped.data):
        i_croped = 0
        inside = False
        for i, row in enumerate(im_square_mask.data[i_slice] == 1):
            j_croped = 0
            for j, in_square in enumerate(row):
                if in_square:
                    im_inverse_croped.data[i_slice][i][j] = slice_croped[i_croped][j_croped]
                    j_croped += 1
                    if not inside:
                        # getting inside the mask
                        inside = True
                elif inside:  # if we are getting out of the mask:
                    i_croped += 1
                    inside = False
    im_inverse_croped.setFileName(im_croped.file_name + '_original_dimension' + im_croped.ext)

    return im_inverse_croped


########################################################################################################################
# --------------------------------------------------- VALIDATION ----------------------------------------------------- #
########################################################################################################################

# ------------------------------------------------------------------------------------------------------------------
def leave_one_out_by_subject(dic_path, dic_3d, denoising=True, reg='Affine', metric='MI', use_levels=True, weight=2.5, eq=1, mode_weighted_sim=False, weighted_label_fusion=False):
    """
    Leave one out cross validation taking 1 SUBJECT out of the dictionary at each step
    and computing the resulting dice coefficient, the time of computation and an error map

    :param dic_path: path to the dictionary to use to do the model validation

    """
    import time
    from msct_multiatlas_seg import Model, SegmentationParam, SupervisedSegmentationMethod
    from sct_segment_graymatter import FullGmSegmentation
    init = time.time()

    wm_dice_file = open('wm_dice_coeff.txt', 'w')
    gm_dice_file = open('gm_dice_coeff.txt', 'w')
    wm_csa_file = open('wm_csa.txt', 'w')
    gm_csa_file = open('gm_csa.txt', 'w')
    hd_file = open('hd.txt', 'w')
    n_slices = 0
    e = None

    level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}
    # for the error map
    gm_diff_by_level = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': [], '': []}
    wm_diff_by_level = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': [], '': []}

    for subject_dir in os.listdir(dic_path):
        subject_path = dic_path + '/' + subject_dir
        if os.path.isdir(subject_path):
            try:
                tmp_dir = 'tmp_' + subject_dir + '_as_target'
                sct.run('mkdir ' + tmp_dir)

                tmp_dic_name = 'dic'
                sct.run('cp -r ' + dic_path + ' ./' + tmp_dir + '/' + tmp_dic_name + '/')
                sct.run('cp -r ' + dic_3d + '/' + subject_dir + ' ./' + tmp_dir + '/')
                sct.run('mv ./' + tmp_dir + '/' + tmp_dic_name + '/' + subject_dir + ' ./' + tmp_dir + '/' + subject_dir + '_by_slice')

                # Gray matter segmentation using this subject as target
                os.chdir(tmp_dir)
                model_param = SegmentationParam()
                model_param.path_model = tmp_dic_name
                model_param.todo_model = 'compute'
                model_param.weight_gamma = float(weight)
                model_param.use_levels = use_levels
                model_param.res_type = 'prob'
                model_param.reg = reg.split(':')
                model_param.reg_metric = metric
                model_param.equation_id = eq
                model_param.mode_weight_similarity = mode_weighted_sim
                model_param.weight_label_fusion = weighted_label_fusion
                model_param.target_denoising = denoising

                model = Model(model_param=model_param, k=0.8)

                n_slices_model = model.dictionary.J

                target = ''
                sc_seg = ''
                ref_gm_seg_im = ''
                level = ''
                for file_name in os.listdir(subject_dir):  # 3d files

                    if 'im' in file_name:
                        target = subject_dir + '/' + file_name
                    elif 'level' in file_name:
                        level = subject_dir + '/' + file_name
                    elif 'gm' in file_name:
                        ref_gm_seg_im = subject_dir + '/' + file_name
                    elif 'seg' in file_name:
                        sc_seg = subject_dir + '/' + file_name


                full_gmseg = FullGmSegmentation(target, sc_seg, None, level, ref_gm_seg=ref_gm_seg_im, model=model, param=model_param)

                # ## VALIDATION ##
                # Dice coeff
                subject_dice = open(full_gmseg.dice_name, 'r')
                dice_lines_list = subject_dice.readlines()
                subject_dice.close()

                gm_dices = None
                wm_dices = None
                n_subject_slices = len(full_gmseg.gm_seg.target_seg_methods.target)
                n_slices += n_subject_slices

                for i, line in enumerate(dice_lines_list):
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

                # hausdorff distance
                subject_hd = open(full_gmseg.hausdorff_name, 'r')
                hd_res_list = subject_hd.readlines()[1:-4]
                for line in hd_res_list:
                    n_slice, res_slice = line.split(':')
                    hd, med1, med2 = res_slice.split('-')
                    n_slice = n_slice[-1:]
                    med1 = float(med1)
                    med2 = float(med2[:-2])

                    if int(n_slice) < 10:
                        target_slice = 'slice0' + n_slice
                    else:
                        target_slice = 'slice' + n_slice
                    slice_level = subject_slices_levels[target_slice]
                    hd_file.write(subject_dir + ' ' + target_slice + ' ' + slice_level + ': ' + str(hd) + ' - ' + str(max(med1, med2)) + '\n')

                # error map

                print 'ERROR MAP BY LEVEL COMPUTATION'
                path_validation = full_gmseg.tmp_dir + '/validation/'
                ref_gm_seg_im = Image(path_validation + 'ref_gm_seg.nii.gz')
                ref_wm_seg_im = Image(path_validation + 'ref_wm_seg.nii.gz')
                res_gm_seg_im = Image(path_validation + 'res_gm_seg_bin_RPI.nii.gz')
                res_wm_seg_im = Image(path_validation + 'res_wm_seg_bin_RPI.nii.gz')

                ref_gm_seg_im.change_orientation('IRP')
                ref_wm_seg_im.change_orientation('IRP')
                res_gm_seg_im.change_orientation('IRP')
                res_wm_seg_im.change_orientation('IRP')

                for i_slice in range(len(ref_gm_seg_im.data)):
                    slice_gm_error = abs(ref_gm_seg_im.data[i_slice] - res_gm_seg_im.data[i_slice])
                    slice_wm_error = abs(ref_wm_seg_im.data[i_slice] - res_wm_seg_im.data[i_slice])
                    if int(i_slice) < 10:
                        target_slice = 'slice0' + str(i_slice)
                    else:
                        target_slice = 'slice' + str(i_slice)
                    slice_level = subject_slices_levels[target_slice]
                    gm_diff_by_level[slice_level].append(slice_gm_error)
                    wm_diff_by_level[slice_level].append(slice_wm_error)

                # csa
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
    # error map
    for l, level_error in gm_diff_by_level.items():
        try:
            n = len(level_error)
            if n != 0:
                Image(param=sum(level_error)/n, absolutepath='error_map_gm_' + str(l) + '.nii.gz').save()
        except ZeroDivisionError:
            sct.printv('WARNING: no data for level ' + str(l), 1, 'warning')

    for l, level_error in wm_diff_by_level.items():
        try:
            n = len(level_error)
            if n != 0:
                Image(param=sum(level_error)/n, absolutepath='error_map_wm_' + str(l) + '.nii.gz').save()
        except ZeroDivisionError:
            sct.printv('WARNING: no data for level ' + str(l), 1, 'warning')

    if e is None:
        wm_dice_file.close()
        gm_dice_file.close()
        wm_csa_file.close()
        gm_csa_file.close()
        hd_file.close()

        # Image(param=error_map_abs_sum/n_slices, absolutepath='error_map_abs.nii.gz').save()
        t = time.time() - init
        print 'Done in ' + str(t) + ' sec'


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
'''
import os
os.chdir('/Volumes/folder_shared/greymattersegmentation/supervised_gmseg_method/res/2015-06-23-LOOCV_CSA')
from msct_gmseg_utils import compute_error_map_by_level
compute_error_map_by_level('with_levels_gamma1.2_Affine_Zregularisation')
'''


# ------------------------------------------------------------------------------------------------------------------
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

            os.chdir(res_dir)
            for fname in os.listdir('.'):
                if 'level' in fname and 't2star' not in fname and 'IRP' in fname:
                    level = fname

                elif 'square_mask' in fname and 'IRP' in fname:
                    sq_mask = fname
                elif 'res_wmseg' in fname and 'corrected' in fname and 'original_dimension' not in fname:
                    res_wm_seg = fname
            if res_wm_seg != '' and level != '' and sq_mask!= '':

                ref_wm_seg = 'validation/ref_wm_seg.nii.gz'
                status, ref_ori = sct.run('sct_image -i ' + ref_wm_seg + ' -getorient')
                # ref_ori = ref_ori[4:7]
                if ref_ori != 'IRP':
                    sct.run('sct_image -i ' + ref_wm_seg + ' -setorient IRP')
                    ref_wm_seg = sct.extract_fname(ref_wm_seg)[0] + sct.extract_fname(ref_wm_seg)[1] + '_IRP' + sct.extract_fname(ref_wm_seg)[2]

                level_im = Image(level)

                ref_wm_seg_im = Image(ref_wm_seg)
                mask_im = Image(sq_mask)
                ref_wm_seg_im.crop_and_stack(mask_im, suffix='_croped', save=True)
                ref_wm_seg = ref_wm_seg_im.absolutepath

                res_wm_seg_im = Image(res_wm_seg)

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
    for l, level_error in diff_by_level.items():
        try:
            n = len(level_error)
            if n != 0:
                Image(param=sum(level_error)/n, absolutepath='error_map_corrected_wm_' + str(l) + '.nii.gz').save()
        except ZeroDivisionError:
            sct.printv('WARNING: no data for level ' + str(l), 1, 'warning')
    Image(param=error_map_abs_sum/n_slices, absolutepath='error_map_abs_corrected_wm.nii.gz').save()
    os.chdir('..')
    os.chdir(original_path)


# ------------------------------------------------------------------------------------------------------------------
def compute_hausdorff_dist_on_loocv_results(data_path):
    import sct_compute_hausdorff_distance as hd
    import xlsxwriter as xl

    original_path = os.path.abspath('.')
    os.chdir(data_path)
    workbook = xl.Workbook('results_hausdorff.xlsx', {'nan_inf_to_errors': True})
    w1 = workbook.add_worksheet('hausdorff_distance')
    w2 = workbook.add_worksheet('median_distance')
    bold = workbook.add_format({'bold': True, 'text_wrap': True})
    w1.write(1, 0, 'Subject', bold)
    w1.write(1, 1, 'Slice #', bold)
    w1.write(1, 2, 'Slice level', bold)
    w2.write(1, 0, 'Subject', bold)
    w2.write(1, 1, 'Slice #', bold)
    w2.write(1, 2, 'Slice level', bold)

    first = True
    col1 = 3
    col2 = 3

    for modality in os.listdir('.'):
        if os.path.isdir(modality) and 'dictionary' not in modality and 'dic' not in modality:
            os.chdir(modality)
            w1.write(0, col1, modality, bold)
            w2.merge_range(0, col2, 0, col2+1, modality, bold)

            row = 2
            for subject_dir in os.listdir('.'):
                if os.path.isdir(subject_dir) and 'dictionary' not in subject_dir and 'dic' not in subject_dir:
                    os.chdir(subject_dir)

                    subject = '_'.join(subject_dir.split('_')[1:3])
                    if 'pilot' in subject:
                        subject = subject.split('_')[0]

                    # ref_gm_seg = ''
                    # ref_sc_seg = ''
                    sq_mask = ''
                    level = ''
                    res_gm_seg = ''
                    res_dir = None
                    for fname in os.listdir('.'):
                        if 'tmp_' + subject in fname and os.path.isdir(fname):
                            res_dir = fname

                    os.chdir(res_dir)
                    for fname in os.listdir('.'):

                        if 'level' in fname and 't2star' not in fname and 'IRP' in fname:
                           level = fname

                        elif 'square_mask' in fname and 'IRP' in fname:
                            sq_mask = fname
                        elif 'res_gmseg' in fname and 'original_dimension' not in fname and 'thinned' not in fname:
                            res_gm_seg = fname
                    if res_gm_seg != '' and sq_mask != '' and level != '':
                        ref_gm_seg = 'ref_gm_seg.nii.gz'
                        status, ref_ori = sct.run('sct_image -i ' + ref_gm_seg+' -getorient')
                        # ref_ori = ref_ori[4:7]
                        if ref_ori != 'IRP':
                            sct.run('sct_image -i ' + ref_gm_seg + ' -setorient IRP')
                            ref_gm_seg = sct.extract_fname(ref_gm_seg)[0] + sct.extract_fname(ref_gm_seg)[1] + '_IRP' + sct.extract_fname(ref_gm_seg)[2]

                        level_im = Image(level)

                        ref_gm_seg_im = Image(ref_gm_seg)
                        mask_im = Image(sq_mask)
                        ref_gm_seg_im.crop_and_stack(mask_im, suffix='_croped', save=True)
                        ref_gm_seg = ref_gm_seg_im.absolutepath

                        # sct.run('fslmaths ' + res_gm_seg + ' -thr 0.5 ' + res_gm_seg)
                        sct.run('sct_maths -i ' + res_gm_seg + ' -thr 0.5 -o ' + res_gm_seg)

                        resample_to = 0.1
                        ref_gm_seg_im = Image(resample_image(ref_gm_seg, binary=True, thr=0.5, npx=resample_to, npy=resample_to))
                        res_gm_seg_im = Image(resample_image(res_gm_seg, binary=True, thr=0.5, npx=resample_to, npy=resample_to))

                        compute_param = hd.Param
                        compute_param.thinning = True
                        compute_param.verbose = 2
                        comp = hd.ComputeDistances(res_gm_seg_im, im2=ref_gm_seg_im, param=compute_param)
                        res_fic = open('../hausdorff_dist.txt', 'w')
                        res_fic.write(comp.res)
                        res_fic.write('\n\nInput 1: ' + res_gm_seg_im.file_name)
                        res_fic.write('\nInput 2: ' + ref_gm_seg_im.file_name)
                        res_fic.close()


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

                        for i_slice in range(len(ref_gm_seg_im.data)):
                            slice_level = level_label[label_by_slice[i_slice]]
                            if first:

                                w1.write(row, 0, subject)
                                w1.write(row, 1, i_slice)
                                w1.write(row, 2, slice_level)

                                w2.write(row, 0, subject)
                                w2.write(row, 1, i_slice)
                                w2.write(row, 2, slice_level)
                            w1.write(row, col1, comp.distances[i_slice].H*comp.dim_pix)

                            med1 = np.median(comp.dist1_distribution[i_slice])
                            med2 = np.median(comp.dist2_distribution[i_slice])
                            w2.write(row, col2, med1*comp.dim_pix)
                            w2.write(row, col2+1, med2*comp.dim_pix)

                            row += 1
                    os.chdir('../..')
            os.chdir('..')
            col1 += 1
            col2 += 2
            first =False
    workbook.close()

    os.chdir('../..')
    os.chdir(original_path)


# ------------------------------------------------------------------------------------------------------------------
def compute_level_similarities(data_path):
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
            os.chdir('..')
    level_file.write('\nmean similarity: ' + str(similarity_sum/n_slices) + '% ')
    level_file.close()
    os.chdir('..')


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    if "-crop" in arguments:
        crop_t2_star_pipeline(arguments['-crop'])
    if "-loocv" in arguments:
        dic_path, dic_3d, denoising, reg, metric, use_levels, weight, eq, mode_weighted_sim, weighted_label_fusion = arguments['-loocv']
        leave_one_out_by_subject(dic_path, dic_3d, denoising=bool(int(denoising)), reg=reg, metric=metric, weight=float(weight), eq=int(eq), mode_weighted_sim=bool(int(mode_weighted_sim)), weighted_label_fusion=bool(int(weighted_label_fusion)))
    if "-error-map" in arguments:
        compute_error_map_by_level(arguments['-error-map'])
    if "-hausdorff" in arguments:
        compute_hausdorff_dist_on_loocv_results(arguments['-hausdorff'])
    if "-save-dic-by-slice" in arguments:
        save_by_slice(arguments['-save-dic-by-slice'])
    if "-preprocess" in arguments:
        dataset_preprocessing(arguments['-preprocess'])
    if "-gmseg-to-wmseg" in arguments:
        gmseg = arguments['-gmseg-to-wmseg'][0]
        gmseg_im = Image(gmseg)
        scseg = arguments['-gmseg-to-wmseg'][1]
        scseg_im = Image(scseg)
        inverse_gmseg_to_wmseg(gmseg_im, scseg_im, sct.extract_fname(gmseg)[1])


