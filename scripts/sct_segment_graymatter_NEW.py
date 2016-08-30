#!/usr/bin/env python
########################################################################################################################
#
#
# Gray matter segmentation - new implementation
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# Modified: 2016-06-14
#
# About the license: see the file LICENSE.TXT
########################################################################################################################
from msct_multiatlas_seg_NEW import Param, ParamData, ParamModel, Model
from msct_gmseg_utils_NEW import pre_processing, register_data, apply_transfo, normalize_slice, average_gm_wm
from sct_utils import printv, tmp_create, extract_fname
from msct_image import Image
from math import exp
import numpy as np
import shutil, os

class ParamSeg:
    def __init__(self, fname_im=None, fname_seg=None, fname_level=None):
        self.fname_im = fname_im
        self.fname_seg = fname_seg
        self.fname_level = fname_level

        # param to compute similarities:
        self.weight_level = 2.5 # gamma
        self.weight_coord = 0.0065 # tau --> need to be validated for specific dataset
        self.thr_similarity = 0.8 # epsilon but on normalized to 1 similarities (by slice of dic and slice of target)
        # TODO = find the best thr

        self.type_seg = 'prob' # 'prob' or 'bin'


class SegmentGM:
    def __init__(self, param_seg=None, param_model=None, param_data=None, param=None):
        self.param_seg = param_seg if param_seg is not None else ParamSeg()
        self.param_model = param_model if param_model is not None else ParamModel()
        self.param_data = param_data if param_data is not None else ParamData()
        self.param = param if param is not None else Param()

        # create model:
        self.model = Model(param_model=self.param_model, param_data=self.param_data, param=self.param)

        # create tmp directory
        self.tmp_dir = tmp_create(verbose=self.param.verbose) # path to tmp directory

        self.target_im = None # list of slices
        self.info_preprocessing = None # dic containing {'orientation': 'xxx', 'im_target_rpi': im, 'interpolated_images': [list of array = interpolated image data per slice]}

        self.projected_target = None


    def segment(self):
        self.copy_data_to_tmp()
        # go to tmp directory
        os.chdir(self.tmp_dir)
        # load model
        self.model.load_model()

        printv('\nPre-processing target image ...', self.param.verbose, 'normal')
        self.target_im, self.info_preprocessing = pre_processing(self.param_seg.fname_im, self.param_seg.fname_seg, self.param_seg.fname_level, new_res=self.param_data.axial_res, square_size_size_mm=self.param_data.square_size_size_mm, denoising=self.param_data.denoising, verbose=self.param.verbose, rm_tmp=self.param.rm_tmp)

        printv('\nRegistering target image to model data ...', self.param.verbose, 'normal')
        # register target image to model dictionary space
        path_warp = self.register_target()

        printv('\nNormalizing intensity of target image ...', self.param.verbose, 'normal')
        self.normalize_target()

        printv('\nProjecting target image into the model reduced space ...', self.param.verbose, 'normal')
        self.project_target()

        printv('\nComputing similarities between target slices and model slices using model reduced space ...', self.param.verbose, 'normal')
        list_dic_indexes_by_slice = self.compute_similarities()

        printv('\nDoing label fusion of model slices most similar to target slices ...', self.param.verbose, 'normal')
        self.label_fusion(list_dic_indexes_by_slice)

        printv('\nWarping back segmentation into image space...', self.param.verbose, 'normal')
        self.warp_back_seg(path_warp)
        self.post_processing()


        # go back to original directory
        os.chdir('..')


    def copy_data_to_tmp(self):
        # copy input image
        if self.param_seg.fname_im is not None:
            shutil.copy(self.param_seg.fname_im, self.tmp_dir)
            self.param_seg.fname_im = ''.join(extract_fname(self.param_seg.fname_im)[1:])
        else:
            printv('ERROR: No input image', self.param.verbose, 'error')

        # copy sc seg image
        if self.param_seg.fname_seg is not None:
            shutil.copy(self.param_seg.fname_seg, self.tmp_dir)
            self.param_seg.fname_seg = ''.join(extract_fname(self.param_seg.fname_seg)[1:])
        else:
            printv('ERROR: No SC segmentation image', self.param.verbose, 'error')

        # copy level file
        if self.param_seg.fname_level is not None:
            shutil.copy(self.param_seg.fname_level, self.tmp_dir)
            self.param_seg.fname_level = ''.join(extract_fname(self.param_seg.fname_level)[1:])

    def register_target(self):
        # create dir to store warping fields
        path_warping_fields = 'warp_target/'
        if not os.path.exists(path_warping_fields):
            os.mkdir(path_warping_fields)

        # get destination image
        im_dest = Image(self.model.mean_image)

        for target_slice in self.target_im:
            im_src = Image(target_slice.im)
            # register slice image to mean dictionary image
            im_src_reg, fname_src2dest, fname_dest2src = register_data(im_src, im_dest, param_reg=self.param_data.register_param, path_copy_warp=path_warping_fields, rm_tmp=self.param.rm_tmp)

            # rename warping fields
            fname_src2dest_slice = 'warp_target_slice'+str(target_slice.id)+'2dic.nii.gz'
            fname_dest2src_slice = 'warp_dic2target_slice' + str(target_slice.id) + '.nii.gz'
            shutil.move(path_warping_fields+fname_src2dest, path_warping_fields+fname_src2dest_slice)
            shutil.move(path_warping_fields+fname_dest2src, path_warping_fields+fname_dest2src_slice)

            # set moved image
            target_slice.set(im_m=im_src_reg.data)

        return path_warping_fields

    def normalize_target(self):
        # get gm seg from model by level
        gm_seg_model, wm_seg_model = self.model.get_gm_wm_by_level()

        # for each target slice: normalize
        for target_slice in self.target_im:
            level_int = int(round(target_slice.level))
            norm_im_M = normalize_slice(target_slice.im_M, gm_seg_model[level_int], wm_seg_model[level_int], self.model.intensities['GM'][level_int], self.model.intensities['WM'][level_int],min=self.model.intensities['MIN'][level_int], max=self.model.intensities['MAX'][level_int])
            target_slice.set(im_m=norm_im_M)

    def project_target(self):
        projected_target_slices = []
        for target_slice in self.target_im:
            # get slice data in the good shape
            slice_data = target_slice.im_M.flatten()
            slice_data = slice_data.reshape(1, -1) # data with single sample
            # project slice data into the model
            slice_data_projected = self.model.fitted_model.transform(slice_data)
            projected_target_slices.append(slice_data_projected)
        # store projected target slices
        self.projected_target = projected_target_slices

    def compute_similarities(self):
        list_dic_indexes_by_slice = []
        for i, target_coord in enumerate(self.projected_target):
            list_dic_similarities = []
            for j, dic_coord in enumerate(self.model.fitted_data):
                # compute square norm using coordinates in the model space
                square_norm = np.linalg.norm((target_coord - dic_coord), 2)
                # compute similarity with or without levels
                if self.param_seg.fname_level is not None:
                    # EQUATION WITH LEVELS
                    similarity = exp(-self.param_seg.weight_level * abs(self.target_im[i].level - self.model.slices[j].level)) * exp(-self.param_seg.weight_coord * square_norm)
                else:
                    # EQUATION WITHOUT LEVELS
                    similarity = exp(-self.param_seg.weight_coord * square_norm)
                # add similarity to list
                list_dic_similarities.append(similarity)
            list_norm_similarities =  [float(s)/sum(list_dic_similarities) for s in list_dic_similarities]
            # select indexes of most similar slices
            list_dic_indexes = []
            for j, norm_sim in enumerate(list_norm_similarities):
                if norm_sim >= self.param_seg.thr_similarity:
                    list_dic_indexes.append(j)
            # save list of indexes into list by slice
            list_dic_indexes_by_slice.append(list_dic_indexes)

        return list_dic_indexes_by_slice

    def label_fusion(self, list_dic_indexes_by_slice):
        for target_slice in self.target_im:
            # get list of slices corresponding to the indexes
            list_dic_slices = [self.model.slices[j] for j in list_dic_indexes_by_slice[target_slice.id]]
            # average slices GM and WM
            data_mean_gm, data_mean_wm = average_gm_wm(list_dic_slices)
            if self.param_seg.type_seg == 'bin':
                # binarize GM seg
                data_mean_gm[data_mean_gm >= 0.5] = 1
                data_mean_gm[data_mean_gm < 0.5] = 0
                # binarize WM seg
                data_mean_wm[data_mean_wm >= 0.5] = 1
                data_mean_wm[data_mean_wm < 0.5] = 0
            # store segmentation into target_im
            target_slice.set(gm_seg_m=data_mean_gm, wm_seg_m=data_mean_wm)

    def warp_back_seg(self, path_warp):
        for target_slice in self.target_im:
            fname_dic_space2slice_space = path_warp+'/warp_dic2target_slice' + str(target_slice.id) + '.nii.gz'
            im_dest = Image(target_slice.im)
            interpolation = 'nn' if self.param_seg.type_seg == 'bin' else 'linear'
            # warp GM
            im_src_gm = Image(target_slice.gm_seg_M)
            im_src_gm_reg = apply_transfo(im_src_gm, im_dest, fname_dic_space2slice_space, interp=interpolation, rm_tmp=self.param.rm_tmp)
            # warp WM
            im_src_wm = Image(target_slice.wm_seg_M)
            im_src_wm_reg = apply_transfo(im_src_wm, im_dest, fname_dic_space2slice_space, interp=interpolation, rm_tmp=self.param.rm_tmp)
            # set slice attributes
            target_slice.set(gm_seg=im_src_gm_reg.data, wm_seg=im_src_wm_reg.data)

    def post_processing(self):
        pass
