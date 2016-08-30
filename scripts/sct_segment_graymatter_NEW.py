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
from msct_gmseg_utils_NEW import pre_processing, register_data, apply_transfo, normalize_slice
from sct_utils import printv, tmp_create, extract_fname
from msct_image import Image
import shutil, os

class ParamSeg:
    def __init__(self, fname_im=None, fname_seg=None, fname_level=None):
        self.fname_im = fname_im
        self.fname_seg = fname_seg
        self.fname_level = fname_level


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
        self.target_im, self.info_preprocessing = pre_processing(self.param_seg.fname_im, self.param_seg.fname_seg, self.param_seg.fname_level, new_res=self.param_data.axial_res, square_size_size_mm=self.param_data.square_size_size_mm, denoising=self.param_data.denoising, verbose=self.param.verbose)

        printv('\nRegistering target image to model data ...', self.param.verbose, 'normal')
        # register target image to model dictionary space
        path_warp = self.register_target()

        printv('\nNormalize intensity of target image ...', self.param.verbose, 'normal')
        self.normalize_target()

        printv('\nProjecting target image into the model reduced space ...', self.param.verbose, 'normal')
        self.project_target()

        printv('\nComputing similarities between target slices and model slices using model reduced space ...', self.param.verbose, 'normal')
        self.compute_similarities()

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
            im_src_reg, fname_src2dest, fname_dest2src = register_data(im_src, im_dest, param_reg=self.param_data.register_param, path_copy_warp=path_warping_fields)

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

        pass


