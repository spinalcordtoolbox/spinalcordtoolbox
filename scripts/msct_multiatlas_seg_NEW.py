#!/usr/bin/env python
########################################################################################################################
#
# Asman et al. groupwise multi-atlas segmentation method implementation, with a lot of changes
#
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Sara Dupont
# Created: 2016-06-15
#
# About the license: see the file LICENSE.TXT
########################################################################################################################
import os
import shutil
import numpy as np
from sct_utils import printv, slash_at_the_end
from msct_gmseg_utils_NEW import pre_processing, register_data, apply_transfo
from msct_image import Image

class ModelParam:
    def __init__(self):
        self.path_data = ''
        self.todo = 'load'# 'compute' or 'load'
        self.new_model_dir = 'gmseg/'

    def __repr__(self):
        info = 'Model Param:\n'
        info += '\t- path to data: '+ self.path_data+'\n'
        info += '\t- created folder: '+self.new_model_dir+'\n'

        return info

class DataParam:
    def __init__(self):
        self.denoising = True
        self.axial_res = 0.3
        self.register_param = 'step=1,type=seg,algo=columnwise,metric=MeanSquares,smooth=5,iter=1:step=2,type=im,algo=syn,smooth=2,metric=MI,iter=4:step=3,iter=0'

    def __repr__(self):
        info = 'Data Param:\n'
        info += '\t- denoising: ' + str(self.denoising)+'\n'
        info += '\t- resampling to an axial resolution of: ' + str(self.axial_res)+'mm\n'
        info += '\t- registration parameters: '+self.register_param+'\n'

        return info

class Param:
    def __init__(self):
        self.verbose = 1

class ModelDictionary:
    def __init__(self, model_param=None, data_param=None, param=None):
        self.model_param = model_param if model_param is not None else ModelParam()
        self.data_param = data_param if data_param is not None else DataParam()
        self.param = param if param is not None else Param()

        self.slices = []
        self.mean_image = None

    def compute_model(self):
        printv('\nComputing the model dictionary ...', self.param.verbose, 'normal')
        os.mkdir(self.model_param.new_model_dir)
        param_fic = open(self.model_param.new_model_dir + 'info.txt', 'w')
        param_fic.write(str(self.model_param))
        param_fic.write(str(self.data_param))
        param_fic.close()

        printv('\n\tLoading data dictionary ...', self.param.verbose, 'normal')
        self.load_data()
        self.mean_image = np.mean([dic_slice.im for dic_slice in self.slices], axis=0)


        printv('\n\tCo-register all the data into a common groupwise space (using the white matter segmentations) ...', self.param.verbose, 'normal')
        self.coregister_data()

        ## TODO: COMPLETE COMPUTE _DIC
        # - NORMALIZE INTENSITY

    def load_data(self):
        '''
        Data should be organized with one folder per subject containing:
            - A WM/GM contrasted image containing 'im' in its name
            - a segmentation of the SC containing 'seg' in its name
            - a/several manual segmentation(s) of GM containing 'gm' in its/their name(s)
            - a file containing vertebral level information as a nifti image or as a text file containing 'level' in its name
        '''
        path_data = slash_at_the_end(self.model_param.path_data, slash=1)

        # total number of slices: J
        j = 0

        for sub in os.listdir(path_data):
            # load images of each subject
            if os.path.isdir(path_data+sub):
                fname_data = ''
                fname_sc_seg = ''
                list_fname_gmseg = []
                fname_level = None
                for file_name in os.listdir(path_data+sub):
                    if 'gm' in file_name:
                        list_fname_gmseg.append(path_data+sub+'/'+file_name)
                    elif 'seg' in file_name:
                        fname_sc_seg = path_data+sub+'/'+file_name
                    elif 'im' in file_name:
                        fname_data = path_data+sub+'/'+file_name
                    if 'level' in file_name:
                        fname_level = path_data+sub+'/'+file_name

                # preprocess data
                list_slices_sub, info = pre_processing(fname_data, fname_sc_seg, fname_level=fname_level, fname_manual_gmseg=list_fname_gmseg, new_res=self.data_param.axial_res, denoising=self.data_param.denoising)
                for slice_sub in list_slices_sub:
                    slice_sub.set(slice_id=slice_sub.id+j)
                    self.slices.append(slice_sub)

                j += len(list_slices_sub)

    def coregister_data(self):
        # compute mean WM image
        list_wm = []
        for dic_slice in self.slices:
            for wm in dic_slice.wm_seg:
                list_wm.append(wm)

        data_mean_wm = np.mean(list_wm, axis=0)
        im_mean_wm = Image(param=data_mean_wm)

        # register all slices WM on mean WM
        for dic_slice in self.slices:
            # create a directory to get the warping fields
            warp_dir = 'wf_slice'+str(dic_slice.id)
            os.mkdir(warp_dir)

            # get slice mean WM image
            data_slice_wm = np.mean(dic_slice.wm_seg, axis=0)
            im_slice_wm = Image(data_slice_wm)
            # register slice WM on mean WM
            im_slice_wm_reg = register_data(im_src=im_slice_wm, im_dest=im_mean_wm, param_reg=self.data_param.register_param, path_warp=warp_dir)

            # use forward warping field to register all slice wm
            list_wmseg_reg = []
            for wm_seg in dic_slice.wm_seg:
                im_wmseg = Image(param=wm_seg)
                im_wmseg_reg = apply_transfo(im_src=im_wmseg, im_dest=im_mean_wm, warp=warp_dir + '/warp_src2dest.nii.gz', interp='nn')
                list_wmseg_reg.append(im_wmseg_reg.data)

            # use forward warping field to register gm seg
            list_gmseg_reg = []
            for gm_seg in dic_slice.gm_seg:
                im_gmseg = Image(param=gm_seg)
                im_gmseg_reg = apply_transfo(im_src=im_gmseg, im_dest=im_mean_wm, warp=warp_dir+'/warp_src2dest.nii.gz', interp='nn')
                list_gmseg_reg.append(im_gmseg_reg.data)

            # use forward warping field to register im
            im_slice = Image(dic_slice.im)
            im_slice_reg = apply_transfo(im_src=im_slice, im_dest=im_mean_wm, warp=warp_dir + '/warp_src2dest.nii.gz')

            # set slice attributes with data registered into the model space
            dic_slice.set(im_m=im_slice_reg.data)
            dic_slice.set(wm_seg_m=list_wmseg_reg)
            dic_slice.set(gm_seg_m=list_gmseg_reg)

            # remove warping fields directory
            shutil.rmtree(warp_dir)


    '''
    ## REGISTRATION WITH IMAGES FIRST, THEN APPLY TO GM/WM SEG
    def coregister_data(self):
        im_mean_image = Image(param=self.mean_image)
        for dic_slice in self.slices:
            # create a directory to get the warping fields
            warp_dir = 'warping_fields_slice'+str(dic_slice.id)
            os.mkdir(warp_dir)
            # get dic image
            im_dic_slice = Image(param=dic_slice.im)
            # register image
            im_reg = register_data(im_src=im_dic_slice, im_dest=im_mean_image, param_reg=self.data_param.register_param, path_warp=warp_dir)
            # use warping fields to register gm seg
            list_gmseg_reg = []
            for gm_seg in dic_slice.gm_seg:
                im_gmseg = Image(param=gm_seg)
                im_gmseg_reg = apply_transfo(im_src=im_gmseg, im_dest=im_mean_image, warp=warp_dir+'/warp_src2dest.nii.gz')
                list_gmseg_reg.append(im_gmseg_reg.data)
            # set slie attributes with data registered into the model space
            dic_slice.set(im_m=im_reg.data)
            dic_slice.set(gm_seg_m=list_gmseg_reg)
            # remove warping fields directory
            shutil.rmtree(warp_dir)
    '''






