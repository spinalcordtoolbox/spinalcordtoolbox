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
import pandas as pd
import pickle, gzip
from sklearn import manifold, decomposition
from sct_utils import printv, slash_at_the_end
from msct_gmseg_utils_NEW import pre_processing, register_data, apply_transfo, normalize_slice
from msct_image import Image

class ModelParam:
    def __init__(self):
        self.path_data = ''
        self.todo = 'load'# 'compute' or 'load'
        self.new_model_dir = 'gmseg/'
        self.method = 'pca' # 'pca' or 'isomap'
        self.k_pca = 0.8
        self.n_compo_iso = 'half' #'half' or int indicating the actual number of components to keep
        self.n_neighbors_iso = 5

    def __repr__(self):
        info = 'Model Param:\n'
        info += '\t- path to data: '+ self.path_data+'\n'
        info += '\t- created folder: '+self.new_model_dir+'\n'
        info += '\t- used method: '+self.method+'\n'
        if self.method == 'pca':
            info += '\t\t-> % of variability kept for PCA: '+str(self.k_pca)+'\n'
        if self.method == 'isomap':
            info += '\t\t-> # components for isomap: '+str(self.n_compo_iso)+'\n'
            info += '\t\t-> # neighbors for isomap: ' + str(self.n_neighbors_iso) + '\n'

        return info

class DataParam:
    def __init__(self):
        self.denoising = True
        self.axial_res = 0.3
        self.register_param = 'step=1,type=seg,algo=columnwise,metric=MeanSquares,smooth=5,iter=1:step=2,type=im,algo=syn,smooth=2,metric=MI,iter=4:step=3,iter=0'
        self.normalization = True

    def __repr__(self):
        info = 'Data Param:\n'
        info += '\t- denoising: ' + str(self.denoising)+'\n'
        info += '\t- resampling to an axial resolution of: ' + str(self.axial_res)+'mm\n'
        info += '\t- registration parameters: '+self.register_param+'\n'
        info += '\t- intensity normalization: ' + str(self.normalization)+'\n'


        return info

class Param:
    def __init__(self):
        self.verbose = 1

class Model:
    def __init__(self, model_param=None, data_param=None, param=None):
        self.model_param = model_param if model_param is not None else ModelParam()
        self.data_param = data_param if data_param is not None else DataParam()
        self.param = param if param is not None else Param()

        self.slices = [] # list of Slice() : Model dictionary
        self.mean_image = None
        self.intensities = None

        self.fitted_model = None # PCA or Isomap model
        self.fitted_data = None

    def compute_model(self):
        printv('\nComputing the model dictionary ...', self.param.verbose, 'normal')
        os.mkdir(self.model_param.new_model_dir)
        param_fic = open(self.model_param.new_model_dir + 'info.txt', 'w')
        param_fic.write(str(self.model_param))
        param_fic.write(str(self.data_param))
        param_fic.close()

        printv('\n\tLoading data dictionary ...', self.param.verbose, 'normal')
        self.load_model_data()
        self.mean_image = np.mean([dic_slice.im for dic_slice in self.slices], axis=0)

        printv('\n\tCo-register all the data into a common groupwise space (using the white matter segmentations) ...', self.param.verbose, 'normal')
        self.coregister_model_data()

        printv('\n\tNormalize data intensities against averaged median values in the dictionary ...', self.param.verbose, 'normal')
        self.normalize_model_data()

        printv('\nComputing the model reduced space ...', self.param.verbose, 'normal')
        self.compute_reduced_space()

        printv('\nSaving model elements ...', self.param.verbose, 'normal')
        self.save_model()
        ### TODO: add compute_beta / compute tau ??


    def average_wm(self, model_space=True):
        # compute mean WM image
        list_wm = []
        for dic_slice in self.slices:
            if model_space:
                for wm in dic_slice.wm_seg_M:
                    list_wm.append(wm)
            else:
                for wm in dic_slice.wm_seg:
                    list_wm.append(wm)

        data_mean_wm = np.mean(list_wm, axis=0)
        return data_mean_wm

    def load_model_data(self):
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

    def coregister_model_data(self):
        # compute mean WM image
        data_mean_wm = self.average_wm(model_space=False)
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
            im_slice_wm_reg, fname_src2dest, fname_dest2src = register_data(im_src=im_slice_wm, im_dest=im_mean_wm, param_reg=self.data_param.register_param, path_copy_warp=warp_dir)

            # use forward warping field to register all slice wm
            list_wmseg_reg = []
            for wm_seg in dic_slice.wm_seg:
                im_wmseg = Image(param=wm_seg)
                im_wmseg_reg = apply_transfo(im_src=im_wmseg, im_dest=im_mean_wm, warp=warp_dir+'/'+fname_src2dest, interp='nn')
                list_wmseg_reg.append(im_wmseg_reg.data)

            # use forward warping field to register gm seg
            list_gmseg_reg = []
            for gm_seg in dic_slice.gm_seg:
                im_gmseg = Image(param=gm_seg)
                im_gmseg_reg = apply_transfo(im_src=im_gmseg, im_dest=im_mean_wm, warp=warp_dir+'/'+fname_src2dest, interp='nn')
                list_gmseg_reg.append(im_gmseg_reg.data)

            # use forward warping field to register im
            im_slice = Image(dic_slice.im)
            im_slice_reg = apply_transfo(im_src=im_slice, im_dest=im_mean_wm, warp=warp_dir+'/'+fname_src2dest)

            # set slice attributes with data registered into the model space
            dic_slice.set(im_m=im_slice_reg.data)
            dic_slice.set(wm_seg_m=list_wmseg_reg)
            dic_slice.set(gm_seg_m=list_gmseg_reg)

            # remove warping fields directory
            shutil.rmtree(warp_dir)


    def normalize_model_data(self):
        # get the id of the slices by vertebral level
        id_by_level = {}
        for dic_slice in self.slices:
            level_int = int(round(dic_slice.level))
            if level_int not in id_by_level.keys():
                id_by_level[level_int] = [dic_slice.id]
            else:
                id_by_level[level_int].append(dic_slice.id)

        # get the average median values by level:
        list_gm_by_level = []
        list_wm_by_level = []
        list_min_by_level = []
        list_max_by_level = []
        list_indexes = []

        for level, list_id_slices in id_by_level.items():
            list_med_gm = []
            list_med_wm = []
            list_min = []
            list_max = []
            # get median GM and WM values for all slices of the same level:
            for id_slice in list_id_slices:
                slice = self.slices[id_slice]
                for gm in slice.gm_seg_M:
                    med_gm = np.median(slice.im_M[gm==1])
                    list_med_gm.append(med_gm)
                for wm in slice.wm_seg_M:
                    med_wm = np.median(slice.im_M[wm == 1])
                    list_med_wm.append(med_wm)

                list_min.append(min(slice.im_M.flatten()))
                list_max.append(max(slice.im_M.flatten()))

            list_gm_by_level.append(np.mean(list_med_gm))
            list_wm_by_level.append(np.mean(list_med_wm))
            list_min_by_level.append(min(list_min))
            list_max_by_level.append(max(list_max))
            list_indexes.append(level)

        # save average median values in a Panda data frame
        data_intensities = {'GM': pd.Series(list_gm_by_level, index=list_indexes), 'WM': pd.Series(list_wm_by_level, index=list_indexes), 'MIN': pd.Series(list_min_by_level, index=list_indexes), 'MAX': pd.Series(list_max_by_level, index=list_indexes)}
        self.intensities = pd.DataFrame(data_intensities)

        # Normalize slices using dic values
        for dic_slice in self.slices:
            level_int = int(round(dic_slice.level))
            norm_im_M = normalize_slice(dic_slice.im_M, dic_slice.gm_seg_M, dic_slice.wm_seg_M, self.intensities['GM'][level_int], self.intensities['WM'][level_int], min=self.intensities['MIN'][level_int], max=self.intensities['MAX'][level_int])
            dic_slice.set(im_m=norm_im_M)


    def compute_reduced_space(self):
        model = None
        model_data =  np.asarray([dic_slice.im_M.flatten() for dic_slice in self.slices])

        if self.model_param.method == 'pca':
            ## PCA
            model = decomposition.PCA(n_components=self.model_param.k_pca)
            self.fitted_data = model.fit_transform(model_data)

        if self.model_param.method == 'isomap':
            ## ISOMAP
            n_neighbors = self.model_param.n_neighbors_iso
            if self.model_param.n_compo_iso == 'half':
                n_components = model_data.shape[0] / 2
            else:
                n_components = self.model_param.n_compo_iso

            model = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
            self.fitted_data = model.fit_transform(model_data)

        # save model after bing fitted to data
        self.fitted_model = model

    def save_model(self):
        os.chdir(self.model_param.new_model_dir)
        ## to save:
        ##   - self.slices = dictionary
        pickle.dump(self.slices, gzip.open('slices.pklz', 'wb'), protocol=2)

        ##   - self.intensities = for normalization
        pickle.dump(self.intensities, gzip.open('intensities.pklz', 'wb'), protocol=2)

        ##   - reduced space (pca or isomap)
        pickle.dump(self.fitted_model, gzip.open('fitted_model.pklz', 'wb'), protocol=2)

        ##   - fitted data (=eigen vectors or embedding vectors )
        pickle.dump(self.fitted_data, gzip.open('fitted_data.pklz', 'wb'), protocol=2)

        ##   - tau value --> still needed ?

        os.chdir('..')



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






