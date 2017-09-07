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
import gzip
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn import decomposition, manifold

from msct_gmseg_utils import (apply_transfo, average_gm_wm, normalize_slice,
                              pre_processing, register_data)
from msct_image import Image
from msct_parser import Parser
from sct_utils import printv, slash_at_the_end
import sct_utils as sct

def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute the model for GM segmentation.\n'
                                 'Dataset should be organized with one folder per subject containing: \n'
                                 '\t- A WM/GM contrasted image containing "im" in its name\n'
                                 '\t- a segmentation of the SC containing "seg" in its name\n'
                                 '\t- a/several manual segmentation(s) of GM containing "gm" in its/their name(s)\n'
                                 '\t- a file containing vertebral level information as a nifti image or as a text file containing "level" in its name\n')
    parser.add_option(name="-path-data",
                      type_value="folder",
                      description="Path to the dataset",
                      mandatory=True,
                      example='my_data/')
    parser.add_option(name="-o",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      default_value=ParamModel().new_model_dir,
                      example='my_model/')
    parser.usage.addSection('MODEL PARAMETERS')
    parser.add_option(name="-model-type",
                      type_value="multiple_choice",
                      description="Type of reduced space (PCA or IsoMap)",
                      mandatory=False,
                      default_value=ParamModel().method,
                      example=['pca', 'isomap'])
    parser.add_option(name="-k-pca",
                      type_value="float",
                      description="-ONLY WITH PCA- Percentage of variability to keep in the PCA reduced space (between 0 and 1)",
                      mandatory=False,
                      default_value=ParamModel().k_pca)
    parser.add_option(name="-n-compo-iso",
                      type_value="float",
                      description='-ONLY WITH ISOMAP- Percentage of components to keep (The total number of components is the number of slices in the model). To keep half of the components, use 0.5. ',
                      mandatory=False,
                      default_value=ParamModel().n_compo_iso)
    parser.add_option(name="-n-neighbors-iso",
                      type_value="int",
                      description='-ONLY WITH ISOMAP- Number of neighbors to consider in the reduced space.',
                      mandatory=False,
                      default_value=ParamModel().n_neighbors_iso)
    parser.usage.addSection('DATA PROCESSING PARAMETERS')
    parser.add_option(name="-denoising",
                      type_value="multiple_choice",
                      description='Apply non-local means denoising (as implemented in dipy) on data',
                      mandatory=False,
                      example=['0', '1'],
                      default_value=int(ParamData().denoising))
    parser.add_option(name="-normalization",
                      type_value='multiple_choice',
                      description="Normalize data intensity using median intensity values of the manual WM and the GM",
                      mandatory=False,
                      default_value=int(ParamData().normalization),
                      example=['0', '1'])
    parser.add_option(name="-axial-res",
                      type_value='float',
                      description="Axial resolution to resample data to",
                      mandatory=False,
                      default_value=ParamData().axial_res)
    parser.add_option(name="-sq-size",
                      type_value='float',
                      description="Size of the square centered on SC to crop data (in mm)",
                      mandatory=False,
                      default_value=ParamData().square_size_size_mm)
    parser.add_option(name="-reg-param",
                      type_value='str',
                      description="Registration parameters to co-register data together",
                      mandatory=False,
                      default_value=ParamData().register_param)
    parser.usage.addSection('Leave One Out Cross Validation')
    parser.add_option(name="-ind-rm",
                      type_value="int",
                      description='Index of the subject to remove to compute a model for Leave one out cross validation',
                      mandatory=False,
                      default_value=str(ParamModel().ind_rm))
    parser.usage.addSection('MISC')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value=str(int(Param().rm_tmp)),
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value=str(Param().verbose))

    return parser


class ParamModel:
    def __init__(self):
        self.path_data = ''
        self.todo = 'load'  # 'compute' or 'load'
        self.new_model_dir = 'gm_model/'
        self.method = 'pca'  # 'pca' or 'isomap'
        self.k_pca = 0.95  # chosen after loocv optimization with the 37subjects-model
        self.n_compo_iso = 0.5  # float between 0 and 1 : percentage of component to keep. 0.5 = keep half of the components
        self.n_neighbors_iso = 5
        #
        self.ind_rm = None
        #
        path_script = os.path.dirname(__file__)
        path_sct = os.path.dirname(path_script)
        self.path_model_to_load = path_sct + '/data/gm_model'

    def __repr__(self):
        info = 'Model Param:\n'
        info += '\t- path to data: ' + self.path_data + '\n'
        info += '\t- created folder: ' + self.new_model_dir + '\n'
        info += '\t- used method: ' + self.method + '\n'
        if self.method == 'pca':
            info += '\t\t-> % of variability kept for PCA: ' + str(self.k_pca) + '\n'
        if self.method == 'isomap':
            info += '\t\t-> # components for isomap: ' + str(self.n_compo_iso) + ' (in percentage: 0.5 = keep half of the components)\n'
            info += '\t\t-> # neighbors for isomap: ' + str(self.n_neighbors_iso) + '\n'

        return info


class ParamData:
    def __init__(self):
        self.denoising = True
        self.axial_res = 0.3
        self.square_size_size_mm = 22.5
        self.register_param = 'step=1,type=seg,algo=centermassrot,metric=MeanSquares,smooth=2,poly=0,iter=1:step=2,type=seg,algo=columnwise,metric=MeanSquares,smooth=1,iter=1'
        self.normalization = True

    def __repr__(self):
        info = 'Data Param:\n'
        info += '\t- denoising: ' + str(self.denoising) + '\n'
        info += '\t- resampling to an axial resolution of: ' + str(self.axial_res) + 'mm\n'
        info += '\t- size of the square mask: ' + str(self.square_size_size_mm) + 'mm\n'
        info += '\t- registration parameters: ' + self.register_param + '\n'
        info += '\t- intensity normalization: ' + str(self.normalization) + '\n'

        return info


class Param:
    def __init__(self):
        self.verbose = 1
        self.rm_tmp = True


class Model:
    def __init__(self, param_model=None, param_data=None, param=None):
        self.param_model = param_model if param_model is not None else ParamModel()
        self.param_data = param_data if param_data is not None else ParamData()
        self.param = param if param is not None else Param()

        self.slices = []  # list of Slice() : Model dictionary
        self.mean_image = None
        self.intensities = None

        self.fitted_model = None  # PCA or Isomap model
        self.fitted_data = None

    # ------------------------------------------------------------------------------------------------------------------
    #                                       FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------
    def compute_model(self):
        printv('\nComputing the model dictionary ...', self.param.verbose, 'normal')
        # create model folder
        if os.path.exists(self.param_model.new_model_dir) and os.listdir(self.param_model.new_model_dir) != []:
            shutil.move(self.param_model.new_model_dir, slash_at_the_end(self.param_model.new_model_dir, slash=0) + '_old')
        if not os.path.exists(self.param_model.new_model_dir):
            os.mkdir(self.param_model.new_model_dir)
        # write model info
        param_fic = open(self.param_model.new_model_dir + 'info.txt', 'w')
        param_fic.write('Model computed on ' + '-'.join(str(t) for t in time.localtime()[:3]) + '\n')
        param_fic.write(str(self.param_model))
        param_fic.write(str(self.param_data))
        param_fic.close()

        printv('\n\tLoading data dictionary ...', self.param.verbose, 'normal')
        self.load_model_data()
        self.mean_image = np.mean([dic_slice.im for dic_slice in self.slices], axis=0)

        printv('\n\tCo-register all the data into a common groupwise space ...', self.param.verbose, 'normal')
        self.coregister_model_data()

        printv('\n\tNormalize data intensities against averaged median values in the dictionary ...', self.param.verbose, 'normal')
        self.normalize_model_data()

        printv('\nComputing the model reduced space ...', self.param.verbose, 'normal')
        self.compute_reduced_space()

        printv('\nSaving model elements ...', self.param.verbose, 'normal')
        self.save_model()

    # ------------------------------------------------------------------------------------------------------------------
    def load_model_data(self):
        '''
        Data should be organized with one folder per subject containing:
            - A WM/GM contrasted image containing 'im' in its name
            - a segmentation of the SC containing 'seg' in its name
            - a/several manual segmentation(s) of GM containing 'gm' in its/their name(s)
            - a file containing vertebral level information as a nifti image or as a text file containing 'level' in its name
        '''
        path_data = slash_at_the_end(self.param_model.path_data, slash=1)

        list_sub = [sub for sub in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, sub))]
        if self.param_model.ind_rm is not None and self.param_model.ind_rm < len(list_sub):
            list_sub.pop(self.param_model.ind_rm)

        # total number of slices: J
        j = 0

        for sub in list_sub:
            # load images of each subject
            fname_data = None
            fname_sc_seg = None
            list_fname_gmseg = []
            fname_level = None
            for file_name in os.listdir(path_data + sub):
                fname = path_data + sub + '/' + file_name
                if os.path.isfile(fname):
                    if 'gm' in file_name:
                        list_fname_gmseg.append(fname)
                    elif 'seg' in file_name:
                        fname_sc_seg = fname
                    elif 'im' in file_name:
                        fname_data = fname
                    if 'level' in file_name:
                        fname_level = fname

            info_data = 'Loaded files: \n'
            info_data += 'Image: ....... ' + str(fname_data) + '\n'
            info_data += 'SC seg: ...... ' + str(fname_sc_seg) + '\n'
            info_data += 'GM seg: ...... ' + str(list_fname_gmseg) + '\n'
            info_data += 'Levels: ...... ' + str(fname_level) + '\n'

            if fname_data == None or fname_sc_seg == None or list_fname_gmseg == []:
                printv(info_data, self.param.verbose, 'error')
            else:
                printv(info_data, self.param.verbose, 'normal')

            # preprocess data
            list_slices_sub, info = pre_processing(fname_data, fname_sc_seg, fname_level=fname_level, fname_manual_gmseg=list_fname_gmseg, new_res=self.param_data.axial_res, square_size_size_mm=self.param_data.square_size_size_mm,  denoising=self.param_data.denoising, for_model=True)
            for i_slice, slice_sub in enumerate(list_slices_sub):
                slice_sub.set(slice_id=i_slice + j)
                self.slices.append(slice_sub)

            j += len(list_slices_sub)

    # ------------------------------------------------------------------------------------------------------------------
    def coregister_model_data(self):
        # get mean image
        im_mean = Image(param=self.mean_image)

        # register all slices WM on mean WM
        for dic_slice in self.slices:
            # create a directory to get the warping fields
            warp_dir = 'wf_slice' + str(dic_slice.id)
            if not os.path.exists(warp_dir):
                os.mkdir(warp_dir)

            # get slice mean WM image
            im_slice = Image(param=dic_slice.im)
            # register slice image on mean dic image
            im_slice_reg, fname_src2dest, fname_dest2src = register_data(im_src=im_slice, im_dest=im_mean, param_reg=self.param_data.register_param, path_copy_warp=warp_dir)
            shape = im_slice_reg.data.shape

            # use forward warping field to register all slice wm
            list_wmseg_reg = []
            for wm_seg in dic_slice.wm_seg:
                im_wmseg = Image(param=wm_seg)
                im_wmseg_reg = apply_transfo(im_src=im_wmseg, im_dest=im_mean, warp=warp_dir + '/' + fname_src2dest, interp='nn')

                list_wmseg_reg.append(im_wmseg_reg.data.reshape(shape))

            # use forward warping field to register gm seg
            list_gmseg_reg = []
            for gm_seg in dic_slice.gm_seg:
                im_gmseg = Image(param=gm_seg)
                im_gmseg_reg = apply_transfo(im_src=im_gmseg, im_dest=im_mean, warp=warp_dir + '/' + fname_src2dest, interp='nn')
                list_gmseg_reg.append(im_gmseg_reg.data.reshape(shape))

            # set slice attributes with data registered into the model space
            dic_slice.set(im_m=im_slice_reg.data)
            dic_slice.set(wm_seg_m=list_wmseg_reg)
            dic_slice.set(gm_seg_m=list_gmseg_reg)

            # remove warping fields directory
            if self.param.rm_tmp:
                shutil.rmtree(warp_dir)

    # ------------------------------------------------------------------------------------------------------------------
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
            if level != 0:
                list_med_gm = []
                list_med_wm = []
                list_min = []
                list_max = []
                # get median GM and WM values for all slices of the same level:
                for id_slice in list_id_slices:
                    slice = self.slices[id_slice]
                    for gm in slice.gm_seg_M:
                        med_gm = np.median(slice.im_M[gm == 1])
                        list_med_gm.append(med_gm)
                    for wm in slice.wm_seg_M:
                        med_wm = np.median(slice.im_M[wm == 1])
                        list_med_wm.append(med_wm)

                    list_min.append(min(slice.im_M.flatten()[slice.im_M.flatten().nonzero()]))
                    list_max.append(max(slice.im_M.flatten()))

                list_gm_by_level.append(np.mean(list_med_gm))
                list_wm_by_level.append(np.mean(list_med_wm))
                list_min_by_level.append(min(list_min))
                list_max_by_level.append(max(list_max))
                list_indexes.append(level)

        # add level 0 for images with no level (or level not in model)
        # average GM and WM for all slices, get min and max of all slices
        list_gm_by_level.append(np.mean(list_gm_by_level))
        list_wm_by_level.append(np.mean(list_wm_by_level))
        list_min_by_level.append(min(list_min_by_level))
        list_max_by_level.append(max(list_max_by_level))
        list_indexes.append(0)

        # save average median values in a Panda data frame
        data_intensities = {'GM': pd.Series(list_gm_by_level, index=list_indexes), 'WM': pd.Series(list_wm_by_level, index=list_indexes), 'MIN': pd.Series(list_min_by_level, index=list_indexes), 'MAX': pd.Series(list_max_by_level, index=list_indexes)}
        self.intensities = pd.DataFrame(data_intensities)

        # Normalize slices using dic values
        for dic_slice in self.slices:
            level_int = int(round(dic_slice.level))
            av_gm_slice, av_wm_slice = average_gm_wm([dic_slice], bin=True)
            norm_im_M = normalize_slice(dic_slice.im_M, av_gm_slice, av_wm_slice, self.intensities['GM'][level_int], self.intensities['WM'][level_int], val_min=self.intensities['MIN'][level_int], val_max=self.intensities['MAX'][level_int])
            dic_slice.set(im_m=norm_im_M)

    # ------------------------------------------------------------------------------------------------------------------
    def compute_reduced_space(self):
        model = None
        model_data =  np.asarray([dic_slice.im_M.flatten() for dic_slice in self.slices])

        if self.param_model.method == 'pca':
            # PCA
            model = decomposition.PCA(n_components=self.param_model.k_pca)
            self.fitted_data = model.fit_transform(model_data)

        if self.param_model.method == 'isomap':
            # ISOMAP
            n_neighbors = self.param_model.n_neighbors_iso
            n_components = int(model_data.shape[0] * self.param_model.n_compo_iso)

            model = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
            self.fitted_data = model.fit_transform(model_data)

        # save model after bing fitted to data
        self.fitted_model = model

    # ------------------------------------------------------------------------------------------------------------------
    def save_model(self):
        os.chdir(self.param_model.new_model_dir)
        # to save:
        # - self.slices = dictionary
        slices = self.slices
        pickle.dump(slices, gzip.open('slices.pklz', 'wb'), protocol=2)

        # - self.intensities = for normalization
        intensities = self.intensities
        pickle.dump(intensities, gzip.open('intensities.pklz', 'wb'), protocol=2)

        # - reduced space (pca or isomap)
        model = self.fitted_model
        pickle.dump(model, gzip.open('fitted_model.pklz', 'wb'), protocol=2)

        # - fitted data (=eigen vectors or embedding vectors )
        data = self.fitted_data
        pickle.dump(data, gzip.open('fitted_data.pklz', 'wb'), protocol=2)

        os.chdir('..')

    # ----------------------------------- END OF FUNCTIONS USED TO COMPUTE THE MODEL -----------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    #                                       FUNCTIONS USED TO LOAD THE MODEL
    # ------------------------------------------------------------------------------------------------------------------
    def load_model(self):
        path = os.path.abspath('.')
        printv('\nLoading model...', self.param.verbose, 'normal')
        os.chdir(self.param_model.path_model_to_load)

        model_files = {'slices': 'slices.pklz', 'intensity': 'intensities.pklz', 'model': 'fitted_model.pklz', 'data': 'fitted_data.pklz'}
        correct_model = True
        for fname in model_files.values():
            if os.path.isfile(fname):
                printv('  OK: ' + fname, self.param.verbose, 'normal')
            else:
                printv('  MISSING FILE: ' + fname, self.param.verbose, 'warning')
                correct_model = False
        if not correct_model:
            path_script = os.path.dirname(__file__)
            path_sct = os.path.dirname(path_script)
            printv('ERROR: The GM segmentation model is not compatible with this version of the code.\n'
                   'To update the model, run the following lines:\n\n'
                   'cd ' + path_sct + '\n'
                   './install_sct -m -b\n', self.param.verbose, 'error')

        # - self.slices = dictionary
        self.slices = pickle.load(gzip.open(model_files['slices'],  'rb'))
        printv('  ' + str(len(self.slices)) + ' slices in the model dataset', self.param.verbose, 'normal')
        self.mean_image = np.mean([dic_slice.im for dic_slice in self.slices], axis=0)

        # - self.intensities = for normalization
        self.intensities = pickle.load(gzip.open(model_files['intensity'], 'rb'))

        # - reduced space (pca or isomap)
        self.fitted_model = pickle.load(gzip.open(model_files['model'], 'rb'))

        # - fitted data (=eigen vectors or embedding vectors )
        self.fitted_data = pickle.load(gzip.open(model_files['data'], 'rb'))

        printv('  model: ' + self.param_model.method)
        printv('  ' + str(self.fitted_data.shape[1]) + ' components kept on ' + str(self.fitted_data.shape[0]), self.param.verbose, 'normal')
        # when model == pca, self.fitted_data.shape[1] = self.fitted_model.n_components_
        os.chdir(path)

    # ------------------------------------------------------------------------------------------------------------------
    #                                                   UTILS FUNCTIONS
    # ------------------------------------------------------------------------------------------------------------------
    def get_gm_wm_by_level(self):
        gm_seg_model = {}  # dic of mean gm seg by vertebral level
        wm_seg_model = {}  # dic of mean wm seg by vertebral level
        # get id of the slices by level
        slices_by_level = {}
        for dic_slice in self.slices:
            level_int = int(round(dic_slice.level))
            if level_int not in slices_by_level.keys():
                slices_by_level[level_int] = [dic_slice]
            else:
                slices_by_level[level_int].append(dic_slice)
        # get average gm and wm by level
        for level, list_slices in slices_by_level.items():
            data_mean_gm, data_mean_wm = average_gm_wm(list_slices)
            gm_seg_model[level] = data_mean_gm
            wm_seg_model[level] = data_mean_wm
        # for level=0 (no leve or level not in model) output average GM and WM seg across all model data
        gm_seg_model[0] = np.mean(gm_seg_model.values(), axis=0)
        wm_seg_model[0] = np.mean(wm_seg_model.values(), axis=0)

        return gm_seg_model, wm_seg_model


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    # create param objects
    param_data = ParamData()
    param_model = ParamModel()
    param = Param()

    # get parser
    parser = get_parser()
    arguments = parser.parse(args)

    param_model.path_data = arguments['-path-data']

    if '-o' in arguments:
        param_model.new_model_dir = arguments['-o']
    if '-model-type' in arguments:
        param_model.method = arguments['-model-type']
    if '-k-pca' in arguments:
        param_model.k_pca = arguments['-k-pca']
    if '-n-compo-iso' in arguments:
        param_model.n_compo_iso = arguments['-n-compo-iso']
    if '-n-neighbors-iso' in arguments:
        param_model.n_neighbors_iso = arguments['-n-neighbors-iso']
    if '-denoising' in arguments:
        param_data.denoising = bool(int(arguments['-denoising']))
    if '-normalization' in arguments:
        param_data.normalization = bool(int(arguments['-normalization']))
    if '-axial-res' in arguments:
        param_data.axial_res = arguments['-axial-res']
    if '-sq-size' in arguments:
        param_data.square_size_size_mm = arguments['-sq-size']
    if '-reg-param' in arguments:
        param_data.register_param = arguments['-reg-param']
    if '-ind-rm' in arguments:
        param_model.ind_rm = arguments['-ind-rm']
    if '-r' in arguments:
        param.rm_tmp = bool(int(arguments['-r']))
    if '-v' in arguments:
        param.verbose = arguments['-v']

    model = Model(param_model=param_model, param_data=param_data, param=param)

    start = time.time()
    model.compute_model()
    end = time.time()
    t = end - start
    printv('Model computed in ' + str(int(round(t / 60))) + ' min, ' + str(t%60) + ' sec', param.verbose, 'info')


if __name__ == "__main__":
    sct.start_stream_logger()
    main()
