#!/usr/bin/env python
########################################################################################################################
#
# Asman et al. groupwise multi-atlas segmentation method implementation, with a lot of changes
#
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Sara Dupont
# Modified: 2015-05-19
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

# TODO change 'target' by 'input'
# TODO : make it faster

# import os
# import sys
# import numpy as np

from msct_pca import PCA
# from msct_image import Image
# from msct_parser import *
from msct_gmseg_utils import *
import sct_utils as sct
import pickle, gzip
import commands
from math import exp


class ModelParam:
    def __init__(self):
        # status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
        path_script = os.path.dirname(__file__)
        path_sct = os.path.dirname(path_script)
        self.path_model = path_sct+'/data/gm_model'  # model_param
        self.todo_model = 'load'  # 'compute'   # model_param
        self.new_model_dir = './gm_model'  # model_param
        self.reg = ['Affine']  # model_param
        self.reg_metric = 'MI'  # model_param
        self.use_levels = 'int'  # model_param
        self.weight_gamma = 2.5  # model_param
        self.model_slices_normalization = False
        self.mean_metric = None
        self.weight_label_fusion = False  # model_param
        self.mode_weight_similarity = False  # model_param
        self.k = 0.8 # percentage of variability explained in the kept eigenvectors (PCA modes)
        self.equation_id = 1  # model_param
        self.verbose = 1  # both

    def __repr__(self):
        s = ''
        s += 'path_model: ' + str(self.path_model) + '\n'
        s += 'todo_model: ' + str(self.todo_model) + '\n'
        s += 'new_model_dir: ' + str(self.new_model_dir) + '  *** only used if todo_model=compute ***\n'
        s += 'reg: ' + str(self.reg) + '\n'
        s += 'reg_metric: ' + str(self.reg_metric) + '\n'
        s += 'use_levels: ' + str(self.use_levels) + '\n'
        s += 'weight_gamma: ' + str(self.weight_gamma) + '\n'
        s += 'model_slices_normalization: ' + str(self.model_slices_normalization) + '\n'
        s += 'mean_metric: ' + str(self.mean_metric) + '\n'
        s += 'weight_label_fusion: ' + str(self.weight_label_fusion) + '\n'
        s += 'mode_weight_similarity: ' + str(self.mode_weight_similarity) + '\n'
        s += 'k: ' + str(self.k) + '\n'
        s += 'equation_id: ' + str(self.equation_id) + '\n'
        return s


class SegmentationParam:
    def __init__(self):
        self.debug = 0
        self.output_path = ''  # SEG
        self.target_denoising = True  # seg_param
        self.target_normalization = True  # seg_param
        self.target_means = None  #   seg_param
        self.z_regularisation = False  # seg_param
        self.res_type = 'prob'  # seg_param
        self.dev = False  # seg_param
        self.qc = 0  # SEG
        self.verbose = 1  # both
        self.remove_tmp = 1  # seg_param

    def __repr__(self):
        s = ''
        s += 'output_path: ' + str(self.output_path) + '\n'
        s += 'target_denoising: ' + str(self.target_denoising) + ' ***WARNING: used in sct_segment_gray_matter not in msct_multiatlas_seg***\n'
        s += 'target_normalization: ' + str(self.target_normalization) + '\n'
        s += 'target_means: ' + str(self.target_means) + '\n'
        s += 'z_regularisation: ' + str(self.z_regularisation) + '\n'
        s += 'res_type: ' + str(self.res_type) + '\n'
        s += 'qc: ' + str(self.qc) + '\n'
        s += 'verbose: ' + str(self.verbose) + '\n'
        s += 'remove_tmp: ' + str(self.remove_tmp) + '\n'

        return s


########################################################################################################################
# ----------------------------------------------------- Classes ------------------------------------------------------ #
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# MODEL DICTIONARY SLICE BY SLICE---------------------------------------------------------------------------------------
class ModelDictionary:
    """
    Dictionary used by the supervised gray matter segmentation method
    """
    def __init__(self, dic_param=None):
        """
        model dictionary constructor

        :param dic_param: dictionary parameters, type: Param
        """
        if dic_param is None:
            self.param = ModelParam()
        else:
            self.param = dic_param

        self.level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7', 15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5'}

        # Initialisation of the parameters
        self.coregistration_transfos = None
        self.slices = None
        self.J = None
        self.N = None
        self.mean_wmseg = None
        self.mean_gmseg = None
        self.mean_image = None

        # list of transformation to apply to each slice to co-register the data into the common groupwise space
        self.coregistration_transfos = self.param.reg

        if self.param.todo_model == 'compute':
            self.compute_dic()
        elif self.param.todo_model == 'load':
            self.load_dic()
        # self.extract_metric_from_dic(gm_percentile=0, wm_percentile=0, save=True)
        # self.mean_seg_by_level(save=True)

    # ------------------------------------------------------------------------------------------------------------------
    # FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def compute_dic(self):

        sct.printv('\nComputing the model dictionary ...', self.param.verbose, 'normal')
        sct.run('mkdir ' + self.param.new_model_dir)
        param_fic = open(self.param.new_model_dir + '/info.txt', 'w')
        param_fic.write(str(self.param))
        param_fic.close()

        sct.printv('\nLoading data dictionary ...', self.param.verbose, 'normal')
        # List of T2star images (im) and their label decision (gmseg) (=segmentation of the gray matter), slice by slice
        self.slices = self.load_data_dictionary()  # type: list of slices
        self.mean_image = np.mean([dic_slice.im for dic_slice in self.slices], axis=0)
        # number of slices in the data set
        self.J = len([dic_slice.im for dic_slice in self.slices])  # type: int
        # dimension of the slices (flattened)
        self.N = len(self.slices[0].im.flatten())  # type: int

        # inverts the segmentation slices : the model uses segmentation of the WM instead of segmentation of the GM
        self.invert_seg()

        sct.printv('\nComputing the transformation to co-register all the data into a common groupwise space (using the white matter segmentations) ...', self.param.verbose, 'normal')
        # mean segmentation image of the dictionary, type: numpy array
        self.mean_wmseg = self.seg_coregistration(transfo_to_apply=self.coregistration_transfos)

        sct.printv('\nCo-registering all the data into the common groupwise space ...', self.param.verbose, 'normal')
        self.coregister_data(transfo_to_apply=self.coregistration_transfos)

        # Normalize dictionary slices
        if self.param.model_slices_normalization:
            sct.printv('\nNormalizing the dictionary slices ...', self.param.verbose, 'normal')
            self.normalize_dic_slices()
            # update dic_metrics after normalization
            self.param.mean_metric = self.normalize_dic_slices(get_dic_metric=True)

        # update the mean image
        self.mean_image = np.mean([dic_slice.im_M for dic_slice in self.slices], axis=0) # type: numpy array

        self.save_dic()

    # ------------------------------------------------------------------------------------------------------------------
    def load_data_dictionary(self):
        """
        each slice of each subject will be loaded separately in a Slice object containing :

        - a slice id

        - the original T2star image crop around the spinal cord: im

        - a manual segmentation of the gray matter: seg

        :return slices: numpy array of all the slices of the data dictionary
        """
        # initialization
        slices = []
        j = -1
        for subject_dir in os.listdir(self.param.path_model):
            subject_path = self.param.path_model + '/' + subject_dir
            if os.path.isdir(subject_path):
                name_slice = None
                if self.param.use_levels == 'float':
                    subject_levels = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: []}
                    j_sub = j
                    i_slice = 0
                for file_name in os.listdir(subject_path):
                    if 'im' in file_name:  # or 'seg_in' in file_name:
                        j += 1
                        slice_int_level = 0
                        name_list = file_name.split('_')
                        for word in name_list:
                            if word.upper() in self.level_label.values():
                                slice_int_level = get_key_from_val(self.level_label, word.upper())
                        slices.append(Slice(slice_id=j, im=Image(subject_path + '/' + file_name).data, level=slice_int_level, list_gm_seg=[], reg_to_m=[]))
                        name_slice = '_'.join(name_list[:-1])

                        if self.param.use_levels == 'float':
                            subject_levels[slice_int_level].append(i_slice)
                            i_slice += 1

                    if name_slice in file_name and ('gm' in file_name or 'seg' in file_name):
                        gm_seg_list = list(slices[j].gm_seg)
                        gm_seg_list.append(Image(subject_path + '/' + file_name).data)
                        slices[j].set(list_gm_seg=gm_seg_list)

                if self.param.use_levels == 'float':
                    for int_level, slices_list in subject_levels.items():
                        n_slices_by_level = len(slices_list)
                        if n_slices_by_level == 1:
                            index = slices_list[0]
                            if index == 0:
                                slices[j_sub+index].set(level=int_level+0.1)
                            elif index == i_slice:
                                slices[j_sub+index].set(level=int_level+0.9)
                            else:
                                slices[j_sub+index].set(level=int_level+0.5)
                        elif n_slices_by_level > 1:
                            gap = 1.0/(n_slices_by_level + 1)
                            for i, index in enumerate(slices_list):
                                slices[j_sub+index].set(level=int_level+((n_slices_by_level-i)*gap))

        return np.asarray(slices)

    # ------------------------------------------------------------------------------------------------------------------
    def invert_seg(self):
        """
        Invert the gray matter segmentation to get segmentation of the white matter instead
        keeps more information, better results
        """
        for dic_slice in self.slices:
            list_wm_seg = [inverse_gmseg_to_wmseg(gmseg, dic_slice.im, save=False) for gmseg in dic_slice.gm_seg]
            dic_slice.set(list_wm_seg=list_wm_seg)

    # ------------------------------------------------------------------------------------------------------------------
    def seg_coregistration(self, transfo_to_apply=None):
        """
        For all the segmentation slices, do a registration of the segmentation slice to the mean segmentation
         applying all the transformations in transfo_to_apply

        Compute, apply and save each transformation warping field for all the segmentation slices

        Compute the new mean segmentation at each step and update self.mean_seg

        :param transfo_to_apply: list of string
        :return resulting_mean_seg:
        """
        current_mean_seg = compute_majority_vote_mean_seg(get_all_seg_from_dic(self.slices, type='wm'))
        first = True
        for transfo in transfo_to_apply:
            sct.printv('Doing a ' + transfo + ' registration of each segmentation slice to the mean segmentation ...', self.param.verbose, 'normal')
            current_mean_seg = self.find_coregistration(mean_seg=current_mean_seg, transfo_type=transfo, first=first)
            first = False

        resulting_mean_seg = current_mean_seg

        return resulting_mean_seg

    # ------------------------------------------------------------------------------------------------------------------
    def find_coregistration(self, mean_seg=None, transfo_type='Affine', first=True):
        """
        For each segmentation slice, apply and save a registration of the specified type of transformation
        the name of the registration file (usually a matlab matrix) is saved in self.RtoM

        :param mean_seg: current mean segmentation

        :param transfo_type: type of transformation for the registration

        :return mean seg: updated mean segmentation
        """

        # Coregistration of the white matter segmentations
        for dic_slice in self.slices:
            name_j_transform = 'transform_slice_' + str(dic_slice.id) + find_ants_transfo_name(transfo_type)[0]
            new_reg_list = dic_slice.reg_to_M.append(name_j_transform)
            dic_slice.set(reg_to_m=new_reg_list)

            if first:
                mean_wm_slice = compute_majority_vote_mean_seg(dic_slice.wm_seg)
                apply_ants_transfo(mean_seg, mean_wm_slice,  transfo_name=name_j_transform, path=self.param.new_model_dir + '/', transfo_type=transfo_type, metric=self.param.reg_metric, search_reg=True, apply_transfo=False)
                list_wm_seg_m = []
                for wm_seg in dic_slice.wm_seg:
                    wm_seg_m = apply_ants_transfo(mean_seg, wm_seg,  transfo_name=name_j_transform, path=self.param.new_model_dir + '/', transfo_type=transfo_type, metric=self.param.reg_metric, search_reg=False, apply_transfo=True)
                    list_wm_seg_m.append(wm_seg_m.astype(int))
                # seg_m = apply_ants_transfo(mean_seg, dic_slice.wm_seg,  transfo_name=name_j_transform, path=self.param.new_model_dir + '/', transfo_type=transfo_type, metric=self.param.reg_metric)
            else:
                mean_wm_slice = compute_majority_vote_mean_seg(dic_slice.wm_seg_M)
                apply_ants_transfo(mean_seg, mean_wm_slice,  transfo_name=name_j_transform, path=self.param.new_model_dir + '/', transfo_type=transfo_type, metric=self.param.reg_metric, search_reg=True, apply_transfo=False)
                list_wm_seg_m = []
                for wm_seg in dic_slice.wm_seg_M:
                    wm_seg_m = apply_ants_transfo(mean_seg, wm_seg,  transfo_name=name_j_transform, path=self.param.new_model_dir + '/', transfo_type=transfo_type, metric=self.param.reg_metric, search_reg=False, apply_transfo=True)
                    list_wm_seg_m.append(wm_seg_m.astype(int))
                # seg_m = apply_ants_transfo(mean_seg, dic_slice.wm_seg_M,  transfo_name=name_j_transform, path=self.param.new_model_dir + '/', transfo_type=transfo_type, metric=self.param.reg_metric)
            dic_slice.set(list_wm_seg_m=list_wm_seg_m)
            # dic_slice.set(wm_seg_m_flat=seg_m.flatten().astype(int))

        mean_seg = compute_majority_vote_mean_seg(get_all_seg_from_dic(self.slices, type='wm_m'))

        return mean_seg

    # ------------------------------------------------------------------------------------------------------------------
    def coregister_data(self,  transfo_to_apply=None):
        """
        Apply to each image slice of the dictionary the transformations found registering the segmentation slices.
        The co_registered images are saved for each slice as im_M

        Delete the directories containing the transformation matrix : not needed after the coregistration of the data.

        :param transfo_to_apply: list of string
        :return:
        """
        mean_gm_seg = compute_majority_vote_mean_seg(get_all_seg_from_dic(self.slices, type='gm'))

        for dic_slice in self.slices:
            list_gm_seg_m = []
            for n_transfo, transfo in enumerate(transfo_to_apply):
                im_m = apply_ants_transfo(self.mean_image, dic_slice.im, search_reg=False, transfo_name=dic_slice.reg_to_M[n_transfo], binary=False, path=self.param.new_model_dir+'/', transfo_type=transfo, metric=self.param.reg_metric)

                for gm_seg in dic_slice.gm_seg:
                    gm_seg_m = apply_ants_transfo(mean_gm_seg, gm_seg, search_reg=False, transfo_name=dic_slice.reg_to_M[n_transfo], binary=True, path=self.param.new_model_dir+'/', transfo_type=transfo, metric=self.param.reg_metric)
                    list_gm_seg_m.append(gm_seg_m)
                    del gm_seg_m
                # apply_2D_rigid_transformation(self.im[j], self.RM[j]['tx'], self.RM[j]['ty'], self.RM[j]['theta'])

            dic_slice.set(im_m=im_m)
            dic_slice.set(list_gm_seg_m=list_gm_seg_m)
            dic_slice.set(im_m_flat=im_m.flatten())

        # Delete the directory containing the transformations : They are not needed anymore
        for transfo_type in transfo_to_apply:
            transfo_dir = transfo_type.lower() + '_transformations'
            if transfo_dir in os.listdir(self.param.new_model_dir + '/'):
                sct.run('rm -rf ' + self.param.new_model_dir + '/' + transfo_dir + '/')

    # ------------------------------------------------------------------------------------------------------------------
    def save_dic(self):
        model_slices = np.asarray([(dic_slice.im_M, tuple(dic_slice.wm_seg_M), tuple(dic_slice.gm_seg_M), dic_slice.level) for dic_slice in self.slices])
        pickle.dump(model_slices, gzip.open(self.param.new_model_dir + '/dictionary_slices.pklz', 'wb'), protocol=2)

    # ------------------------------------------------------------------------------------------------------------------
    def normalize_dic_slices(self, method='median', save=True, get_dic_metric=False):
        dic_metrics = extract_metric_from_slice_set(self.slices, metric=method, save=save)
        wm_metrics = []
        gm_metrics = []
        for wm_m, gm_m, wm_s, gm_s in dic_metrics.values():
            wm_metrics.append(wm_m)
            gm_metrics.append(gm_m)
        dic_wm_mean = np.mean(wm_metrics)
        dic_gm_mean = np.mean(gm_metrics)
        metric = {'wm': dic_wm_mean, 'gm': dic_gm_mean}
        if get_dic_metric:
            return metric
        else:
            for slice in self.slices:
                slice_wm_m, slice_gm_m, slice_wm_s, slice_gm_s = dic_metrics[slice.id]

                old_image = slice.im_M
                old_image[old_image < 0.0001] = 0  # put at 0 the background
                new_image = (old_image - slice_wm_m)*(dic_gm_mean - dic_wm_mean)/(slice_gm_m - slice_wm_m) + dic_wm_mean
                new_image[old_image < 0.0001] = 0  # put at 0 the background
                slice.set(im_m=new_image)

    # ------------------------------------------------------------------------------------------------------------------
    # END OF FUNCTIONS USED TO COMPUTE THE MODEL DICTIONARY
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def load_dic(self):

        model_slices = pickle.load(gzip.open(self.param.path_model + '/dictionary_slices.pklz', 'rb'))

        self.slices = [Slice(slice_id=i_slice, level=dic_slice[3], im_m=dic_slice[0], list_wm_seg_m=list(dic_slice[1]), list_gm_seg_m=list(dic_slice[2]), im_m_flat=dic_slice[0].flatten()) for i_slice, dic_slice in enumerate(model_slices)]  # type: list of slices

        # number of slices in the data set
        self.J = len([dic_slice.im_M for dic_slice in self.slices])  # type: int
        # dimension of the slices (flattened)
        self.N = len(self.slices[0].im_M_flat)  # type: int

        self.mean_wmseg = compute_majority_vote_mean_seg(get_all_seg_from_dic(self.slices, type='wm_m'))
        self.mean_gmseg = compute_majority_vote_mean_seg(get_all_seg_from_dic(self.slices, type='gm_m'))

    # ------------------------------------------------------------------------------------------------------------------
    def mean_seg_by_level(self, type='binary', save=False):
        gm_seg_by_level = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': [], 'T3': [], 'T4': [], 'T5': [], 'T6': [], 'T7': [], 'T8': [], 'T10': [], 'T11': [], 'T12': [], 'L1': [], 'L2': [], 'L3': [], 'L4': [], 'L5': [], '': []}
        im_by_level = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': [], 'T3': [], 'T4': [], 'T5': [], 'T6': [], 'T7': [], 'T8': [], 'T10': [], 'T11': [], 'T12': [], 'L1': [], 'L2': [], 'L3': [], 'L4': [], 'L5': [], '': []}
        for dic_slice in self.slices:
            for gm_seg_m in dic_slice.gm_seg_M:
                gm_seg_by_level[self.level_label[int(dic_slice.level)]].append(gm_seg_m)
            im_by_level[self.level_label[int(dic_slice.level)]].append(dic_slice.im_M)
        seg_averages = {}
        im_averages = {}
        for level, seg_data_set in gm_seg_by_level.items():
            seg_averages[level] = compute_majority_vote_mean_seg(seg_data_set=seg_data_set, type=type)
        seg_averages[''] = compute_majority_vote_mean_seg(seg_data_set=get_all_seg_from_dic(self.slices, type='gm_m'), type=type)
        for level, im_data_set in im_by_level.items():
            im_averages[level] = np.mean(im_data_set, axis=0)
        im_averages[''] = np.mean([dic_slice.im_M for dic_slice in self.slices], axis=0)

        for dic in [seg_averages, im_averages]:
            for level, average in dic.items():
                if np.asarray(average).shape == ():
                    dic[level] = np.zeros(dic[''].shape)

        if save:
            for level, mean_gm_seg in seg_averages.items():
                Image(param=mean_gm_seg, absolutepath='./mean_seg_' + level + '.nii.gz').save()
            for level, mean_im in im_averages.items():
                Image(param=mean_im, absolutepath='./mean_im_' + level + '.nii.gz').save()
        return seg_averages, im_averages

    # ------------------------------------------------------------------------------------------------------------------
    def show_dictionary_data(self):
        """
        show the 10 first slices of the model dictionary
        """
        import matplotlib.pyplot as plt
        for dic_slice in self.slices[:10]:
            fig = plt.figure()

            if dic_slice.wm_seg is not None:
                seg_subplot = fig.add_subplot(2, 3, 1)
                seg_subplot.set_title('Original space - seg')
                im_seg = seg_subplot.imshow(dic_slice.wm_seg)
                im_seg.set_interpolation('nearest')
                im_seg.set_cmap('gray')

            seg_m_subplot = fig.add_subplot(2, 3, 2)
            seg_m_subplot.set_title('Common groupwise space - seg')
            im_seg_m = seg_m_subplot.imshow(dic_slice.wm_seg_M)
            im_seg_m.set_interpolation('nearest')
            im_seg_m.set_cmap('gray')

            if self.mean_wmseg is not None:
                mean_seg_subplot = fig.add_subplot(2, 3, 3)
                mean_seg_subplot.set_title('Mean seg')
                im_mean_seg = mean_seg_subplot.imshow(np.asarray(self.mean_wmseg))
                im_mean_seg.set_interpolation('nearest')
                im_mean_seg.set_cmap('gray')

            if dic_slice.im is not None:
                slice_im_subplot = fig.add_subplot(2, 3, 4)
                slice_im_subplot.set_title('Original space - data ')
                im_slice_im = slice_im_subplot.imshow(dic_slice.im)
                im_slice_im.set_interpolation('nearest')
                im_slice_im.set_cmap('gray')

            slice_im_m_subplot = fig.add_subplot(2, 3, 5)
            slice_im_m_subplot.set_title('Common groupwise space - data ')
            im_slice_im_m = slice_im_m_subplot.imshow(dic_slice.im_M)
            im_slice_im_m.set_interpolation('nearest')
            im_slice_im_m.set_cmap('gray')

            plt.suptitle('Slice ' + str(dic_slice.id))
            plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# MODEL ---------------------------------------------------------------------------------------------------------------
class Model:
    """
    Model used by the supervised gray matter segmentation method

    """
    def __init__(self, model_param=None):
        """
        Model constructor

        :param model_param: model parameters, type: Param
        """
        if model_param is None:
            self.param = ModelParam()
        else:
            self.param = model_param

        # Model dictionary
        self.dictionary = ModelDictionary(dic_param=self.param)

        self.pca = None
        self.epsilon = round(1.0/self.dictionary.J, 4)/2
        self.tau = 0

        if self.param.todo_model == 'compute':
            self.compute_model()
        elif self.param.todo_model == 'load':
            self.load_model()

        if self.param.verbose == 2:
            self.pca.plot_projected_dic()

    # ------------------------------------------------------------------------------------------------------------------
    def compute_model(self):
        sct.printv('\nCreating a reduced common space (using a PCA) ...', self.param.verbose, 'normal')
        self.pca = PCA(np.asarray(self.dictionary.slices), k=self.param.k, verbose=self.param.verbose)
        self.pca.save_data(self.param.new_model_dir)
        # updating the dictionary mean_image
        self.dictionary.mean_image = self.pca.mean_image

        self.tau = self.compute_tau()
        pickle.dump(self.tau, open(self.param.new_model_dir + '/tau_levels_'+str(self.param.use_levels)+'.txt', 'w'), protocol=0)  # or protocol=2 and 'wb'

    # ------------------------------------------------------------------------------------------------------------------
    def load_model(self):
        sct.printv('\nLoading a reduced common space (using a PCA) ...', self.param.verbose, 'normal')
        pca_data = pickle.load(gzip.open(self.param.path_model + '/pca_data.pklz', 'rb'))
        self.pca = PCA(np.asarray(self.dictionary.slices), mean_vect=pca_data[0], eig_pairs=pca_data[1], k=self.param.k, verbose=self.param.verbose)
        # updating the dictionary mean_image
        self.dictionary.mean_image = self.pca.mean_image
        self.tau = pickle.load(open(self.param.path_model + '/tau_levels_'+str(self.param.use_levels)+'.txt', 'r'))  # if protocol was 2 : 'rb'

    # ------------------------------------------------------------------------------------------------------------------
    def compute_beta(self, coord_target, target_levels=None, dataset_coord=None, dataset_levels=None, tau=0.006):
        """
        Compute the model similarity (beta) between each model slice and each target image slice

        beta_j = (1/Z)exp(-tau*square_norm(target_coordinate - slice_j_coordinate))

        Z is the partition function that enforces the constraint that sum(beta)=1

        :param coord_target: coordinates of the target image in the reduced model space

        :param tau: weighting parameter indicating the decay constant associated with a geodesic distance
        between a given dictionary slice and a projected target image slice

        :return:
        """
        if dataset_coord is None:
            # in the dataset_coord matrix, each column correspond to the projection of one of the original data image,
            # the transpose operator .T enable the loop to iterate over all the images coord
            dataset_coord = self.pca.dataset_coord.T
            dataset_levels = [dic_slice.level for dic_slice in self.dictionary.slices]

        beta = []
        if self.param.mode_weight_similarity:
            mode_weight = [val/sum(self.pca.kept_eigenval) for val in self.pca.kept_eigenval]
            # TODO: WARNING: see if the weights shouldnt be inversed: a bigger weight for the first modes will make the distances along those modes bigger: maybe we want to do the opposite
        else:
            mode_weight = None

        # 3D TARGET
        if isinstance(coord_target[0], (list, np.ndarray)):
            for i_target, coord_projected_slice in enumerate(coord_target):
                beta_slice = []
                for j_slice, coord_slice_j in enumerate(dataset_coord):
                    if mode_weight is None:
                        square_norm = np.linalg.norm((coord_projected_slice - coord_slice_j), 2)
                    else:
                        from scipy.spatial.distance import wminkowski
                        square_norm = wminkowski(coord_projected_slice, coord_slice_j, 2, mode_weight)

                    if target_levels is not None and target_levels is not [None] and self.param.use_levels is not '0':
                        if self.param.equation_id == 1:
                            # EQUATION #1 (better results ==> kept)
                            beta_slice.append(exp(-self.param.weight_gamma*abs(target_levels[i_target] - dataset_levels[j_slice]))*exp(-tau*square_norm))  # TODO: before = no absolute
                        elif self.param.equation_id == 2:
                            # EQUATION #2
                            if target_levels[i_target] == dataset_levels[j_slice]:
                                beta_slice.append(exp(tau*square_norm))
                            else:
                                beta_slice.append(exp(-tau*square_norm)/self.param.weight_gamma*abs(target_levels[i_target] - dataset_levels[j_slice])) #TODO: before = no absolute

                    else:
                        beta_slice.append(exp(-tau*square_norm))

                try:
                    beta_slice /= np.sum(beta_slice)
                except ZeroDivisionError:
                    sct.printv('WARNING : similarities are null', self.param.verbose, 'warning')
                    print beta_slice

                beta.append(beta_slice)

        # 2D TARGET
        else:
            for j_slice, coord_slice_j in enumerate(dataset_coord):
                if mode_weight is None:
                    square_norm = np.linalg.norm((coord_target - coord_slice_j), 2)
                else:
                    from scipy.spatial.distance import wminkowski
                    square_norm = wminkowski(coord_target, coord_slice_j, 2, mode_weight)
                if target_levels is not None and self.param.use_levels is not '0':
                    if self.param.equation_id == 1:
                        # EQUATION #1 (better results ==> kept)
                        beta.append(exp(-self.param.weight_gamma*abs(target_levels - dataset_levels[j_slice]))*exp(-tau*square_norm) )#TODO: before = no absolute
                    elif self.param.equation_id == 2:
                        # EQUATION #2
                        if target_levels == dataset_levels[j_slice]:
                            beta.append(exp(tau*square_norm))
                        else:
                            beta.append(exp(-tau*square_norm)/self.param.weight_gamma*abs(target_levels - dataset_levels[j_slice]))
                else:
                    beta.append(exp(-tau*square_norm))

            try:
                beta /= np.sum(beta)
            except ZeroDivisionError:
                sct.printv('WARNING : similarities are null', self.param.verbose, 'warning')
                print beta

        return np.asarray(beta)

    # ------------------------------------------------------------------------------------------------------------------
    def compute_tau(self):
        """
        Compute the weighting parameter indicating the decay constant associated with a geodesic distance
        between a given dictionary slice and a projected target image slice
        :return:
        """
        sct.printv('\nComputing Tau ... \n'
                   '(Tau is a weighting parameter indicating the decay constant associated with a geodesic distance between a given atlas and a projected target image, see [Asman et al., Medical Image Analysis 2014], eq (16))', 1, 'normal')
        from scipy.optimize import minimize

        def to_minimize(tau):
            """
            Compute the sum of the L0 norm between a slice segmentation and the resulting segmentation that would be
            found if the slice was a target image for a given tau

            For a given model, Tau is the parameter that would minimize this function

            :param tau:

            :return sum_norm:

            """
            sum_norm = 0
            for dic_slice in self.dictionary.slices:
                projected_dic_slice_coord = self.pca.project_array(dic_slice.im_M_flat)
                coord_dic_slice_dataset = np.delete(self.pca.dataset_coord.T, dic_slice.id, 0)
                if self.param.use_levels is not '0':
                    dic_slice_dataset_levels = np.delete(np.asarray(dic_levels), dic_slice.id, 0)
                    beta_dic_slice = self.compute_beta(projected_dic_slice_coord, target_levels=dic_slice.level, dataset_coord=coord_dic_slice_dataset, dataset_levels=dic_slice_dataset_levels, tau=tau)
                else:
                    beta_dic_slice = self.compute_beta(projected_dic_slice_coord, target_levels=None, dataset_coord=coord_dic_slice_dataset, dataset_levels=None, tau=tau)
                kj = self.select_k_slices(beta_dic_slice)
                if self.param.weight_label_fusion:
                    est_segm_j = self.label_fusion(dic_slice, kj, beta=beta_dic_slice)[0]
                else:
                    # default case
                    est_segm_j = self.label_fusion(dic_slice, kj)[0]

                sum_norm += l0_norm(compute_majority_vote_mean_seg(dic_slice.wm_seg_M), est_segm_j.data)

            return sum_norm

        dic_levels = [dic_slice.level for dic_slice in self.dictionary.slices]

        est_tau = minimize(to_minimize, 0.001, method='Nelder-Mead', options={'xtol': 0.0005})
        sct.printv('Estimated tau : ' + str(est_tau.x[0]))

        return float(est_tau.x[0])

    # ------------------------------------------------------------------------------------------------------------------
    def select_k_slices(self, beta):
        """
        Select the K dictionary slices most similar to the target slice

        :param beta: Dictionary similarities

        :return selected: numpy array of segmentation of the selected dictionary slices
        """
        kept_slice_index = []

        if isinstance(beta[0], (list, np.ndarray)):
            for beta_slice in beta:
                selected_index = beta_slice > self.epsilon
                kept_slice_index.append(selected_index)

        else:
            kept_slice_index = beta > self.epsilon

        return np.asarray(kept_slice_index)

    # ------------------------------------------------------------------------------------------------------------------
    def label_fusion(self, target, selected_index, beta=None, type='binary'):

        """
        Compute the resulting segmentation by label fusion of the segmentation of the selected dictionary slices

        :param selected_index: array of indexes (as a boolean array) of the selected dictionary slices

        :return res_seg_model_space: Image of the resulting segmentation for the target image (in the model space)
        """
        wm_segmentation_slices = np.asarray([dic_slice.wm_seg_M for dic_slice in self.dictionary.slices])
        gm_segmentation_slices = np.asarray([dic_slice.gm_seg_M for dic_slice in self.dictionary.slices])

        res_wm_seg_model_space = []
        res_gm_seg_model_space = []

        if isinstance(selected_index[0], (list, np.ndarray)):
            # 3D image
            for i, selected_ind_by_slice in enumerate(selected_index):  # selected_slices:
                # did not adapted the WEIGHTED LABEL FUSION to multiple segmentations per slice
                '''
                if beta is None:
                    n_selected_dic_slices = wm_segmentation_slices[selected_ind_by_slice].shape[0]
                    if n_selected_dic_slices > 0:
                        weights = [1.0/n_selected_dic_slices] * n_selected_dic_slices
                    else:
                        weights = None
                else:
                    weights = beta[i][selected_ind_by_slice]
                    weights = [w/sum(weights) for w in weights]
                '''
                # if slices from teh dictionary were selected:
                if np.any(selected_ind_by_slice):
                    weights = None

                    #list_selected_slices_wm = np.array(wm_segmentation_slices[selected_ind_by_slice])
                    selected_slices_wmseg = []
                    for seg_by_slice in wm_segmentation_slices[selected_ind_by_slice]:
                        for seg in seg_by_slice:
                            selected_slices_wmseg.append(seg)

                    wm_slice_seg = compute_majority_vote_mean_seg(selected_slices_wmseg, weights=weights, type=type, threshold=0.50001)
                    res_wm_seg_model_space.append(wm_slice_seg)
                    target[i].set(list_wm_seg_m=[wm_slice_seg])

                    # list_selected_slices_gm = gm_segmentation_slices[selected_ind_by_slice]
                    selected_slices_gmseg = []
                    for seg_by_slice in gm_segmentation_slices[selected_ind_by_slice]:
                        for seg in seg_by_slice:
                            selected_slices_gmseg.append(seg)
                    gm_slice_seg = compute_majority_vote_mean_seg(selected_slices_gmseg, weights=weights, type=type)
                    res_gm_seg_model_space.append(gm_slice_seg)
                    target[i].set(list_gm_seg_m=[gm_slice_seg])
                else:
                    # no selected dictionary slices
                    target[i].set(list_wm_seg_m=[self.dictionary.mean_wmseg])
                    target[i].set(list_gm_seg_m=[self.dictionary.mean_gmseg])


        else:
            # 2D image
            # did not adapted the WEIGHTED LABEL FUSION to multiple segmentations per slice
            '''
            if beta is None:
                n_selected_dic_slices = wm_segmentation_slices[selected_index].shape[0]
                weights = [1.0/n_selected_dic_slices] * n_selected_dic_slices
            else:
                weights = beta[selected_index]
                weights = [w/sum(weights) for w in weights]
            '''
            weights = None

            selected_slices_wmseg = []
            for seg_by_slice in wm_segmentation_slices[selected_index]:
                for seg in seg_by_slice:
                    selected_slices_wmseg.append(seg)

            selected_slices_gmseg = []
            for seg_by_slice in gm_segmentation_slices[selected_index]:
                for seg in seg_by_slice:
                    selected_slices_gmseg.append(seg)

            res_wm_seg_model_space = compute_majority_vote_mean_seg(np.array(selected_slices_wmseg), weights=weights, type=type, threshold=0.50001)
            res_gm_seg_model_space = compute_majority_vote_mean_seg(np.array(selected_slices_gmseg), weights=weights, type=type)

        res_wm_seg_model_space = np.asarray(res_wm_seg_model_space)
        res_gm_seg_model_space = np.asarray(res_gm_seg_model_space)

        return Image(param=res_wm_seg_model_space), Image(param=res_gm_seg_model_space)


# ----------------------------------------------------------------------------------------------------------------------
# TARGET SEGMENTATION PAIRWISE -----------------------------------------------------------------------------------------
class TargetSegmentationPairwise:
    """
    Contains all the function to segment the gray matter an a target image given a model

        - registration of the target to the model space

        - projection of the target slices on the reduced model space

        - selection of the model slices most similar to the target slices

        - computation of the resulting target segmentation by label fusion of their segmentation
    """
    def __init__(self, model, target_image=None, levels_image=None, epsilon=None, seg_param=None):
        """
        Target gray matter segmentation constructor

        :param model: Model used to compute the segmentation, type: Model

        :param target_image: Target image to segment gray matter on, type: Image

        """
        self.model = model
        self.param = seg_param if seg_param is not None else SegmentationParam()

        self.im_target = target_image
        self.target_slices = []
        self.target_dim = 0

        self.im_levels = levels_image

        self.coord_projected_target = None
        self.beta = None
        self.selected_k_slices = None

        # Process and segment the target image
        self.process()

    # ------------------------------------------------------------------------------------------------------------------
    def process(self):
        # ####### Initialization of the target image #######
        if len(self.im_target.data.shape) == 3:
            self.target_slices = [Slice(slice_id=i_slice, im=target_slice, reg_to_m=[]) for i_slice, target_slice in enumerate(self.im_target.data)]
            self.target_dim = 3
        elif len(self.im_target.data.shape) == 2:
            self.target_slices = [Slice(slice_id=0, im=self.im_target.data, reg_to_m=[])]
            self.target_dim = 2

        if self.im_levels is not None and self.model.param.use_levels is not '0':
            list_levels = load_level(self.im_levels, type=self.model.param.use_levels, verbose=self.param.verbose)
            for target_slice, l in zip(self.target_slices, list_levels):
                target_slice.set(level=l)

        # ####### Registration of the target slices to the dictionary space #######
        self.target_pairwise_registration()

        if self.param.target_normalization:
            # ####### Normalization of the intensity #######
            self.target_normalization(method='median')  # 'mean')

        # TODO: remove after testing:
        Image(param=np.asarray([target_slice.im_M for target_slice in self.target_slices]), absolutepath='target_moved_after_normalization.nii.gz').save()

        sct.printv('\nProjecting the target image in the reduced common space ...', self.param.verbose, 'normal')
        # coord_projected_target is a list of all the coord of the target's projected slices
        self.coord_projected_target = self.model.pca.project([target_slice.im_M for target_slice in self.target_slices])

        sct.printv('\nComputing the similarities between the target and the model slices ...', self.param.verbose, 'normal')
        if self.im_levels is not None and self.model.param.use_levels is not '0':
            target_levels = np.asarray([target_slice.level for target_slice in self.target_slices])
        else:
            target_levels = None
        # ##### The similarities beta are the similarity of each slice of the target with each slice of the dictionary.
        # the similarities beta are used to select the slices of the dictionary that will be used to compute the target segmentation
        self.beta = self.model.compute_beta(self.coord_projected_target, target_levels=target_levels, tau=self.model.tau)

        sct.printv('\nSelecting the dictionary slices most similar to the target ...', self.param.verbose, 'normal')
        # ##### The selected k slices are the dictionary slices used to compute the target segmentation
        self.selected_k_slices = self.model.select_k_slices(self.beta)
        self.save_selected_slices(self.im_target.file_name[:-3])

        if self.param.verbose == 2:
            # Display the selected slices in the PCA space
            self.plot_projected_dic(nb_modes=3, to_highlight=None)  # , to_highlight='all')  # , to_highlight=(6, self.selected_k_slices[6]))

        sct.printv('\nComputing the result gray matter segmentation ...', self.param.verbose, 'normal')
        if self.model.param.weight_label_fusion:
            use_beta = self.beta
        else:
            use_beta = None
        # ##### The selected k slcies are "averaged" to get the target segmentation
        self.model.label_fusion(self.target_slices, self.selected_k_slices, beta=use_beta, type=self.param.res_type)

        if self.param.z_regularisation:
            sct.printv('\nRegularisation of the segmentation along the Z axis ...', self.param.verbose, 'normal')
            self.z_regularisation_2d_iteration()

        sct.printv('\nRegistering the result gray matter segmentation back into the target original space...', self.param.verbose, 'normal')
        self.target_pairwise_registration(inverse=True)

    # ------------------------------------------------------------------------------------------------------------------
    def target_normalization(self, method='median', dic_wm_mean=None, dic_gm_mean=None):
        """
        Normalization of the target using the intensity values of the mean dictionary image
        :return None: the target image is modified
        """
        sct.printv('Linear target normalization using '+method+' ...', self.param.verbose, 'normal')

        if method == 'mean' or method == 'median':
            if self.model.dictionary.param.mean_metric is None:
                metrics = self.model.dictionary.normalize_dic_slices(method=method, get_dic_metric=True)
                dic_wm_mean = metrics['wm']
                dic_gm_mean = metrics['gm']

            # getting the mean values of WM and GM in the target
            if self.param.target_means is None:
                if self.model.param.use_levels is not '0':
                    seg_averages_by_level = self.model.dictionary.mean_seg_by_level(type='binary')[0]
                    mean_seg_by_level = [seg_averages_by_level[self.model.dictionary.level_label[int(target_slice.level)]] for target_slice in self.target_slices]

                    # saving images to check results of normalisation
                    Image(param=np.asarray(mean_seg_by_level), absolutepath='mean_seg_by_level.nii.gz').save()
                    Image(param=np.asarray([target_slice.im_M for target_slice in self.target_slices]), absolutepath='target_moved.nii.gz').save()

                    target_metric = extract_metric_from_slice_set(self.target_slices, seg_to_use=mean_seg_by_level, metric=method, save=True, output='metric_in_target.txt')

                else:
                    sct.printv('WARNING: No mean value of the white matter and gray matter intensity were provided, nor the target vertebral levels to estimate them\n'
                               'The target will not be normalized.', self.param.verbose, 'warning')
                    self.param.target_normalization = False
                    target_metric = None
            else:
                target_metric = {}
                for i in range(len(self.target_slices)):
                    target_metric[i] = (self.param.target_means[0], self.param.target_means[1], 0, 0)

            if type(target_metric) == type({}):  # if target_metric is a dictionary
                val_differences = [m[1]-m[0] for m in target_metric.values()]
                val_std = [np.median([m[2], m[3]]) for m in target_metric.values()]
                val_diff_med = np.median(val_differences)
                val_diff_std = np.std(val_differences)

                val_std_med =np.median(val_std)

                # contrast_type = diff_med /np.abs(diff_med)  # if 1: GM bright, WM dark ; if -1: GM dark, WM bright
                lim_diff = np.abs(val_diff_med - val_diff_std)
            else:
                val_differences = [0]
                val_diff_med = 0
                val_std_med = 0
                lim_diff = 0

            # normalizing
            if target_metric is not None:
                if val_diff_med < val_std_med:
                    # If the difference GM-WM is smaller than the std of the values in GM and WM, the contrast isn't sharp enough to use this normalization method
                    # recall function with method min-max
                    sct.printv('WARNING: The contrast between GM and WM is too small to use the normalization based on median WM-GM values estimated with a pre-registration of the GM template, it will be replaced by a linear normalization using minimum/maximum intensity values', self.param.verbose, 'warning')
                    self.target_normalization(method='min-max')
                else:
                    i = 0
                    for target_slice in self.target_slices:
                        old_image = target_slice.im_M

                        wm_metric, gm_metric, wm_std, gm_std = target_metric[target_slice.id]
                        if np.abs(gm_metric-wm_metric) < lim_diff:
                            wm_metric = gm_metric-val_diff_med  # np.median(differences)  # if med>0: GM bright, WM dark ; if med<0: GM dark, WM bright
                        old_image[old_image < 0.0001] = 0  # put at 0 the background
                        new_image = (old_image - wm_metric)*(dic_gm_mean - dic_wm_mean)/(gm_metric - wm_metric) + dic_wm_mean
                        new_image[old_image < 0.0001] = 0  # put at 0 the background

                        target_slice.im_M = new_image
                        Image(param=new_image, absolutepath='target_slice'+str(i)+'_mean_normalized.ni.gz').save()
                        i += 1

        if method == 'min-max':
            min_sum = 0
            for model_slice in self.model.dictionary.slices:
                min_sum += model_slice.im_M[model_slice.im_M > 1].min()
            new_min = min_sum/self.model.dictionary.J
            # new_min = self.model.dictionary.mean_image[self.model.dictionary.mean_image > 300].min()
            new_max = self.model.dictionary.mean_image.max()

            i = 0
            for target_slice in self.target_slices:
                # with mean image as reference
                # target_slice.im = target_slice.im/self.model.dictionary.mean_image*self.model.dictionary.mean_image.max()

                # linear with min=0
                # target_slice.im = target_slice.im*self.model.dictionary.mean_image.max()/(target_slice.im.max()-target_slice.im.min())

                # linear with actual min (WM min)
                old_image = target_slice.im_M
                old_min = target_slice.im_M[target_slice.im_M >= 0].min()
                old_max = target_slice.im_M.max()
                new_image = (old_image - old_min)*(new_max - new_min)/(old_max - old_min) + new_min
                # new_image[new_image < new_min+1] = 0  # put a 0 the min background
                new_image[old_image < 1] = 0

                target_slice.im_M = new_image
                Image(param=new_image, absolutepath='target_slice'+str(i)+'_min_max_normalized.ni.gz').save()
                i += 1

        if method == 'mean-sep':
            # test normalization with separate means
            import copy
            dic_metrics = extract_metric_from_slice_set(self.model.dictionary.slices, save=True)
            wm_metrics = []
            gm_metrics = []
            wm_stds = []
            gm_stds = []
            for wm_m, gm_m, wm_s, gm_s in dic_metrics.values():
                wm_metrics.append(wm_m)
                gm_metrics.append(gm_m)
                wm_stds.append(wm_s)
                gm_stds.append(gm_s)
            dic_wm_mean = np.mean(wm_metrics)
            dic_gm_mean = np.mean(gm_metrics)
            dic_wm_std = np.std(wm_stds)
            dic_gm_std = np.std(gm_stds)

            print 'Dic wm: ', dic_wm_mean, ' +- ', dic_wm_std
            print 'Dic gm: ', dic_gm_mean, ' +- ', dic_gm_std

            seg_averages_by_level_bin = self.model.dictionary.mean_seg_by_level(type='binary')[0]
            mean_seg_by_level_bin = [seg_averages_by_level_bin[self.model.dictionary.level_label[int(target_slice.level)]] for target_slice in self.target_slices]
            seg_averages_by_level_prob = self.model.dictionary.mean_seg_by_level(type='prob')[0]
            mean_seg_by_level_prob = [seg_averages_by_level_prob[self.model.dictionary.level_label[int(target_slice.level)]] for target_slice in self.target_slices]

            target_metric = extract_metric_from_slice_set(self.target_slices, seg_to_use=mean_seg_by_level_bin, save=True)

            for i, target_slice in enumerate(self.target_slices):
                old_image = target_slice.im_M
                # new_image = copy.deepcopy(old_image)
                wm_metric, gm_metric, wm_std, gm_std = target_metric[target_slice.id]

                # GM:
                new_gm = ((old_image - gm_metric)*dic_gm_std/gm_std+dic_gm_mean)*mean_seg_by_level_prob[i]
                # WM:
                new_wm = ((old_image - gm_metric)*dic_gm_std/gm_std+dic_gm_mean)*(1-mean_seg_by_level_prob[i])
                # concatenation of GM and WM:
                new_image = new_wm + new_gm

                new_image[old_image < 1] = 0
                target_slice.im_M = new_image

                Image(param=new_image, absolutepath='target_slice'+str(i)+'_mean_sep_normalized.ni.gz').save()

    # ------------------------------------------------------------------------------------------------------------------
    def target_pairwise_registration(self, inverse=False):
        """
        Register the target image into the model space

        Affine (or rigid + affine) registration of the target on the mean model image --> pairwise

        :param inverse: if True, apply the inverse warping field of the registration target -> model space
        to the result gray matter segmentation of the target
        (put it back in it's original space)

        :return None: the target attributes are set in the function
        """
        if not inverse:
            # Registration target --> model space
            mean_dic_im = self.model.pca.mean_image
            for i, target_slice in enumerate(self.target_slices):
                moving_target_slice = target_slice.im
                for transfo in self.model.dictionary.coregistration_transfos:
                    transfo_name = transfo + '_transfo_target2model_space_slice_' + str(i) + find_ants_transfo_name(transfo)[0]
                    target_slice.reg_to_M.append((transfo, transfo_name))

                    moving_target_slice = apply_ants_transfo(mean_dic_im, moving_target_slice, binary=False, transfo_type=transfo, transfo_name=transfo_name, metric=self.model.param.reg_metric)
                self.target_slices[i].set(im_m=moving_target_slice)

        else:
            # Inverse registration result in model space --> target original space
            for i, target_slice in enumerate(self.target_slices):
                moving_wm_seg_slice = target_slice.wm_seg_M[0]
                moving_gm_seg_slice = target_slice.gm_seg_M[0]

                for transfo in target_slice.reg_to_M:
                    if self.param.res_type == 'binary':
                        bin = True
                    else:
                        bin = False
                    moving_wm_seg_slice = apply_ants_transfo(self.model.dictionary.mean_wmseg, moving_wm_seg_slice, search_reg=False, binary=bin, inverse=1, transfo_type=transfo[0], transfo_name=transfo[1], metric=self.model.param.reg_metric)
                    moving_gm_seg_slice = apply_ants_transfo(self.model.dictionary.mean_gmseg, moving_gm_seg_slice, search_reg=False, binary=bin, inverse=1, transfo_type=transfo[0], transfo_name=transfo[1], metric=self.model.param.reg_metric)

                target_slice.set(list_wm_seg=[moving_wm_seg_slice])
                target_slice.set(list_gm_seg=[moving_gm_seg_slice])

    # ------------------------------------------------------------------------------------------------------------------
    def z_regularisation_2d_iteration(self, coeff=0.4):
        """
        Z regularisation option WARNING: DOESN'T IMPROVE THE GM SEGMENTATION RESULT
        Use the result segmentation of the first iteration, the segmentation of slice i is the weighted average of the segmentations of slices i-1 and i+1 and the segmentation of slice i
        :param coeff: weight on each adjacent slice
        :return:
        """
        for i, target_slice in enumerate(self.target_slices[1:-1]):
            adjacent_wm_seg = []  # coeff * self.target[i-1].wm_seg_M, (1-2*coeff) * target_slice.wm_seg_M, coeff * self.target[i+1].wm_seg_M
            adjacent_gm_seg = []  # coeff * self.target[i-1].gm_seg_M, (1-2*coeff) * target_slice.gm_seg_M, coeff * self.target[i+1].gm_seg_M

            precision = 100
            print int(precision*coeff)
            for k in range(int(precision*coeff)):
                adjacent_wm_seg.append(self.target_slices[i-1].wm_seg_M)
                adjacent_wm_seg.append(self.target_slices[i+1].wm_seg_M)
                adjacent_gm_seg.append(self.target_slices[i-1].gm_seg_M)
                adjacent_gm_seg.append(self.target_slices[i+1].gm_seg_M)

            for k in range(precision - 2*int(precision*coeff)):
                adjacent_wm_seg.append(target_slice.wm_seg_M)
                adjacent_gm_seg.append(target_slice.gm_seg_M)

            adjacent_wm_seg = np.asarray(adjacent_wm_seg)
            adjacent_gm_seg = np.asarray(adjacent_gm_seg)

            new_wm_seg = compute_majority_vote_mean_seg(adjacent_wm_seg, type=self.param.res_type, threshold=0.50001)
            new_gm_seg = compute_majority_vote_mean_seg(adjacent_gm_seg, type=self.param.res_type)  # , threshold=0.4999)

            target_slice.set(wm_seg_m=new_wm_seg)
            target_slice.set(gm_seg_m=new_gm_seg)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_projected_dic(self, nb_modes=3, to_highlight=1):
        """
        plot the pca first modes and the target projection if target is provided.

        on a second plot, highlight the selected dictionary slices for one target slice in particular

        :param nb_modes:
        :return:
        """
        self.model.pca.plot_projected_dic(nb_modes=nb_modes, target_coord=self.coord_projected_target, target_levels=[t_slice.level for t_slice in self.target_slices]) if self.coord_projected_target is not None \
            else self.model.pca.plot_projected_dic(nb_modes=nb_modes)

        if to_highlight == 'all':
            for i in range(len(self.target_slices)):
                self.model.pca.plot_projected_dic(nb_modes=nb_modes, target_coord=self.coord_projected_target, target_levels=[t_slice.level for t_slice in self.target_slices], to_highlight=(i, self.selected_k_slices[i]))
        elif to_highlight is not None:
            self.model.pca.plot_projected_dic(nb_modes=nb_modes, target_coord=self.coord_projected_target, target_levels=[t_slice.level for t_slice in self.target_slices], to_highlight=(to_highlight, self.selected_k_slices[to_highlight])) if self.coord_projected_target is not None \
            else self.model.pca.plot_projected_dic()

    # ------------------------------------------------------------------------------------------------------------------
    def save_selected_slices(self, target_name):
        slice_levels = np.asarray([(dic_slice.id, self.model.dictionary.level_label[int(dic_slice.level)]) for dic_slice in self.model.dictionary.slices])
        fic_selected_slices = open(target_name + '_selected_slices.txt', 'w')
        if self.target_dim == 2:
            fic_selected_slices.write(str(slice_levels[self.selected_k_slices.reshape(self.model.dictionary.J,)]))
        elif self.target_dim == 3:
            for target_slice in self.target_slices:
                fic_selected_slices.write('slice ' + str(target_slice.id) + ': ' + str(slice_levels[self.selected_k_slices[target_slice.id]]) + '\n')
        fic_selected_slices.close()


# ----------------------------------------------------------------------------------------------------------------------
# SUPERVISED SEGMENTATION METHOD ---------------------------------------------------------------------------------------
class SupervisedSegmentationMethod():
    """
    Supervised segmentation method:

    Load a dictionary (training data set), compute or load a model from this dictionary
sct_Image
    Load a target image to segment and do the segmentation using the model
    """
    def __init__(self, target_fname, level, model, gm_seg_param=None):
        # build the appearance model
        self.model = model
        self.param = gm_seg_param
        sct.printv('\nConstructing target image ...', verbose=gm_seg_param.verbose, type='normal')
        # construct target image
        self.im_target = Image(target_fname)
        self.original_hdr = self.im_target.hdr

        self.level = level
        self.im_level = None

        self.target_seg_methods = None

        self.res_wm_seg = None
        self.res_gm_seg = None

        self.segment()

    # ------------------------------------------------------------------------------------------------------------------
    def segment(self):
        if self.level is not None:
            if os.path.isfile(self.level) and 'nii' in sct.extract_fname(self.level)[2]:
                self.im_level = Image(self.level)
            else:
                # in this case the level is a string or a file name in .txt, not an image
                self.im_level = self.level

        else:
            self.param.use_levels = '0'

        # TARGET PAIRWISE SEGMENTATION
        self.target_seg_methods = TargetSegmentationPairwise(self.model, target_image=self.im_target, levels_image=self.im_level, seg_param=self.param)

        # get & save the result gray matter segmentation
        # if self.param.output_name == '':
        suffix = ''
        if self.param.dev:
            suffix += '_' + self.param.res_type
            for transfo in self.model.dictionary.coregistration_transfos:
                suffix += '_' + transfo
            if self.model.param.use_levels is not '0':
                suffix += '_with_levels_' + '_'.join(str(self.model.param.weight_gamma).split('.'))  # replace the '.' by a '_'
            else:
                suffix += '_no_levels'
            if self.model.param.z_regularisation:
                suffix += '_Zregularisation'
            if self.model.param.target_normalization:
                suffix += '_normalized'

        name_res_wmseg = self.im_target.file_name + '_wmseg' + suffix  # TODO: remove suffix when parameters are all optimized
        name_res_gmseg = self.im_target.file_name + '_gmseg' + suffix  # TODO: remove suffix when parameters are all optimized
        ext = self.im_target.ext

        if len(self.target_seg_methods.target_slices) == 1: # if target is 2D (1 SLICE)
            self.res_wm_seg = Image(param=np.asarray(self.target_seg_methods.target_slices[0].wm_seg[0]), absolutepath=name_res_wmseg + ext)
            self.res_gm_seg = Image(param=np.asarray(self.target_seg_methods.target_slices[0].gm_seg[0]), absolutepath=name_res_gmseg + ext)
        else:
            self.res_wm_seg = Image(param=np.asarray([target_slice.wm_seg[0] for target_slice in self.target_seg_methods.target_slices]), absolutepath=name_res_wmseg + ext)
            self.res_gm_seg = Image(param=np.asarray([target_slice.gm_seg[0] for target_slice in self.target_seg_methods.target_slices]), absolutepath=name_res_gmseg + ext)

        self.res_wm_seg.hdr = self.original_hdr
        self.res_wm_seg.file_name = name_res_wmseg
        self.res_wm_seg.save(type='minimize')

        self.res_gm_seg.hdr = self.original_hdr
        self.res_gm_seg.file_name = name_res_gmseg
        self.res_gm_seg.save(type='minimize')


    def show(self):

        sct.printv('\nShowing the pca modes ...')
        self.model.pca.show_all_modes()

        sct.printv('\nPloting the projected dictionary ...')
        self.target_seg_methods.plot_projected_dic(nb_modes=3)

        sct.printv('\nShowing PCA mode graphs ...')
        self.model.pca.show_mode_variation()


########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################

if __name__ == "__main__":
    model_param = ModelParam()
    seg_param = SegmentationParam()
    input_target_fname = None
    input_level_fname = None
    if seg_param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_input = model_param.path_model + "/errsm_34.nii.gz"
        fname_input = model_param.path_model + "/errsm_34_seg_in.nii.gz"
    else:
        param_default = SegmentationParam()

        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Classes for the segmentation of the white/gray matter on a T2star or MT image\n'
                                     'Multi-Atlas based method: the model containing a template of the white/gray matter segmentation along the cervical spinal cord, and a PCA space to describe the variability of intensity in that template is provided in the toolbox. ')
        parser.add_option(name="-i",
                          type_value="file",
                          description="target image to segment"
                                      "if -i isn't used, only the model is computed/loaded",
                          mandatory=False,
                          example='t2star.nii.gz')
        parser.add_option(name="-ofolder",
                          type_value="str",
                          description="output name for the results",
                          mandatory=False,
                          example='t2star_res.nii.gz')
        parser.add_option(name="-model",
                          type_value="folder",
                          description="Path to the dictionary of images",
                          mandatory=False,
                          example='/home/jdoe/data/dictionary')
        parser.add_option(name="-todo-model",
                          type_value="multiple_choice",
                          description="Load or compute the model",
                          mandatory=False,
                          example=['load', 'compute'])
        parser.add_option(name="-vert",
                          type_value="str",
                          description="Image containing level labels for the target or str indicating the level",
                          mandatory=False,
                          example='MNI-Poly-AMU_level_IRP.nii.gz')
        parser.add_option(name="-reg",
                          type_value=[[','], 'str'],
                          description="list of transformations to apply to co-register the dictionary data",
                          mandatory=False,
                          default_value=['Affine'],
                          example=['SyN'])
        parser.add_option(name="-weight",
                          type_value='float',
                          description="weight parameter on the level differences to compute the similarities (beta)",
                          mandatory=False,
                          default_value=2.5,
                          example=2.0)
        parser.add_option(name="-use-levels",
                          type_value='multiple_choice',
                          description="Use the level information as integers or float numbers for the model or not",
                          mandatory=False,
                          default_value='int',
                          example=['0', 'int', 'float'])
        parser.add_option(name="-model-normalization",
                          type_value='multiple_choice',
                          description="1: Normalize the nitensity in all the model slices (when computing the model), 0: no ",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1'])
        parser.add_option(name="-denoising",
                          type_value='multiple_choice',
                          description="1: Adaptative denoising from F. Coupe algorithm, 0: no  WARNING: It affects the model you should use (if denoising is applied to the target, the model should have been coputed with denoising too)",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-normalize",
                          type_value='multiple_choice',
                          description="1: Normalization of the target image's intensity using mean intensity values of the WM and the GM",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-means",
                          type_value=[[','], 'float'],
                          description="Mean intensity values in the target white matter and gray matter (separated by a comma without white space)\n"
                                      "If not specified, the mean intensity values of the target WM and GM  are estimated automatically using the dictionary average segmentation by level.\n"
                                      "Only if the -normalize flag is used",
                          mandatory=False,
                          default_value=None,
                          example=["450,540"])
        parser.add_option(name="-res-type",
                          type_value='multiple_choice',
                          description="Type of result segmentation : binary or probabilistic",
                          mandatory=False,
                          default_value='prob',
                          example=['binary', 'prob'])
        parser.add_option(name="-v",
                          type_value='multiple_choice',
                          description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1', '2'])

        arguments = parser.parse(sys.argv[1:])

        if "-i" in arguments:
            input_target_fname = arguments["-i"]
        if "-ofolder" in arguments:
            seg_param.output_path = arguments["-ofolder"]
        if "-model" in arguments:
            model_param.path_model = arguments["-model"]
        if "-todo-model" in arguments:
            model_param.todo_model = arguments["-todo-model"]
        if "-reg" in arguments:
            model_param.reg = arguments["-reg"]
        if "-l" in arguments:
            input_level_fname = arguments["-l"]
        if "-weight" in arguments:
            model_param.weight_gamma = arguments["-weight"]
        if "-use-levels" in arguments:
            model_param.use_levels = arguments["-use-levels"]
        if "-model-normalization" in arguments:
            model_param.model_slices_normalization = bool(int(arguments["-model-normalization"]))
            print bool(int(arguments["-model-normalization"]))
        if "-denoising" in arguments:
            seg_param.target_denoising = bool(int(arguments["-denoising"]))
        if "-normalize" in arguments:
            seg_param.target_normalization = bool(int(arguments["-normalize"]))
        if "-means" in arguments:
            seg_param.target_means = arguments["-means"]
        if "-res-type" in arguments:
            seg_param.res_type = arguments["-res-type"]
        if "-v" in arguments:
            seg_param.verbose = int(arguments["-v"])
            model_param.verbose = int(arguments["-v"])

    model = Model(model_param=model_param)
    if input_target_fname is not None:
        seg_method = SupervisedSegmentationMethod(input_target_fname, input_level_fname, model, gm_seg_param=seg_param)
        if seg_param.verbose == 2:
            seg_method.show()
