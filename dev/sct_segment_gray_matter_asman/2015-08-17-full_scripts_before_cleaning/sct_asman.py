#!/usr/bin/env python
########################################################################################################################
#
# Asman et al. groupwise multi-atlas segmentation method implementation
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
# import matplotlib.pyplot as plt

from msct_pca import PCA
# from msct_image import Image
# from msct_parser import *
from msct_gmseg_utils import *
import sct_utils as sct
import pickle

from math import sqrt
from math import exp
# from math import fabs


class Param:
    def __init__(self):
        self.debug = 0
        self.path_dictionary = None  # '/Volumes/folder_shared/greymattersegmentation/data_asman/dictionary'
        self.todo_model = None  # 'compute'
        self.model_dir = './gm_seg_model_data'
        self.reg = ['Affine']  # default is Affine  TODO : REMOVE THAT PARAM WHEN REGISTRATION IS OPTIMIZED
        self.reg_metric = 'MI'
        self.target_denoising = True
        self.first_reg = False
        self.use_levels = True
        self.weight_gamma = 1.2
        self.equation_id = 1
        self.weight_label_fusion = False
        self.mode_weight_similarity = False
        self.z_regularisation = False
        self.res_type = 'binary'
        self.verbose = 1

    def __repr__(self):
        s = ''
        s += 'path_dictionary: ' + str(self.path_dictionary) + '\n'
        s += 'todo_model: ' + str(self.todo_model) + '\n'
        s += 'model_dir: ' + str(self.model_dir) + '\n'
        s += 'reg: ' + str(self.reg) + '\n'
        s += 'reg_metric: ' + str(self.reg_metric) + '\n'
        s += 'target_denoising: ' + str(self.target_denoising) + ' ***WARNING: used in sct_segment_gray_matter not in sct_asman***\n'
        s += 'first_reg: ' + str(self.first_reg) + '\n'
        s += 'use_levels: ' + str(self.use_levels) + '\n'
        s += 'weight_gamma: ' + str(self.weight_gamma) + '\n'
        s += 'equation_id: ' + str(self.equation_id) + '\n'
        s += 'weight_label_fusion: ' + str(self.weight_label_fusion) + '\n'
        s += 'mode_weight_similarity: ' + str(self.mode_weight_similarity) + '\n'
        s += 'z_regularisation: ' + str(self.z_regularisation) + '\n'
        s += 'res_type: ' + str(self.res_type) + '\n'
        s += 'verbose: ' + str(self.verbose) + '\n'

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
            self.param = Param()
        else:
            self.param = dic_param

        self.level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}

        # Initialisation of the parameters
        self.coregistration_transfos = None
        self.slices = None
        self.J = None
        self.N = None
        self.mean_seg = None
        self.mean_image = None

        # list of transformation to apply to each slice to co-register the data into the common groupwise space
        self.coregistration_transfos = self.param.reg

        if self.param.todo_model == 'compute':
            self.compute_model()
        elif self.param.todo_model == 'load':
            self.load_model()

    # ------------------------------------------------------------------------------------------------------------------
    # FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def compute_model(self):

        sct.printv('\nComputing the model dictionary ...', self.param.verbose, 'normal')
        sct.run('mkdir ' + self.param.model_dir)
        param_fic = open(self.param.model_dir + '/info.txt', 'w')
        param_fic.write(str(self.param))
        param_fic.close()
        sct.printv('\nLoading data dictionary ...', self.param.verbose, 'normal')
        # List of T2star images (im) and their label decision (gmseg) (=segmentation of the gray matter), slice by slice
        self.slices = self.load_data_dictionary()  # type: list of slices

        # number of slices in the data set
        self.J = len([dic_slice.im for dic_slice in self.slices])  # type: int
        # dimension of the slices (flattened)
        self.N = len(self.slices[0].im.flatten())  # type: int

        # inverts the segmentation slices : the model uses segmentation of the WM instead of segmentation of the GM
        self.invert_seg()

        sct.printv('\nComputing the transformation to co-register all the data into a common groupwise space (using the white matter segmentations) ...', self.param.verbose, 'normal')
        # mean segmentation image of the dictionary, type: numpy array
        self.mean_seg = self.seg_coregistration(transfo_to_apply=self.coregistration_transfos)

        sct.printv('\nCo-registering all the data into the common groupwise space ...', self.param.verbose, 'normal')
        self.coregister_data(transfo_to_apply=self.coregistration_transfos)

        self.mean_image = self.compute_mean_dic_image(np.asarray([dic_slice.im_M for dic_slice in self.slices])) # type: numpy array

        self.save_model()

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
        j = 0
        # TODO: change the name of files to find to a more general structure
        for subject_dir in os.listdir(self.param.path_dictionary):
            subject_path = self.param.path_dictionary + '/' + subject_dir
            if os.path.isdir(subject_path):
                for file_name in os.listdir(subject_path):
                    if 'im' in file_name:  # or 'seg_in' in file_name:
                        slice_level = 0
                        name_list = file_name.split('_')
                        for word in name_list:
                            if word.upper() in self.level_label.values():
                                slice_level = get_key_from_val(self.level_label, word.upper())

                        slices.append(Slice(slice_id=j, im=Image(subject_path + '/' + file_name).data, level=slice_level, reg_to_m=[]))

                        seg_file = sct.extract_fname(file_name)[1][:-3] + '_seg.nii.gz'
                        slices[j].set(gm_seg=Image(subject_path + '/' + seg_file).data)
                        j += 1

        return np.asarray(slices)

    # ------------------------------------------------------------------------------------------------------------------
    def invert_seg(self):
        """
        Invert the gray matter segmentation to get segmentation of the white matter instead
        keeps more information, better results
        """
        for dic_slice in self.slices:
            im_dic = Image(param=dic_slice.im)
            sc = im_dic.copy()
            nz_coord_sc = sc.getNonZeroCoordinates()
            im_seg = Image(param=dic_slice.gm_seg)
            '''
            nz_coord_d = im_seg.getNonZeroCoordinates()
            for coord in nz_coord_sc:
                sc.data[coord.x, coord.y] = 1
            for coord in nz_coord_d:
                im_seg.data[coord.x, coord.y] = 1
            # cast of the -1 values (-> GM pixel at the exterior of the SC pixels) to +1 --> WM pixel
            inverted_slice_decision = np.absolute(sc.data - im_seg.data).astype(int)
            '''
            inverted_slice_decision = inverse_gmseg_to_wmseg(im_seg, im_dic, save=False)
            dic_slice.set(wm_seg=inverted_slice_decision.data)

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

        current_mean_seg = compute_majority_vote_mean_seg(np.asarray([dic_slice.wm_seg for dic_slice in self.slices]))
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

        # Coregistraion of the white matter segmentations
        for dic_slice in self.slices:
            name_j_transform = 'transform_slice_' + str(dic_slice.id) + find_ants_transfo_name(transfo_type)[0]
            new_reg_list = dic_slice.reg_to_M.append(name_j_transform)
            dic_slice.set(reg_to_m=new_reg_list)

            if first:
                seg_m = apply_ants_transfo(mean_seg, dic_slice.wm_seg,  transfo_name=name_j_transform, path=self.param.model_dir + '/', transfo_type=transfo_type, metric=self.param.reg_metric)
            else:
                seg_m = apply_ants_transfo(mean_seg, dic_slice.wm_seg_M,  transfo_name=name_j_transform, path=self.param.model_dir + '/', transfo_type=transfo_type, metric=self.param.reg_metric)
            dic_slice.set(wm_seg_m=seg_m.astype(int))
            dic_slice.set(wm_seg_m_flat=seg_m.flatten().astype(int))

        mean_seg = compute_majority_vote_mean_seg([dic_slice.wm_seg_M for dic_slice in self.slices])

        return mean_seg

    # ------------------------------------------------------------------------------------------------------------------
    def compute_mean_dic_image(self, im_data_set):
        """
        Compute the mean image of the dictionary

        Used to co-register the dictionary images into teh common groupwise space

        :param im_data_set:
        :return mean: mean image of the input data set
        """
        mean = np.sum(im_data_set, axis=0)

        mean /= float(len(im_data_set))

        return mean

    # ------------------------------------------------------------------------------------------------------------------
    def coregister_data(self,  transfo_to_apply=None):
        """
        Apply to each image slice of the dictionary the transformations found registering the segmentation slices.
        The co_registered images are saved for each slice as im_M

        Delete the directories containing the transformation matrix : not needed after the coregistration of the data.

        :param transfo_to_apply: list of string
        :return:
        """
        list_im = [dic_slice.im for dic_slice in self.slices]
        list_gm_seg = [dic_slice.gm_seg for dic_slice in self.slices]

        for dic_slice in self.slices:
            for n_transfo, transfo in enumerate(transfo_to_apply):
                im_m = apply_ants_transfo(self.compute_mean_dic_image(list_im), dic_slice.im, search_reg=False, transfo_name=dic_slice.reg_to_M[n_transfo], binary=False, path=self.param.model_dir+'/', transfo_type=transfo, metric=self.param.reg_metric)
                gm_seg_m = apply_ants_transfo(compute_majority_vote_mean_seg(list_gm_seg), dic_slice.gm_seg, search_reg=False, transfo_name=dic_slice.reg_to_M[n_transfo], binary=True, path=self.param.model_dir+'/', transfo_type=transfo, metric=self.param.reg_metric)
                # apply_2D_rigid_transformation(self.im[j], self.RM[j]['tx'], self.RM[j]['ty'], self.RM[j]['theta'])

            dic_slice.set(im_m=im_m)
            dic_slice.set(gm_seg_m=gm_seg_m)
            dic_slice.set(im_m_flat=im_m.flatten())

        # Delete the directory containing the transformations : They are not needed anymore
        for transfo_type in transfo_to_apply:
            transfo_dir = transfo_type.lower() + '_transformations'
            if transfo_dir in os.listdir(self.param.model_dir + '/'):
                sct.run('rm -rf ' + self.param.model_dir + '/' + transfo_dir + '/')

    # ------------------------------------------------------------------------------------------------------------------
    def save_model(self):
        model_slices = np.asarray([(dic_slice.im_M, dic_slice.wm_seg_M, dic_slice.gm_seg_M, dic_slice.level) for dic_slice in self.slices])
        pickle.dump(model_slices, open(self.param.model_dir + '/dictionary_slices.pkl', 'wb'), protocol=2)

    # ------------------------------------------------------------------------------------------------------------------
    # END OF FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def load_model(self):

        model_slices = pickle.load(open(self.param.path_dictionary + '/dictionary_slices.pkl', 'rb'))

        self.slices = [Slice(slice_id=i_slice, level=dic_slice[3], im_m=dic_slice[0], wm_seg_m=dic_slice[1], gm_seg_m=dic_slice[2], im_m_flat=dic_slice[0].flatten(),  wm_seg_m_flat=dic_slice[1].flatten()) for i_slice, dic_slice in enumerate(model_slices)]  # type: list of slices

        # number of slices in the data set
        self.J = len([dic_slice.im_M for dic_slice in self.slices])  # type: int
        # dimension of the slices (flattened)
        self.N = len(self.slices[0].im_M_flat)  # type: int

        self.mean_seg = compute_majority_vote_mean_seg([dic_slice.wm_seg_M for dic_slice in self.slices])

    # ------------------------------------------------------------------------------------------------------------------
    def show_dictionary_data(self):
        """
        show the 10 first slices of the model dictionary
        """
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

            if self.mean_seg is not None:
                mean_seg_subplot = fig.add_subplot(2, 3, 3)
                mean_seg_subplot.set_title('Mean seg')
                im_mean_seg = mean_seg_subplot.imshow(np.asarray(self.mean_seg))
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
    def __init__(self, model_param=None, k=0.8):
        """
        Model constructor

        :param model_param: model parameters, type: Param

        :param k: Amount of variability to keep in the PCA reduced space, type: float
        """
        if model_param is None:
            self.param = Param()
        else:
            self.param = model_param

        # Model dictionary
        self.dictionary = ModelDictionary(dic_param=self.param)

        sct.printv("The shape of the dictionary used for the PCA is (" + str(self.dictionary.N) + "," + str(self.dictionary.J) + ")", verbose=self.param.verbose)

        # Instantiate a PCA object given the dictionary just build
        if self.param.todo_model == 'compute':
            sct.printv('\nCreating a reduced common space (using a PCA) ...', self.param.verbose, 'normal')
            self.pca = PCA(np.asarray(self.dictionary.slices), k=k)
            self.pca.save_data(self.param.model_dir)
        elif self.param.todo_model == 'load':
            sct.printv('\nLoading a reduced common space (using a PCA) ...', self.param.verbose, 'normal')
            pca_data = pickle.load(open(self.param.path_dictionary + '/pca_data.pkl', 'rb'))
            self.pca = PCA(np.asarray(self.dictionary.slices), mean_vect=pca_data[0], eig_pairs=pca_data[1], k=k)

        # updating the dictionary mean_image
        self.dictionary.mean_image = self.pca.mean_image

        # Other model parameters
        self.epsilon = round(1.0/self.dictionary.J, 4)/2

        if self.param.todo_model == 'compute':
            self.tau = self.compute_tau()
            pickle.dump(self.tau, open(self.param.model_dir + '/tau.txt', 'w'), protocol=0)  # or protocol=2 and 'wb'
        elif self.param.todo_model == 'load':
            self.tau = pickle.load(open(self.param.path_dictionary + '/tau.txt', 'r'))  # if protocol was 2 : 'rb'

        if self.param.verbose == 2:
            self.pca.plot_projected_dic()

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

                    if target_levels is not None and target_levels is not [None] and self.param.use_levels:
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
                if target_levels is not None and self.param.use_levels:
                    '''
                    if target_levels == dataset_levels[j_slice]:
                        beta.append(exp(tau*square_norm))
                    else:
                        beta.append(exp(-tau*square_norm)/self.param.weight_gamma*abs(target_levels - dataset_levels[j_slice]))
                    '''
                    beta.append(exp(-self.param.weight_gamma*abs(target_levels - dataset_levels[j_slice]))*exp(-tau*square_norm) )#TODO: before = no absolute
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
                   '(Tau is a weighting parameter indicating the decay constant associated with a geodesic distance between a given atlas and a projected target image, see Asman paper, eq (16))', 1, 'normal')
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
                if self.param.use_levels:
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

                sum_norm += l0_norm(dic_slice.wm_seg_M, est_segm_j.data)

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
                if beta is None:
                    n_selected_dic_slices = wm_segmentation_slices[selected_ind_by_slice].shape[0]
                    weights = [1.0/n_selected_dic_slices] * n_selected_dic_slices
                else:
                    weights = beta[i][selected_ind_by_slice]
                    weights = [w/sum(weights) for w in weights]
                wm_slice_seg = compute_majority_vote_mean_seg(wm_segmentation_slices[selected_ind_by_slice], weights=weights, type=type, threshold=0.50001)
                res_wm_seg_model_space.append(wm_slice_seg)
                target[i].set(wm_seg_m=wm_slice_seg)

                gm_slice_seg = compute_majority_vote_mean_seg(gm_segmentation_slices[selected_ind_by_slice], weights=weights, type=type)
                res_gm_seg_model_space.append(gm_slice_seg)
                target[i].set(gm_seg_m=gm_slice_seg)

        else:
            # 2D image
            if beta is None:
                n_selected_dic_slices = wm_segmentation_slices[selected_index].shape[0]
                weights = [1.0/n_selected_dic_slices] * n_selected_dic_slices
            else:
                weights = beta[selected_index]
                weights = [w/sum(weights) for w in weights]
            res_wm_seg_model_space = compute_majority_vote_mean_seg(wm_segmentation_slices[selected_index], weights=weights, type=type, threshold=0.50001)
            res_gm_seg_model_space = compute_majority_vote_mean_seg(gm_segmentation_slices[selected_index], weights=weights, type=type)

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
    def __init__(self, model, target_image=None, levels_image=None, epsilon=None):
        """
        Target gray matter segmentation constructor

        :param model: Model used to compute the segmentation, type: Model

        :param target_image: Target image to segment gray matter on, type: Image

        """
        self.model = model

        # Initialization of the target image
        if len(target_image.data.shape) == 3:
            self.target = [Slice(slice_id=i_slice, im=target_slice, reg_to_m=[]) for i_slice, target_slice in enumerate(target_image.data)]
            self.target_dim = 3
        elif len(target_image.data.shape) == 2:
            self.target = [Slice(slice_id=0, im=target_image.data, reg_to_m=[])]
            self.target_dim = 2

        if levels_image is not None and self.model.param.use_levels:
            self.load_level(levels_image)

        if self.model.param.first_reg:
            self.first_reg()

        self.target_pairwise_registration()

        sct.printv('\nProjecting the target image in the reduced common space ...', model.param.verbose, 'normal')
        # coord_projected_target is a list of all the coord of the target's projected slices
        self.coord_projected_target = model.pca.project([target_slice.im_M for target_slice in self.target])

        sct.printv('\nComputing the similarities between the target and the model slices ...', model.param.verbose, 'normal')
        if levels_image is not None and self.model.param.use_levels:
            self.beta = self.model.compute_beta(self.coord_projected_target, target_levels=np.asarray([target_slice.level for target_slice in self.target]), tau=self.model.tau)
        else:
            self.beta = self.model.compute_beta(self.coord_projected_target, tau=self.model.tau)

        sct.printv('\nSelecting the dictionary slices most similar to the target ...', model.param.verbose, 'normal')
        self.selected_k_slices = self.model.select_k_slices(self.beta)
        self.save_selected_slices(target_image.file_name[:-3])

        if self.model.param.verbose == 2:
            self.plot_projected_dic(nb_modes=3, to_highlight='all')  # , to_highlight=(6, self.selected_k_slices[6]))

        sct.printv('\nComputing the result gray matter segmentation ...', model.param.verbose, 'normal')
        if self.model.param.weight_label_fusion:
            use_beta = self.beta
        else:
            use_beta = None
        self.model.label_fusion(self.target, self.selected_k_slices, beta=use_beta, type=self.model.param.res_type)

        if self.model.param.z_regularisation:
            sct.printv('\nRegularisation of the segmentation along the Z axis ...', model.param.verbose, 'normal')
            self.z_regularisation_2d_iteration()

        sct.printv('\nRegistering the result gray matter segmentation back into the target original space...',
                   model.param.verbose, 'normal')
        self.target_pairwise_registration(inverse=True)

    # ------------------------------------------------------------------------------------------------------------------
    def load_level(self, level_image):
        """
        Find the vertebral level of the target image slice(s) for a level image (or a string if the target is 2D)
        :param level_image: image (or a string if the target is 2D) containing level information
        :return None: the target level is set in the function
        """
        if isinstance(level_image, Image):
            nz_coord = level_image.getNonZeroCoordinates()
            for i_level_slice, level_slice in enumerate(level_image.data):
                nz_val = []
                for coord in nz_coord:
                    if coord.x == i_level_slice:
                        nz_val.append(level_slice[coord.y, coord.z])
                try:
                    self.target[i_level_slice].set(level=int(round(sum(nz_val)/len(nz_val))))
                except ZeroDivisionError:
                            sct.printv('WARNING: No level label for slice ' + str(i_level_slice) + ' of target', self.model.param.verbose, 'warning')
                            self.target[i_level_slice].set(level=0)
        elif isinstance(level_image, str):
            self.target[0].set(level=get_key_from_val(self.model.dictionary.level_label, level_image.upper()))

    # ------------------------------------------------------------------------------------------------------------------
    def first_reg(self):
        """
        Do a registration of rhe target image on the mean spinal cord segmentation to hhelp the target registration
        WARNING: DOESN'T IMPROVE THE GM SEGMENTATION RESULT

        :return None: the target moved image is set in the function
        """
        mean_sc_seg = (np.asarray(self.model.pca.mean_image) > 0).astype(int)
        save_image(self.model.pca.mean_image, 'mean_image')
        for i, target_slice in enumerate(self.target):
            moving_target_seg = (np.asarray(target_slice.im) > 0).astype(int)
            transfo = 'BSplineSyN'
            transfo_name = transfo + '_first_reg_slice_' + str(i) + find_ants_transfo_name(transfo)[0]

            apply_ants_transfo(mean_sc_seg, moving_target_seg, binary=True, apply_transfo=False, transfo_type=transfo, transfo_name=transfo_name, metric=self.model.param.reg_metric)
            moved_target_slice = apply_ants_transfo(mean_sc_seg, target_slice.im, binary=False, search_reg=False, transfo_type=transfo, transfo_name=transfo_name, metric=self.model.param.reg_metric)

            target_slice.set(im_m=moved_target_slice)
            target_slice.reg_to_M.append((transfo, transfo_name))

            save_image(target_slice.im, 'slice' + str(target_slice.id) + '_original_im')
            save_image(target_slice.im_M, 'slice' + str(target_slice.id) + '_moved_im')

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
            for i, target_slice in enumerate(self.target):
                if not self.model.param.first_reg:
                    moving_target_slice = target_slice.im
                else:
                    moving_target_slice = target_slice.im_M
                for transfo in self.model.dictionary.coregistration_transfos:
                    transfo_name = transfo + '_transfo_target2model_space_slice_' + str(i) + find_ants_transfo_name(transfo)[0]
                    target_slice.reg_to_M.append((transfo, transfo_name))

                    moving_target_slice = apply_ants_transfo(mean_dic_im, moving_target_slice, binary=False, transfo_type=transfo, transfo_name=transfo_name, metric=self.model.param.reg_metric)
                self.target[i].set(im_m=moving_target_slice)

        else:
            # Inverse registration result in model space --> target original space
            for i, target_slice in enumerate(self.target):
                moving_wm_seg_slice = target_slice.wm_seg_M
                moving_gm_seg_slice = target_slice.gm_seg_M

                for transfo in target_slice.reg_to_M:
                    if self.model.param.res_type == 'binary':
                        bin = True
                    else:
                        bin = False
                    moving_wm_seg_slice = apply_ants_transfo(self.model.dictionary.mean_seg, moving_wm_seg_slice, search_reg=False, binary=bin, inverse=1, transfo_type=transfo[0], transfo_name=transfo[1], metric=self.model.param.reg_metric)
                    moving_gm_seg_slice = apply_ants_transfo(self.model.dictionary.mean_seg, moving_gm_seg_slice, search_reg=False, binary=bin, inverse=1, transfo_type=transfo[0], transfo_name=transfo[1], metric=self.model.param.reg_metric)

                target_slice.set(wm_seg=moving_wm_seg_slice)
                target_slice.set(gm_seg=moving_gm_seg_slice)

    # ------------------------------------------------------------------------------------------------------------------
    def z_regularisation_2d_iteration(self, coeff=0.4):
        """
        Z regularisation option WARNING: DOESN'T IMPROVE THE GM SEGMENTATION RESULT
        Use the result segmentation of the first iteration, the segmentation of slice i is the weighted average of the segmentations of slices i-1 and i+1 and the segmentation of slice i
        :param coeff: weight on each adjacent slice
        :return:
        """
        for i, target_slice in enumerate(self.target[1:-1]):
            adjacent_wm_seg = []  # coeff * self.target[i-1].wm_seg_M, (1-2*coeff) * target_slice.wm_seg_M, coeff * self.target[i+1].wm_seg_M
            adjacent_gm_seg = []  # coeff * self.target[i-1].gm_seg_M, (1-2*coeff) * target_slice.gm_seg_M, coeff * self.target[i+1].gm_seg_M

            precision = 100
            print int(precision*coeff)
            for k in range(int(precision*coeff)):
                adjacent_wm_seg.append(self.target[i-1].wm_seg_M)
                adjacent_wm_seg.append(self.target[i+1].wm_seg_M)
                adjacent_gm_seg.append(self.target[i-1].gm_seg_M)
                adjacent_gm_seg.append(self.target[i+1].gm_seg_M)

            for k in range(precision - 2*int(precision*coeff)):
                adjacent_wm_seg.append(target_slice.wm_seg_M)
                adjacent_gm_seg.append(target_slice.gm_seg_M)

            adjacent_wm_seg = np.asarray(adjacent_wm_seg)
            adjacent_gm_seg = np.asarray(adjacent_gm_seg)

            new_wm_seg = compute_majority_vote_mean_seg(adjacent_wm_seg, type=self.model.param.res_type, threshold=0.50001)
            new_gm_seg = compute_majority_vote_mean_seg(adjacent_gm_seg, type=self.model.param.res_type)  # , threshold=0.4999)

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
        self.model.pca.plot_projected_dic(nb_modes=nb_modes, target_coord=self.coord_projected_target, target_levels=[t_slice.level for t_slice in self.target]) if self.coord_projected_target is not None \
            else self.model.pca.plot_projected_dic(nb_modes=nb_modes)

        if to_highlight == 'all':
            for i in range(len(self.target)):
                self.model.pca.plot_projected_dic(nb_modes=nb_modes, target_coord=self.coord_projected_target, target_levels=[t_slice.level for t_slice in self.target], to_highlight=(i, self.selected_k_slices[i]))
        else:
            self.model.pca.plot_projected_dic(nb_modes=nb_modes, target_coord=self.coord_projected_target, target_levels=[t_slice.level for t_slice in self.target], to_highlight=(to_highlight, self.selected_k_slices[to_highlight])) if self.coord_projected_target is not None \
            else self.model.pca.plot_projected_dic()

    # ------------------------------------------------------------------------------------------------------------------
    def save_selected_slices(self, target_name):
        slice_levels = np.asarray([(dic_slice.id, self.model.dictionary.level_label[dic_slice.level]) for dic_slice in self.model.dictionary.slices])
        fic_selected_slices = open(target_name + '_selected_slices.txt', 'w')
        if self.target_dim == 2:
            fic_selected_slices.write(str(slice_levels[self.selected_k_slices.reshape(self.model.dictionary.J,)]))
        elif self.target_dim == 3:
            for target_slice in self.target:
                fic_selected_slices.write('slice ' + str(target_slice.id) + ': ' + str(slice_levels[self.selected_k_slices[target_slice.id]]) + '\n')
        fic_selected_slices.close()

# ----------------------------------------------------------------------------------------------------------------------
# GRAY MATTER SEGMENTATION SUPERVISED METHOD ---------------------------------------------------------------------------
class GMsegSupervisedMethod():
    """
    Gray matter segmentation supervised method:

    Load a dictionary (training data set), compute or load a model from this dictionary
sct_Image
    Load a target image to segment and do the segmentation using the model
    """
    def __init__(self, target_fname, level_fname, model, gm_seg_param=None):
        # build the appearance model
        self.model = model

        sct.printv('\nConstructing target image ...', verbose=gm_seg_param.verbose, type='normal')
        # construct target image
        self.target_image = Image(target_fname)
        original_hdr = self.target_image.hdr
        # build a target segmentation
        level_im = None
        if level_fname is not None:
            if len(level_fname) < 3:
                # in this case the level is a string and not an image
                level_im = level_fname
            else:
                level_im = Image(level_fname)

        # TARGET PAIRWISE SEGMENTATION
        if level_im is not None:
            self.target_seg_methods = TargetSegmentationPairwise(self.model, target_image=self.target_image, levels_image=level_im)
        else:
            self.target_seg_methods = TargetSegmentationPairwise(self.model, target_image=self.target_image)

        # get & save the result gray matter segmentation
        suffix = '_'
        suffix += '_' + gm_seg_param.res_type
        for transfo in self.model.dictionary.coregistration_transfos:
            suffix += '_' + transfo
        if self.model.param.use_levels:
            suffix += '_with_levels_' + str(self.model.param.weight_gamma)
        else:
            suffix += '_no_levels'
        if self.model.param.z_regularisation:
            suffix += '_Zregularisation'

        name_res_wmseg = sct.extract_fname(target_fname)[1] + '_res_wmseg' + suffix  # TODO: remove suffix when parameters are all optimized
        name_res_gmseg = sct.extract_fname(target_fname)[1] + '_res_gmseg' + suffix  # TODO: remove suffix when parameters are all optimized

        if len(self.target_seg_methods.target) == 1:
            self.res_wm_seg = Image(param=np.asarray(self.target_seg_methods.target[0].wm_seg), absolutepath=name_res_wmseg + '.nii.gz')
            self.res_gm_seg = Image(param=np.asarray(self.target_seg_methods.target[0].gm_seg), absolutepath=name_res_gmseg + '.nii.gz')
        else:
            self.res_wm_seg = Image(param=np.asarray([target_slice.wm_seg for target_slice in self.target_seg_methods.target]), absolutepath=name_res_wmseg + '.nii.gz')
            self.res_gm_seg = Image(param=np.asarray([target_slice.gm_seg for target_slice in self.target_seg_methods.target]), absolutepath=name_res_gmseg + '.nii.gz')

        self.res_wm_seg.hdr = original_hdr
        self.res_wm_seg.file_name = name_res_wmseg
        self.res_wm_seg.save(type='minimize')

        self.res_gm_seg.hdr = original_hdr
        self.res_gm_seg.file_name = name_res_gmseg
        self.res_gm_seg.save(type='minimize')

        # save_image(self.res_wm_seg.data, name_res_wmseg, hdr=original_hdr)
        # save_image(self.res_gm_seg.data, name_res_gmseg, hdr=original_hdr)

        self.corrected_wm_seg = correct_wmseg(self.res_gm_seg, self.target_image, name_res_wmseg, original_hdr)

    def show(self):

        sct.printv('\nShowing the pca modes ...')
        self.model.pca.show_all_modes()

        sct.printv('\nPloting the projected dictionary ...')
        self.target_seg_methods.plot_projected_dic(nb_modes=3)

        sct.printv('\nShowing PCA mode graphs ...')
        self.model.pca.show_mode_variation()

        # sct.printv('\nShowing the projected target ...')
        # self.target_seg_methods.show_projected_target()



########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################

if __name__ == "__main__":
    param = Param()
    input_target_fname = None
    input_level_fname = None
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_input = param.path_dictionary + "/errsm_34.nii.gz"
        fname_input = param.path_dictionary + "/errsm_34_seg_in.nii.gz"
    else:
        param_default = Param()

        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Project all the input image slices on a PCA generated from set of t2star images')
        parser.add_option(name="-i",
                          type_value="file",
                          description="T2star image you want to segment"
                                      "if -i isn't used, only the model is computed/loaded",
                          mandatory=False,
                          example='t2star.nii.gz')
        parser.add_option(name="-dic",
                          type_value="folder",
                          description="Path to the dictionary of images",
                          mandatory=True,
                          example='/home/jdoe/data/dictionary')
        parser.add_option(name="-model",
                          type_value="multiple_choice",
                          description="Load or compute the model",
                          mandatory=True,
                          example=['load', 'compute'])
        parser.add_option(name="-l",
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
                          default_value=1.2,
                          example=2.0)
        parser.add_option(name="-use-levels",
                          type_value='multiple_choice',
                          description="1: Use vertebral level information, 0: no ",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-z",
                          type_value='multiple_choice',
                          description="1: Z regularisation, 0: no ",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1'])
        parser.add_option(name="-denoising",
                          type_value='multiple_choice',
                          description="1: Adaptative denoising from F. Coupe algorithm, 0: no  WARNING: It affects the model you should use (if denoising is applied to the target, the model should have been coputed with denoising too",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-first-reg",
                          type_value='multiple_choice',
                          description="Apply a Bspline registration using the spinal cord edges target --> model first",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1'])
        parser.add_option(name="-weighted-label-fusion",
                          type_value='multiple_choice',
                          description="Use the similarities as a weights for the label fusion",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1'])
        parser.add_option(name="-weighted-similarity",
                          type_value='multiple_choice',
                          description="Use a PCA mode weighted norm for the computation of the similarities instead of the euclidean square norm",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1'])
        parser.add_option(name="-res-type",
                          type_value='multiple_choice',
                          description="Type of result segmentation : binary or probabilistic",
                          mandatory=False,
                          default_value='binary',
                          example=['binary', 'prob'])
        parser.add_option(name="-v",
                          type_value='multiple_choice',
                          description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                          mandatory=False,
                          default_value=0,
                          example=['0', '1', '2'])

        arguments = parser.parse(sys.argv[1:])
        param.path_dictionary = arguments["-dic"]
        param.todo_model = arguments["-model"]

        if "-i" in arguments:
            input_target_fname = arguments["-i"]
        if "-reg" in arguments:
            param.reg = arguments["-reg"]
        if "-l" in arguments:
            input_level_fname = arguments["-l"]
        if "-weight" in arguments:
            param.weight_gamma = arguments["-weight"]
        if "-use-levels" in arguments:
            param.use_levels = bool(int(arguments["-use-levels"]))
        if "-weighted-label-fusion" in arguments:
            param.weight_label_fusion = bool(int(arguments["-weighted-label-fusion"]))
        if "-weighted-similarity" in arguments:
            param.mode_weight_similarity = bool(int(arguments["-weighted-similarity"]))
        if "-z" in arguments:
            param.z_regularisation = bool(int(arguments["-z"]))
        if "-denoising" in arguments:
            param.target_denoising = bool(int(arguments["-denoising"]))
        if "-first-reg" in arguments:
            param.first_reg = bool(int(arguments["-first-reg"]))
        if "-res-type" in arguments:
            param.res_type = arguments["-res-type"]
        if "-v" in arguments:
            param.verbose = int(arguments["-v"])

    model = Model(model_param=param, k=0.8)
    if input_target_fname is not None:
        gm_seg_method = GMsegSupervisedMethod(input_target_fname, input_level_fname, model, gm_seg_param=param)
        print param.verbose == 2
        if param.verbose == 2:
            gm_seg_method.show()