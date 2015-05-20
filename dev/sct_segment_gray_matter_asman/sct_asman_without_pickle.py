#!/usr/bin/env python
########################################################################################################################
#
# Asman et al. groupwise multi-atlas segmentation method implementation
#
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Sara Dupont
# Modified: 2015-03-24
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

from math import sqrt
from math import exp
# from math import fabs


class Param:
    def __init__(self):
        self.debug = 0
        self.path_dictionary = None  # '/Volumes/folder_shared/greymattersegmentation/data_asman/dictionary'
        self.todo_model = None  # 'compute'
        self.reg = None  # TODO : REMOVE THAT PARAM WHEN REGISTRATION IS OPTIMIZED
        self.target_reg = None  # TODO : REMOVE THAT PARAM WHEN GROUPWISE/PAIR IS OPTIMIZED
        self.level_fname = None
        #  self.seg_type = None
        self.verbose = 1


########################################################################################################################
# ----------------------------------------------------- Classes ------------------------------------------------------ #
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# MODEL DICTIONARY SLICE BY SLICE---------------------------------------------------------------------------------------
class ModelDictionaryBySlice:
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

        self.level_label = {0: '', 1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7', 8: 'T1', 9: 'T2',
                            10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6'}

        # list of the slices of the dictionary
        self.slices = []  # type: list of slices
        # number of slices
        self.J = 0  # type: int
        # dimension of the slices (flattened)
        self.N = 0  # type: int
        # mean segmentation image of the dictionary
        self.mean_seg = None  # type: numpy array
        # mean image of the dictionary
        self.mean_image = None  # type: numpy array

        # dictionary containing information about the data dictionary slices by subject : used to save the model only
        self.dic_data_info = {}  # type: dictionary

        # list of transformation to apply to each slice to co-register the data into the common groupwise space
        if self.param.reg is not None:
            self.coregistration_transfos = self.param.reg
        else:
            self.coregistration_transfos = ['Affine']

        suffix = ''
        for transfo in self.coregistration_transfos:
            suffix += '_' + transfo

        # folder containing the saved model
        self.model_dic_name = ''
        if self.param.todo_model == 'compute':
            self.model_dic_name = './gmseg_model_dictionary' + suffix  # TODO: remove suffix when reg is optimized
            self.compute_model_dictionary()
        elif self.param.todo_model == 'load':
            self.model_dic_name = self.param.path_dictionary  # TODO change the path by the name of the dic ?? ...
            self.load_model_dictionary()
        else:
            sct.printv('WARNING: no todo_model param', self.param.verbose, 'warning')

    # ------------------------------------------------------------------------------------------------------------------
    def compute_model_dictionary(self):
        """
        Compute the model dictionary using the provided data set
        """
        sct.printv('\nComputing the model dictionary ...', self.param.verbose, 'normal')
        # Load all the images' slices from param.path_dictionary
        sct.printv('\nLoading data dictionary ...', self.param.verbose, 'normal')
        # List of T2star images (im) and their label decision (seg) (=segmentation of the gray matter), slice by slice

        sct.run('mkdir ' + self.model_dic_name)

        self.slices = self.load_data_dictionary()

        # number of slices in the data set
        self.J = len([dic_slice.im for dic_slice in self.slices])
        # dimension of the data (flatten slices)
        self.N = len(self.slices[0].im.flatten())

        # inverts the segmentation slices : the model uses segmentation of the WM instead of segmentation of the GM
        self.invert_seg()
        self.save_model_data('inverted_gm_seg')

        sct.printv('\nComputing the transformation to co-register all the data into a common groupwise space ...',
                   self.param.verbose, 'normal')

        self.mean_seg = self.seg_coregistration(transfo_to_apply=self.coregistration_transfos)
        self.save_model_data('wm_seg_M')

        sct.printv('\nCo-registering all the data into the common groupwise space ...', self.param.verbose, 'normal')

        # List of images (im_M) and their label decision (seg_M) (=segmentation of the gray matter),
        #  --> slice by slice in the common groupwise space
        self.coregister_data(transfo_to_apply=self.coregistration_transfos)

        self.save_model_data('im_M')

        self.mean_image = self.compute_mean_dic_image(np.asarray([dic_slice.im_M for dic_slice in self.slices]))
        save_image(self.mean_image, 'mean_image', path=self.model_dic_name+'/', im_type='uint8')

    # ------------------------------------------------------------------------------------------------------------------
    # FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------

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
        total_j_im = 0
        total_j_seg = 0
        # TODO: change the name of files to find to a more general structure
        for subject_dir in os.listdir(self.param.path_dictionary):
            subject_path = self.param.path_dictionary + '/' + subject_dir
            if os.path.isdir(subject_path):
                self.dic_data_info[subject_dir] = {'n_slices': 0, 'inverted_gm_seg': [], 'im_M': [], 'seg_M': [], 'levels': []}
                j_im = 0
                j_seg = 0
                first_slice = total_j_im
                sct.run('mkdir ' + self.model_dic_name + '/' + subject_dir, verbose=self.param.verbose)
                for file_name in os.listdir(subject_path):
                    if 'im' in file_name:
                        slice_level = 0
                        name_list = file_name.split('_')
                        for word in name_list:
                            if word in self.level_label.values():
                                slice_level = get_key_from_val(self.level_label, word)

                        slices.append(Slice(slice_id=total_j_im, im=Image(subject_path + '/' + file_name).data, level=slice_level, reg_to_m=[]))
                        self.dic_data_info[subject_dir]['levels'].append(slice_level)

                        # copy of the slice image in the saved model folder
                        if j_im < 10:
                            j_im_str = str(j_im)
                            j_im_str = '0' + j_im_str
                        else:
                            j_im_str = str(j_im)

                        im_file_name = subject_dir + '_slice' + j_im_str + '_' + self.level_label[slice_level] + '_im.nii.gz'
                        sct.run('cp ./' + self.param.path_dictionary + '/' + subject_dir + '/' + file_name + ' ' + self.model_dic_name + '/' + subject_dir + '/' + im_file_name)
                        j_im += 1
                        total_j_im += 1

                    if 'seg' in file_name:
                        slices[total_j_seg].set(gm_seg=Image(subject_path + '/' + file_name).data)
                        j_seg += 1
                        total_j_seg += 1

                if j_im == j_seg:
                    self.dic_data_info[subject_dir]['n_slices'] = (first_slice, total_j_im)
                else:
                    sct.printv('ERROR: subject ' + subject_dir + ' doesn\'t have the same number of slice images and segmentations', verbose=self.param.verbose, type='error')

        return np.asarray(slices)

    # ------------------------------------------------------------------------------------------------------------------
    def save_model_data(self, what_to_save):
        """
        save 3D images of the model dictionary using the dictionary of information about the data slices by subject

        :param what_to_save: type of data to be saved
        """
        suffix = ''
        data_to_save = []
        if what_to_save == 'gm_seg':
            suffix = '_gm_seg'
            data_to_save = [dic_slice.gm_seg for dic_slice in self.slices]
        if what_to_save == 'inverted_gm_seg':
            suffix = '_wm_seg'
            data_to_save = [dic_slice.wm_seg for dic_slice in self.slices]
        elif what_to_save == 'im':
            suffix = '_im'
            data_to_save = [dic_slice.im for dic_slice in self.slices]
        elif what_to_save == 'im_M':
            suffix = '_im_model_space'
            data_to_save = [dic_slice.im_M for dic_slice in self.slices]
        elif what_to_save == 'gm_seg_M':
            suffix = '_gm_seg_model_space'
            data_to_save = [dic_slice.gm_seg_M for dic_slice in self.slices]
        elif what_to_save == 'wm_seg_M':
            suffix = '_wm_seg_model_space'
            data_to_save = [dic_slice.wm_seg_M for dic_slice in self.slices]

        for subject_name in sorted(self.dic_data_info.keys()):
            first_subject_slice, last_subject_slice = self.dic_data_info[subject_name]['n_slices']
            self.dic_data_info[subject_name][what_to_save] = data_to_save[first_subject_slice:last_subject_slice]

            for i, slice_i in enumerate(self.dic_data_info[subject_name][what_to_save]):
                to_save = Image(param=np.asarray(slice_i))
                to_save.path = self.model_dic_name + '/' + subject_name + '/'

                if i < 10:
                    i_str = str(i)
                    i_str = '0' + i_str
                else:
                    i_str = str(i)
                to_save.file_name = subject_name + '_slice' + i_str + '_' + self.level_label[self.dic_data_info[subject_name]['levels'][i]] + suffix
                to_save.ext = '.nii.gz'
                to_save.save(type='minimize')

    # ------------------------------------------------------------------------------------------------------------------
    def invert_seg(self):
        """
        Invert the gray matter segmentation to get segmentation of the white matter instead
        keeps more information, theoretically better results
        """
        for dic_slice in self.slices:
            im_dic = Image(param=dic_slice.im)
            sc = im_dic.copy()
            nz_coord_sc = sc.getNonZeroCoordinates()
            im_seg = Image(param=dic_slice.gm_seg)
            nz_coord_d = im_seg.getNonZeroCoordinates()
            for coord in nz_coord_sc:
                sc.data[coord.x, coord.y] = 1
            for coord in nz_coord_d:
                im_seg.data[coord.x, coord.y] = 1
            # cast of the -1 values (-> GM pixel at the exterior of the SC pixels) to +1 --> WM pixel
            inverted_slice_decision = np.absolute(sc.data - im_seg.data).astype(int)
            dic_slice.set(wm_seg=inverted_slice_decision)

    # ------------------------------------------------------------------------------------------------------------------
    def seg_coregistration(self, transfo_to_apply=None):
        """
        For all the segmentation slices, do a registration of the segmentation slice to the mean segmentation
         applying all the transformations in transfo_to_apply

        Compute, apply and save each transformation warping field for all the segmentation slices

        Compute the new mean segmentation at each step and update self.mean_seg

        :param transfo_to_apply: list of string
        :return:
        """

        current_mean_seg = compute_majority_vote_mean_seg(np.asarray([dic_slice.wm_seg for dic_slice in self.slices]))

        for transfo in transfo_to_apply:
            sct.printv('Doing a ' + transfo + ' registration of each segmentation slice to the mean segmentation ...', self.param.verbose, 'normal')
            current_mean_seg = self.find_coregistration(mean_seg=current_mean_seg, transfo_type=transfo)

        resulting_mean_seg = current_mean_seg

        return resulting_mean_seg

    # ------------------------------------------------------------------------------------------------------------------
    def find_coregistration(self, mean_seg=None, transfo_type='Rigid', first=True):
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
                seg_m = apply_ants_transfo(mean_seg, dic_slice.wm_seg,  transfo_name=name_j_transform, path=self.model_dic_name + '/', transfo_type=transfo_type)
            else:
                seg_m = apply_ants_transfo(mean_seg, dic_slice.wm_seg_M,  transfo_name=name_j_transform, path=self.model_dic_name + '/', transfo_type=transfo_type)
            dic_slice.set(wm_seg_m=seg_m.astype(int))
            dic_slice.set(wm_seg_m_flat=seg_m.flatten().astype(int))

        mean_seg = compute_majority_vote_mean_seg([dic_slice.wm_seg_M for dic_slice in self.slices])

        save_image(mean_seg, 'mean_seg', path=self.model_dic_name+'/', im_type='uint8')
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

        :param transfo_to_apply: list of string
        :return:
        """
        list_im = [dic_slice.im for dic_slice in self.slices]

        for dic_slice in self.slices:
            for n_transfo, transfo in enumerate(transfo_to_apply):
                im_m = apply_ants_transfo(self.compute_mean_dic_image(list_im), dic_slice.im, search_reg=False, transfo_name=dic_slice.reg_to_M[n_transfo], binary=False, path=self.model_dic_name+'/', transfo_type=transfo)
                # apply_2D_rigid_transformation(self.im[j], self.RM[j]['tx'], self.RM[j]['ty'], self.RM[j]['theta'])

            dic_slice.set(im_m=im_m)
            dic_slice.set(im_m_flat=im_m.flatten())

    # ------------------------------------------------------------------------------------------------------------------
    # END OF FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------
    # TODO: ADAPT LOAD MODEL TO GMSEG AND WMSEG ATTRIBUTES OF SLCIES
    # ------------------------------------------------------------------------------------------------------------------
    def load_model_dictionary(self):
        """
        Load the model dictionary from a saved one
        """

        sct.printv('\nLoading the model dictionary ...', self.param.verbose, 'normal')

        j_im = 0
        j_seg = 0
        j_im_m = 0
        j_seg_m = 0

        for subject_dir in os.listdir(self.model_dic_name):
            subject_path = self.model_dic_name + '/' + subject_dir
            if os.path.isdir(subject_path) and 'transformations' not in subject_path:

                for file_name in os.listdir(subject_path):
                    if '_im.nii' in file_name:
                        reg_list = ['transform_slice_' + str(j_im) + find_ants_transfo_name(transfo_type)[0]
                                    for transfo_type in self.coregistration_transfos]

                        slice_level = 0
                        name_list = file_name.split('_')
                        for word in name_list:
                            if word in self.level_label.values():
                                slice_level = get_key_from_val(self.level_label, word)

                        self.slices.append(Slice(slice_id=j_im, im=Image(subject_path + '/' + file_name).data,
                                                 level=slice_level, reg_to_m=reg_list))
                        j_im += 1

                    if '_seg.nii' in file_name:
                        self.slices[j_seg].set(gm_seg=Image(subject_path + '/' + file_name).data)
                        j_seg += 1

                    if '_im_model_space.nii' in file_name:
                        im_m_slice = Image(subject_path + '/' + file_name).data
                        self.slices[j_im_m].set(im_m=im_m_slice, im_m_flat=im_m_slice.flatten())
                        j_im_m += 1

                    if '_seg_model_space.nii' in file_name:
                        seg_m_slice = Image(subject_path + '/' + file_name).data
                        self.slices[j_seg_m].set(wm_seg_m=seg_m_slice, wm_seg_m_flat=seg_m_slice.flatten())
                        j_seg_m += 1

        # number of atlases in the dictionary
        self.J = len(self.slices)  # len([slice.im for slice in self.slices])

        # dimension of the data (flatten slices)
        self.N = len(self.slices[0].im_M.flatten())

        self.mean_image = Image(self.model_dic_name + '/mean_image.nii.gz').data
        self.mean_seg = Image(self.model_dic_name + '/mean_seg.nii.gz').data

    # ------------------------------------------------------------------------------------------------------------------
    def show_data(self):
        """
        show the 10 first slices of the model dictionary
        """
        for dic_slice in self.slices[:10]:
            fig = plt.figure()

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

            mean_seg_subplot = fig.add_subplot(2, 3, 3)
            mean_seg_subplot.set_title('Mean seg')
            im_mean_seg = mean_seg_subplot.imshow(np.asarray(self.mean_seg))
            im_mean_seg.set_interpolation('nearest')
            im_mean_seg.set_cmap('gray')

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
    def __init__(self, model_param=None, dictionary=None, k=0.8):
        """
        Model constructor

        :param model_param: model parameters, type: Param

        :param dictionary: type: ModelDictionary

        :param k: Amount of variability to keep in the PCA reduced space, type: float
        """
        if model_param is None:
            if dictionary is not None:
                self.param = dictionary.param
            else:
                self.param = Param()
        else:
            self.param = model_param

        self.dictionary = dictionary

        sct.printv("The shape of the dictionary used for the PCA is "
                   "(" + str(self.dictionary.N) + "," + str(self.dictionary.J) + ")", verbose=self.param.verbose)

        # Instantiate a PCA object given the dictionary just build
        sct.printv('\nCreating a reduced common space (using a PCA) ...', self.param.verbose, 'normal')

        if self.param.todo_model == 'compute':
            # creation of a PCA
            self.pca = PCA(np.asarray(self.dictionary.slices), k=k)
            # updating the dictionary mean_image 
            self.dictionary.mean_image = self.pca.mean_image

            save_image(self.pca.mean_image, 'mean_image', path=self.dictionary.model_dic_name+'/')
            # saving the PCA data into a text file
            self.pca.save_data(self.dictionary.model_dic_name)

        elif self.param.todo_model == 'load':
            # loading PCA data from file
            pca_mean_data, pca_eig_pairs = self.load_pca_file()

            # creation of a PCA from the loaded data
            self.pca = PCA(np.asarray(self.dictionary.slices), mean_vect=pca_mean_data, eig_pairs=pca_eig_pairs, k=k)
        if self.param.verbose == 2:
            self.pca.plot_projected_dic()

    # ----------------------------------------------------------------------------------------------------------------------
    def load_pca_file(self, file_name='data_pca.txt'):
        """
        Load a PCA from a text file containing the appropriate information (previously saved)

        :param file_name: name of the PCA text file
        """
        fic_data_pca = open(self.param.path_dictionary + '/' + file_name, 'r')
        mean_data_list = fic_data_pca.readline().split(',')
        eig_pairs_list = fic_data_pca.readline().split(',')
        fic_data_pca.close()

        mean_data_vect = []
        for val in mean_data_list:
            mean_data_vect.append([float(val)])
        mean_data_vect = np.asarray(mean_data_vect)

        eig_pairs_vect = []
        for pair in eig_pairs_list:
            eig_val_str, eig_vect_str = pair.split(';')
            eig_vect_str = eig_vect_str.split(' ')
            # eig_vect_str[-1] = eig_vect_str[-1][:-1]
            eig_vect = []
            for i, v in enumerate(eig_vect_str):
                if v != '' and v != '\n':
                    eig_vect.append(float(v))
            eig_pairs_vect.append((float(eig_val_str), eig_vect))

        return mean_data_vect, eig_pairs_vect


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
    def __init__(self, model, target_image=None, levels_image=None, tau=None):
        """
        Target gray matter segmentation constructor

        :param model: Model used to compute the segmentation, type: Model

        :param target_image: Target image to segment gray matter on, type: Image

        :param tau: Weighting parameter associated with the geodesic distances in the model dictionary, type: float
        """
        self.model = model

        # Get the target image
        if len(target_image.data.shape) == 3:
            self.target = [Slice(slice_id=i_slice, im=target_slice) for i_slice, target_slice in enumerate(target_image.data)]
            self.target_dim = 3
        elif len(target_image.data.shape) == 2:
            self.target = [Slice(slice_id=0, im=target_image.data)]
            self.target_dim = 2

        # if levels_image is not None:
        if isinstance(levels_image, Image):
            nz_coord = levels_image.getNonZeroCoordinates()
            for i_level_slice, level_slice in enumerate(levels_image.data):
                nz_val = []
                for coord in nz_coord:
                    if coord.x == i_level_slice:
                        nz_val.append(level_slice[coord.y, coord.z])
                try:
                    self.target[i_level_slice].set(level=int(round(sum(nz_val)/len(nz_val))))
                except ZeroDivisionError:
                            sct.printv('No level label for slice ' + str(i_level_slice) + ' of target')
                            self.target[i_level_slice].set(level=0)
        elif isinstance(levels_image, str):
            self.target[0].set(level=get_key_from_val(self.model.dictionary.level_label, levels_image))

        # self.majority_vote_segmentation()

        # self.target_M = self.target_pairwise_registration()
        self.target_pairwise_registration()

        # coord_projected_target is a list of all the coord of the target's projected slices
        sct.printv('\nProjecting the target image in the reduced common space ...', model.param.verbose, 'normal')
        # self.coord_projected_target = model.pca.project(self.target_M) if self.target_M is not None else None
        self.coord_projected_target = model.pca.project([target_slice.im_M for target_slice in self.target])

        self.epsilon = round(1.0/self.model.dictionary.J, 4)/2
        print 'epsilon : ', self.epsilon

        if tau is None:
            self.tau = self.compute_tau()
        else:
            self.tau = tau

        if levels_image is not None:
            self.beta = self.compute_beta(self.coord_projected_target, target_levels=np.asarray([target_slice.level for target_slice in self.target]), tau=self.tau)
        else:
            self.beta = self.compute_beta(self.coord_projected_target, tau=self.tau)

        sct.printv('\nSelecting the dictionary slices most similar to the target ...', model.param.verbose, 'normal')

        self.selected_k_slices = self.select_k_slices(self.beta)

        slice_levels = np.asarray([self.model.dictionary.level_label[dic_slice.level] for dic_slice in self.model.dictionary.slices])
        fic_selected_slices = open('selected_slices.txt', 'w')
        if self.target_dim == 2:
            fic_selected_slices.write(str(slice_levels[self.selected_k_slices.reshape(self.model.dictionary.J,)]))
        elif self.target_dim == 3:
            for target_slice in self.target:
                fic_selected_slices.write('slice ' + str(target_slice.id) + ': ' + str(slice_levels[self.selected_k_slices[target_slice.id]]))
        fic_selected_slices.close()

        sct.printv('\nComputing the result gray matter segmentation ...', model.param.verbose, 'normal')
        # self.target_GM_seg_M = self.label_fusion(self.selected_k_slices)
        self.label_fusion(self.selected_k_slices)
        self.sc_label_fusion(self.selected_k_slices)

        sct.printv('\nRegistering the result gray matter segmentation back into the target original space...',
                   model.param.verbose, 'normal')
        self.target_pairwise_registration(inverse=True)

    # ------------------------------------------------------------------------------------------------------------------
    def target_pairwise_registration(self, inverse=False):
        """
        Register the target image into the model space

        Affine (or rigid + affine) registration of the target on the mean model image --> pairwise

        :param inverse: if True, apply the inverse warping field of the registration target -> model space
        to the result gray matter segmentation of the target
        (put it back in it's original space)
        """
        if not inverse:
            # Registration target --> model space
            mean_dic_im = self.model.pca.mean_image
            for i, target_slice in enumerate(self.target):
                moving_target_slice = target_slice.im
                for transfo in self.model.dictionary.coregistration_transfos:
                    transfo_name = transfo + '_transfo_target2model_space_slice_' + str(i) + find_ants_transfo_name(transfo)[0]

                    moving_target_slice = apply_ants_transfo(mean_dic_im, moving_target_slice, binary=False, transfo_type=transfo, transfo_name=transfo_name)
                self.target[i].set(im_m=moving_target_slice)

        else:
            # Inverse registration result in model space --> target original space
            for i, target_slice in enumerate(self.target):
                moving_seg_slice = target_slice.wm_seg_M
                moving_sc_seg_slice = target_slice.sc_seg

                for transfo in self.model.dictionary.coregistration_transfos:
                    transfo_name = transfo + '_transfo_target2model_space_slice_' + str(i) + find_ants_transfo_name(transfo)[0]
                    moving_seg_slice = apply_ants_transfo(self.model.dictionary.mean_seg, moving_seg_slice, search_reg=False, binary=True, inverse=1, transfo_type=transfo, transfo_name=transfo_name)
                    moving_sc_seg_slice = apply_ants_transfo(self.model.dictionary.mean_seg, moving_sc_seg_slice, search_reg=False, binary=True, inverse=1, transfo_type=transfo, transfo_name=transfo_name)

                target_slice.set(wm_seg=moving_seg_slice)
                target_slice.set(sc_seg=moving_sc_seg_slice)

    # ------------------------------------------------------------------------------------------------------------------
    def compute_beta(self, coord_target, target_levels=None, dataset_coord=None, dataset_levels=None, tau=0.01):
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
            dataset_coord = self.model.pca.dataset_coord.T
            dataset_levels = [dic_slice.level for dic_slice in self.model.dictionary.slices]

        beta = []

        # TODO: SEE IF WE NEED TO CHECK THE SECOND DIMENSION OF COORD TARGET OR THE FIRST ...
        if isinstance(coord_target[0], (list, np.ndarray)):
            for i_target, coord_projected_slice in enumerate(coord_target):
                beta_slice = []
                for j_slice, coord_slice_j in enumerate(dataset_coord):
                    square_norm = np.linalg.norm((coord_projected_slice - coord_slice_j), 2)
                    if target_levels is not None:
                        if target_levels[i_target] == dataset_levels[j_slice]:
                            beta_slice.append(exp(tau*square_norm))
                        else:
                            beta_slice.append(exp(-tau*square_norm)/1.2*(target_levels[i_target] - dataset_levels[j_slice]))
                    else:
                        beta_slice.append(exp(tau*square_norm))

                try:
                    beta_slice /= np.sum(beta_slice)
                except ZeroDivisionError:
                    sct.printv('WARNING : similarities are null', self.model.param.verbose, 'warning')
                    print beta_slice

                beta.append(beta_slice)
        else:
            for j_slice, coord_slice_j in enumerate(dataset_coord):
                square_norm = np.linalg.norm((coord_target - coord_slice_j), 2)
                if target_levels is not None:
                    if target_levels == dataset_levels[j_slice]:
                        beta.append(exp(tau*square_norm))
                    else:
                        beta.append(exp(-tau*square_norm)/1.2*(target_levels - dataset_levels[j_slice]))
                else:
                    beta.append(exp(tau*square_norm))

            try:
                beta /= np.sum(beta)
            except ZeroDivisionError:
                sct.printv('WARNING : similarities are null', self.model.param.verbose, 'warning')
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
            for dic_slice in self.model.dictionary.slices:
                projected_dic_slice_coord = self.model.pca.project_array(dic_slice.im_M_flat)
                coord_dic_slice_dataset = np.delete(self.model.pca.dataset_coord.T, dic_slice.id, 0)
                dic_slice_dataset_levels = np.delete(np.asarray(dic_levels), dic_slice.id, 0)
                beta_dic_slice = self.compute_beta(projected_dic_slice_coord, target_levels=dic_slice.level, dataset_coord=coord_dic_slice_dataset, dataset_levels=dic_slice_dataset_levels, tau=tau)
                kj = self.select_k_slices(beta_dic_slice)
                est_segm_j = self.label_fusion(kj)

                sum_norm += l0_norm(dic_slice.wm_seg_M, est_segm_j.data)

            return sum_norm

        dic_levels = [dic_slice.level for dic_slice in self.model.dictionary.slices]

        est_tau = minimize(to_minimize, 0.001, method='Nelder-Mead', options={'xtol': 0.0005})
        sct.printv('Estimated tau : ' + str(est_tau.x[0]))
        if self.model.param.todo_model == 'compute':
            fic = open(self.model.dictionary.model_dic_name + '/tau.txt', 'w')
            fic.write(str(est_tau.x[0]))
            fic.close()
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
    def label_fusion(self, selected_index):
        """
        Compute the resulting segmentation by label fusion of the segmentation of the selected dictionary slices

        :param selected_index: array of indexes (as a boolean array) of the selected dictionary slices

        :return res_seg_model_space: Image of the resulting segmentation for the target image (in the model space)
        """
        segmentation_slices = np.asarray([dic_slice.wm_seg_M for dic_slice in self.model.dictionary.slices])

        res_seg_model_space = []

        if isinstance(selected_index[0], (list, np.ndarray)):

            for i, selected_ind_by_slice in enumerate(selected_index):  # selected_slices:
                slice_seg = compute_majority_vote_mean_seg(segmentation_slices[selected_ind_by_slice])
                res_seg_model_space.append(slice_seg)
                self.target[i].set(wm_seg_m=slice_seg)

        else:
            res_seg_model_space = compute_majority_vote_mean_seg(segmentation_slices[selected_index])

        res_seg_model_space = np.asarray(res_seg_model_space)

        return Image(param=res_seg_model_space)

    # ------------------------------------------------------------------------------------------------------------------
    def sc_label_fusion(self, selected_index):
        """
        Compute the resulting segmentation by label fusion of the segmentation of the selected dictionary slices

        :param selected_index: array of indexes (as a boolean array) of the selected dictionary slices

        :return res_seg_model_space: Image of the resulting segmentation for the target image (in the model space)
        """
        sc_slices = np.asarray([(dic_slice.im_M > 200).astype(int) for dic_slice in self.model.dictionary.slices])

        for i, selected_ind_by_slice in enumerate(selected_index):  # selected_slices:
            slice_sc_seg = compute_majority_vote_mean_seg(sc_slices[selected_ind_by_slice])
            self.target[i].set(sc_seg=slice_sc_seg)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_projected_dic(self, nb_modes=3):
        """
        plot the pca first modes and the target projection if target is provided.

        on a second plot, highlight the selected dictionary slices for one target slice in particular

        :param nb_modes:
        :return:
        """
        self.model.pca.plot_projected_dic(nb_mode=nb_modes, target_coord=self.coord_projected_target) if self.coord_projected_target is not None \
            else self.model.pca.plot_projected_dic()

        self.model.pca.plot_projected_dic(nb_mode=nb_modes, target_coord=self.coord_projected_target, to_highlight=(5, self.selected_k_slices[5])) if self.coord_projected_target is not None \
            else self.model.pca.plot_projected_dic()

# ----------------------------------------------------------------------------------------------------------------------
# TARGET SEGMENTATION GROUPWISE ----------------------------------------------------------------------------------------
class TargetSegmentationGroupwise:
    """
    Contains all the function to segment the gray matter an a target image given a model

        - registration of the target to the model space

        - projection of the target slices on the reduced model space

        - selection of the model slices most similar to the target slices

        - computation of the resulting target segmentation by label fusion of their segmentation
    """
    def __init__(self, model, target_image=None, tau=None):
        """
        Target gray matter segmentation constructor

        :param model: Model used to compute the segmentation, type: Model

        :param target_image: Target image to segment gray matter on, type: Image

        :param tau: Weighting parameter associated with the geodesic distances in the model dictionary, type: float
        """
        self.model = model

        # Get the target image
        self.target = target_image
        '''
        save_image(self.target.data, 'target_image')
        print '---TARGET IMAGE IN CLASS TargetSegmentation : ', self.target.data
        save_image(self.target.data[0],'target_slice0_targetSeg_'+self.model.param.todo_model)
        '''
        self.epsilon = round(1.0/self.model.dictionary.J, 5) - 0.0001  # /2

        if tau is None:
            self.tau = self.compute_tau()
        else:
            self.tau = tau

        self.target_M, self.R_target_to_M = self.target_groupwise_registration()
        self.target_M.file_name = 'target_model_space'
        self.target_M.ext = '.nii.gz'
        # self.target_M.data = self.target_M.data.astype('float32')
        self.target_M.save()

        '''
        print '----- registered target ---- ', self.target_M.data
        save_image(self.target_M.data, 'target_image_model_space')
        save_image(self.target_M.data[0],'target_registered_slice0_rigidreg_'+self.model.param.todo_model)
        '''

        # coord_projected_target is a list of all the coord of the target's projected slices
        sct.printv('\nProjecting the target image in the reduced common space ...', model.param.verbose, 'normal')
        self.coord_projected_target = model.pca.project(self.target_M) if self.target_M is not None else None

        '''
        print '----SHAPE COORD PROJECTED TARGET -----', self.coord_projected_target.shape
        print '----COORD PROJECTED TARGET -----', self.coord_projected_target
        '''

        self.beta = self.compute_beta(self.coord_projected_target, tau=self.tau)
        '''
        print '----------- BETAS :', self.beta
        self.beta = self.compute_beta(self.coord_projected_target, tau=0.00114)
        '''
        sct.printv('\nSelecting the dictionary slices most similar to the target ...', model.param.verbose, 'normal')

        self.selected_k_slices = self.select_k_slices(self.beta)
        '''
        print '----SELECTED K -----', self.selected_k_slices
        print '----SHAPE SELECTED K -----', self.selected_k_slices.shape
        '''
        sct.printv('\nComputing the result gray matter segmentation ...', model.param.verbose, 'normal')
        self.target_GM_seg_M = self.label_fusion(self.selected_k_slices)
        self.target_GM_seg_M.file_name = 'res_gmseg_model_space'
        self.target_GM_seg_M.ext = '.nii.gz'
        # self.target_M.data = self.target_M.data.astype('float32')
        self.target_GM_seg_M.save()
        sct.printv('\nRegistering the result gray matter segmentation back into the target original space...',
                   model.param.verbose, 'normal')
        self.target_GM_seg = self.target_groupwise_registration(inverse=True)

    # ------------------------------------------------------------------------------------------------------------------
    def compute_beta(self, coord_target, dataset_coord=None, tau=0.01):
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
            dataset_coord = self.model.pca.dataset_coord.T

        beta = []
        '''
        sct.printv('----------- COMPUTING BETA --------------', 1, 'info')
        print '------ TAU = ', tau
        print '---------- IN BETA : coord_target ---------------->', coord_target
        print '---------- IN BETA : shape coord_target ---------------->', coord_target.shape, ' len = ',
         len(coord_target.shape)
        print '---------- IN BETA : type coord_target[0][0] ---------------->', type(coord_target[0][0])
        '''
        # TODO: SEE IF WE NEED TO CHECK THE SECOND DIMENSION OF COORD TARGET OR THE FIRST ...
        if isinstance(coord_target[0], (list, np.ndarray)):
            for i_target, coord_projected_slice in enumerate(coord_target):
                beta_slice = []
                for coord_slice_j in dataset_coord:
                    square_norm = np.linalg.norm((coord_projected_slice - coord_slice_j), 2)
                    beta_slice.append(exp(-tau*square_norm))

                '''
                print 'beta case 1 :', beta
                print '--> sum beta ', Z
                '''
                try:
                    if np.sum(beta_slice) != 0:
                        beta_slice /= np.sum(beta_slice)
                except ZeroDivisionError:
                    sct.printv('WARNING : similarities are null', self.model.param.verbose, 'warning')
                    print beta_slice

                beta.append(beta_slice)
        else:
            for coord_slice_j in dataset_coord:
                square_norm = np.linalg.norm((coord_target - coord_slice_j), 2)
                beta.append(exp(-tau*square_norm))

            try:
                beta /= np.sum(beta)
            except ZeroDivisionError:
                sct.printv('WARNING : similarities are null', self.model.param.verbose, 'warning')
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
                   '(Tau is a weighting parameter indicating the decay constant associated with a geodesic distance '
                   'between a given atlas and a projected target image, see Asman paper, eq (16))', 1, 'normal')
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
            for dic_slice in self.model.dictionary.slices:
                projected_dic_slice_coord = self.model.pca.project_array(dic_slice.im_M_flat)

                coord_dic_slice_dataset = np.delete(self.model.pca.dataset_coord.T, dic_slice.id, 0)

                beta_dic_slice = self.compute_beta(projected_dic_slice_coord, dataset_coord=coord_dic_slice_dataset,
                                                   tau=tau)
                kj = self.select_k_slices(beta_dic_slice)  # , poped=dic_slice.id)
                est_segm_j = self.label_fusion(kj)

                sum_norm += l0_norm(dic_slice.seg_M, est_segm_j.data)
            return sum_norm

        est_tau = minimize(to_minimize, 0, method='Nelder-Mead', options={'xtol': 0.0005})
        sct.printv('Estimated tau : ' + str(est_tau.x[0]))
        if self.model.param.todo_model == 'compute':
            fic = open(self.model.dictionary.model_dic_name + '/tau.txt', 'w')
            fic.write(str(est_tau.x[0]))
            fic.close()
        return float(est_tau.x[0])

    # ------------------------------------------------------------------------------------------------------------------
    def compute_mu(self, beta):
        """
        Compute the weighted mean of the dictionary slices projection weights

        :param beta: similarities vector for one target slice

        :return mu:
        """
        '''
        mu = []
        for beta_slice in beta:
            mu.append(self.model.pca.dataset_coord.dot(beta_slice))
        return np.asarray(mu)
        '''
        return self.model.pca.dataset_coord.dot(beta)

    # ------------------------------------------------------------------------------------------------------------------
    def compute_sigma(self, beta, mu):
        """
        Compute the weighted standard deviation of the dictionary slices projection weights

        :param beta: similarities vector for one target slice

        :param mu: weighted mean of the dictionary slices projection weights for one target slice

        :return sigma:
        """
        '''
        sigma = []
        for beta_slice, mu_slice in zip(beta, mu):

            sigma.append([beta_slice.dot((self.model.pca.dataset_coord[v, :] - mu_slice[v]) ** 2)
                          for v in range(len(mu_slice))])

        return np.asarray(sigma)
        '''
        return np.asarray([beta.dot((self.model.pca.dataset_coord[v, :] - mu[v]) ** 2) for v in range(len(mu))])

    # ------------------------------------------------------------------------------------------------------------------
    def target_groupwise_registration(self, inverse=False):
        """
        Register the target image into the model space

        Affine (or rigid + affine) registration of the target on the mean model image --> pairwise

        :param inverse: if True, apply the inverse warping field of the registration target -> model space
        to the result gray matter segmentation of the target
        (put it back in it's original space)
        """
        if not inverse:
            # Registration target --> model space
            from scipy.optimize import minimize

            # Initialisation
            target_m = []

            def to_minimize(t_param, n_slice):
                """

                :param :

                :return sum_norm:

                """
                moved_target_slice = Image(param=np.asarray(apply_2d_transformation(self.target.data[n_slice],
                                                                                    tx=t_param[0], ty=t_param[1],
                                                                                    theta=t_param[2],
                                                                                    s=t_param[3])[0]))

                coord_moved_target = self.model.pca.project_array(moved_target_slice.data.flatten())
                coord_moved_target = coord_moved_target.reshape(coord_moved_target.shape[0],)

                beta = self.compute_beta(coord_moved_target, tau=self.tau)
                mu = self.compute_mu(beta)
                sigma = self.compute_sigma(beta, mu)

                target_th_slice = np.sum((np.asarray([dic_slice.im_M for dic_slice in self.model.dictionary.slices]).T
                                          * beta[n_slice]).T, axis=0)
                sq_norm = np.linalg.norm(target_th_slice - moved_target_slice.data, 2)**2

                gauss = np.sum(((coord_moved_target - mu)/sigma)**2)
                return sq_norm*gauss

            r_target_to_m = []
            for i_slice, target_slice in enumerate(self.target.data):
                x0 = [0, 0, 0, 1]
                est_transfo = minimize(to_minimize, x0, args=i_slice, method='Nelder-Mead', options={'xtol': 0.00005})
                target_m_slice, r_slice = apply_2d_transformation(target_slice, tx=est_transfo.x[0],
                                                                  ty=est_transfo.x[1], theta=est_transfo.x[2],
                                                                  s=est_transfo.x[3])
                print est_transfo.x
                target_m.append(target_m_slice)
                r_target_to_m.append(r_slice)
            return Image(param=np.asarray(target_m)), r_target_to_m

        else:
            # Inverse registration result in model space --> target original space
            moved_res = []
            for i_slice, res_m_slice in enumerate(self.target_GM_seg_M.data):
                moved_res.append(apply_2d_transformation(res_m_slice, transfo=self.R_target_to_M[i_slice].inverse)[0])
            return Image(np.asarray(moved_res).astype('uint8'))

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
                '''
                if poped is not None:
                    selected_index = np.delete(selected_index, poped)
                '''
                # kept_seg_slices.append(segmentation_slices[selected_index])
                kept_slice_index.append(selected_index)

        else:
            kept_slice_index = beta > self.epsilon
            '''
            if poped is not None:
                selected_index = np.delete(selected_index, poped)
            '''
            # kept_seg_slices = segmentation_slices[selected_index]

        return np.asarray(kept_slice_index)

    def label_fusion(self, selected_index):
        """
        Compute the resulting segmentation by label fusion of the segmentation of the selected dictionary slices

        :param selected_index: array of indexes (as a boolean array) of the selected dictionary slices

        :return res_seg_model_space: Image of the resulting segmentation for the target image (in the model space)
        """
        segmentation_slices = np.asarray([dic_slice.seg_M for dic_slice in self.model.dictionary.slices])

        res_seg_model_space = []

        # if isinstance(selected_slices[0][0][0], (list, np.ndarray)):
        # if len(selected_slices[0].shape) == 3:
        if isinstance(selected_index[0], (list, np.ndarray)):

            for selected_ind_by_slice in selected_index:  # selected_slices:
                slice_seg = compute_majority_vote_mean_seg(segmentation_slices[selected_ind_by_slice], threshold=0.3)
                res_seg_model_space.append(slice_seg)
            # res_seg_model_space = map(compute_majority_vote_mean_seg, selected_slices)

        else:
            # res_seg_model_space = compute_majority_vote_mean_seg(selected_slices)
            res_seg_model_space = compute_majority_vote_mean_seg(segmentation_slices[selected_index], threshold=0.3)

        res_seg_model_space = np.asarray(res_seg_model_space)
        # save_image(res_seg_model_space, 'res_GM_seg_model_space')

        return Image(param=res_seg_model_space)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_projected_dic(self, nb_modes=3):
        """
        plot the pca first modes and the target projection if target is provided.

        on a second plot, highlight the selected dictionary slices for one target slice in particular

        :param nb_modes:
        :return:
        """
        self.model.pca.plot_projected_dic(nb_mode=nb_modes, target_coord=self.coord_projected_target) if self.coord_projected_target is not None \
            else self.model.pca.plot_projected_dic()

        self.model.pca.plot_projected_dic(nb_mode=nb_modes, target_coord=self.coord_projected_target, to_highlight=(6, self.selected_k_slices[6])) if self.coord_projected_target is not None \
            else self.model.pca.plot_projected_dic()
        print self.selected_k_slices[6]


# ----------------------------------------------------------------------------------------------------------------------
# GRAY MATTER SEGMENTATION SUPERVISED METHOD ---------------------------------------------------------------------------
class GMsegSupervisedMethod():
    """
    Gray matter segmentation supervised method:

    Load a dictionary (training data set), compute or load a model from this dictionary
sct_Image
    Load a target image to segment and do the segmentation using the model
    """
    def __init__(self, target_fname, gm_seg_param=None):

        self.dictionary = ModelDictionaryBySlice(dic_param=gm_seg_param)

        sct.printv('\nBuilding the appearance model...', verbose=gm_seg_param.verbose, type='normal')
        # build the appearance model
        self.model = Model(model_param=gm_seg_param, dictionary=self.dictionary, k=0.8)

        sct.printv('\nConstructing target image ...', verbose=gm_seg_param.verbose, type='normal')
        # construct target image
        self.target_image = Image(target_fname)

        tau = None  # 0.000765625  # 0.00025  # 0.000982421875  # 0.00090625  # None

        if gm_seg_param.todo_model == 'load':
            fic = open(self.model.dictionary.model_dic_name + '/tau.txt', 'r')
            tau = float(fic.read())
            fic.close()

        # build a target segmentation
        levels_im = None
        if gm_seg_param.level_fname is not None:
            if len(gm_seg_param.level_fname) < 3:
                # in this case the level is a string and not an image
                levels_im = gm_seg_param.level_fname
            else:
                levels_im = Image(gm_seg_param.level_fname)
        if gm_seg_param.target_reg == 'pairwise':
            if levels_im is not None:
                self.target_seg_methods = TargetSegmentationPairwise(self.model, target_image=self.target_image, levels_image=levels_im, tau=tau)
            else:
                self.target_seg_methods = TargetSegmentationPairwise(self.model, target_image=self.target_image, tau=tau)

        elif gm_seg_param.target_reg == 'groupwise':
            self.target_seg_methods = TargetSegmentationGroupwise(self.model, target_image=self.target_image, tau=tau)

        suffix = '_'
        suffix += gm_seg_param.target_reg
        for transfo in self.dictionary.coregistration_transfos:
            suffix += '_' + transfo
        if levels_im is not None:
            suffix += '_with_levels'
        else:
            suffix += '_no_levels'

        # save the result gray matter segmentation
        # self.res_GM_seg = self.target_seg_methods.target_GM_seg
        if len(self.target_seg_methods.target) == 1:
            self.res_wm_seg = Image(param=np.asarray(self.target_seg_methods.target[0].wm_seg))
            self.res_sc_seg = Image(param=np.asarray(self.target_seg_methods.target[0].sc_seg))
        else:
            self.res_wm_seg = Image(param=np.asarray([target_slice.wm_seg for target_slice in self.target_seg_methods.target]))
            self.res_sc_seg = Image(param=np.asarray([target_slice.sc_seg for target_slice in self.target_seg_methods.target]))

        name_res = sct.extract_fname(target_fname)[1] + '_res_wmseg' + suffix  # TODO: remove suffix
        self.res_wm_seg.file_name = name_res
        self.res_wm_seg.ext = '.nii.gz'
        self.res_wm_seg.save()

        self.res_gm_seg = inverse_wmseg_to_gmseg(self.res_wm_seg, self.res_sc_seg, name_res)

        corrected_wm_seg = correct_wmseg(self.res_gm_seg, self.target_image)
        corrected_wm_seg.file_name = name_res + '_corrected'
        corrected_wm_seg.ext = '.nii.gz'
        corrected_wm_seg.save()

        sct.printv('Done! \nTo see the result, type :')
        sct.printv('fslview ' + target_fname + ' ' + name_res + '_corrected.nii.gz -l Red -t 0.4 ' + name_res + '_inv_to_gm.nii.gz -l Blue -t 0.4 &', gm_seg_param.verbose, 'info')

    def show(self):

        sct.printv('\nShowing the pca modes ...')
        self.model.pca.show_all_modes()

        sct.printv('\nPloting Omega ...')
        self.target_seg_methods.plot_projected_dic(nb_modes=3)

        sct.printv('\nShowing PCA mode graphs ...')
        self.model.pca.show_mode_variation()
        print 'J :', self.dictionary.J

        '''
        sct.printv('\nShowing the projected target ...')
        self.target_seg_methods.show_projected_target()
        '''


########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################

if __name__ == "__main__":
    param = Param()
    input_target_fname = None
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
                          description="T2star image you want to project",
                          mandatory=True,
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
        parser.add_option(name="-target-reg",
                          type_value='multiple_choice',
                          description="type of registration of the target to the model space "
                                      "(if pairwise, the registration applied to the target are the same as"
                                      " those of the -reg flag)",
                          mandatory=False,
                          default_value='pairwise',
                          example=['pairwise', 'groupwise'])
        '''
        parser.add_option(name="-seg-type",
                          type_value='multiple_choice',
                          description="type of segmentation (gray matter or white matter)",
                          mandatory=False,
                          default_value='wm',
                          example=['wm', 'gm', 'gm-model'])
        '''
        parser.add_option(name="-v",
                          type_value="int",
                          description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                          mandatory=False,
                          default_value=0,
                          example='1')

        arguments = parser.parse(sys.argv[1:])
        input_target_fname = arguments["-i"]
        param.path_dictionary = arguments["-dic"]
        param.todo_model = arguments["-model"]

        if "-reg" in arguments:
            param.reg = arguments["-reg"]
        if "-target-reg" in arguments:
            param.target_reg = arguments["-target-reg"]
        '''
        if "-seg-type" in arguments:
            param.seg_type = arguments["-seg-type"]
        '''
        if "-l" in arguments:
            param.level_fname = arguments["-l"]
        if "-v" in arguments:
            param.verbose = arguments["-v"]

    gm_seg_method = GMsegSupervisedMethod(input_target_fname, gm_seg_param=param)

    if param.verbose == 2:
        gm_seg_method.show()