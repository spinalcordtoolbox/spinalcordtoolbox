#!/usr/bin/env python
########################################################################################################################
#
# Asman et al. groupwise multi-atlas segmentation method implementation
# The name of the attributes of each class correspond to the names in Asman et al. paper
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Sara Dupont
# Modified: 2015-03-24
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

# TODO change 'target' by 'input'
#TODO : make it faster



import os
import sys
import numpy as np

from msct_pca import PCA
from msct_image import Image
from msct_parser import *
from msct_gmseg_utils import *
import sct_utils as sct

from math import sqrt
from math import exp
from math import fabs



class Param:
    def __init__(self):
        self.debug = 0
        self.path_dictionary = '/Volumes/folder_shared/greymattersegmentation/data_asman/dictionary'
        self.todo_model = 'compute'
        self.target_fname = ''
        self.split_data = 0  # this flag enables to duplicate the image in the right-left direction in order to have more dataset for the PCA
        self.verbose = 1


########################################################################################################################
######------------------------------------------------- Classes --------------------------------------------------######
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# DATASET --------------------------------------------------------------------------------------------------------------
class Dataset:
    def __init__(self, param=None):
        if param is None:
            self.param = Param()
        else:
            self.param = param
        self.slices = []
        self.J = 0
        self.N = 0
        self.L = []
        self.mean_seg = None
        self.mean_data = None
        self.model_dic_name = ''
        if self.param.todo_model == 'compute':
            self.model_dic_name = './gmseg_model_dictionary'
            self.compute_model_dataset()
        elif self.param.todo_model == 'load':
            self.model_dic_name = self.param.path_dictionary #TODO change the path by the name of the dic ?? ...
            self.load_model_dataset()

        save_image(self.slices[3].A,'atlas_slice_3')
        save_image(self.slices[3].D,'dec_slice_3')

        #self.show_data()

    # ------------------------------------------------------------------------------------------------------------------
    # Compute the model using the dictionary dataset
    def compute_model_dataset(self):
        sct.printv('\nComputing the model dataset ...', self.param.verbose, 'normal')
        # Load all the images' slices from param.path_dictionary
        sct.printv('\nLoading dictionary ...', self.param.verbose, 'normal')
        #List of atlases (A) and their label decision (D) (=segmentation of the gray matter), slice by slice
        #zip(self.A,self.D) would give a list of tuples (slice_image,slice_segmentation)

        #self.A, self.D = self.load_dictionary()

        sct.runProcess('mkdir ' + self.model_dic_name)

        self.slices, dic_data_info = self.load_dictionary()

        #number of atlases in the dataset
        self.J = len([slice.A for slice in self.slices])
        #dimension of the data (flatten slices)
        self.N = len(self.slices[0].A.flatten())

        #inverts the segmentations to use segmentations of the WM instead of segmentations of the GM
        self.invert_seg()
        dic_data_info = self.save_model_data(dic_data_info,'inverted_D')

        #set of possible labels that can be assigned to a given voxel in the segmentation
        self.L = [0, 1] #1=WM, 0=GM or CSF

        sct.printv('\nComputing the rigid transformation to coregister all the data into a common groupwise space ...', self.param.verbose, 'normal')
        #list of rigid transformation for each slice to coregister the data into the common groupwise space
        self.mean_seg = self.find_rigid_coregistration()
        dic_data_info = self.save_model_data(dic_data_info,'DM')

        sct.printv('\nCoregistering all the data into the common groupwise space ...', self.param.verbose, 'normal')
        #List of atlases (A_M) and their label decision (D_M) (=segmentation of the gray matter), slice by slice in the common groupwise space
        #zip(self.A_M,self.D_M) would give a list of tuples (slice_image,slice_segmentation)
        self.coregister_data()
        dic_data_info = self.save_model_data(dic_data_info,'AM')

        self.mean_data = self.mean_dataset(np.asarray([slice.AM for slice in self.slices]))
        save_image(self.mean_data, 'mean_data', path=self.model_dic_name+'/', type='uint8')

    # ------------------------------------------------------------------------------------------------------------------
    # FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Load the dictionary:
    # each slice of each patient will be load separately in A with its corresponding GM segmentation in D
    def load_dictionary(self):
        # init
        slices = []
        j = 0
        # loop across all the volume
        data_info = {}
        #TODO: change the name of files to find to a more general structure
        for subject_dir in os.listdir(self.param.path_dictionary):
            subject_path = self.param.path_dictionary + '/' + subject_dir
            if os.path.isdir(subject_path):
                data_info[subject_dir] = {'n_slices':0,'inverted_D':[],'AM':[], 'DM':[]}

                subject_seg_in = ''
                subject_GM = ''
                sct.runProcess('mkdir ./' + self.model_dic_name + '/' + subject_dir, verbose=self.param.verbose)
                for file in os.listdir(subject_path):
                    if 'seg_in.nii' in file:
                        subject_seg_in = file
                        sct.runProcess('cp ./' +self.param.path_dictionary  + subject_dir + '/' + file + ' ' + self.model_dic_name + '/' + subject_dir + '/' + subject_dir + '_atlas.nii.gz')
                    if 'GM.nii' in file:
                        subject_GM = file
                        sct.runProcess('cp ./' +self.param.path_dictionary  + subject_dir + '/' + file + ' ' + self.model_dic_name + '/' + subject_dir + '/' + subject_dir + '_dec.nii.gz')
                atlas = Image(subject_path + '/' + subject_seg_in)
                seg = Image(subject_path + '/' + subject_GM)

                for atlas_slice, seg_slice in zip(atlas.data, seg.data):
                    data_info[subject_dir]['n_slices'] += 1
                    if self.param.split_data:
                        left_slice, right_slice = split(atlas_slice)

                        left_slice_seg, right_slice_seg = split(seg_slice)

                        slices.append(Slice(id=j, A=left_slice, D=left_slice_seg))
                        slices.append(Slice(id=j+1, A=right_slice, D=right_slice_seg))
                        j += 2


                    else:
                        slices.append(Slice(id=j, A=atlas_slice, D=seg_slice))
                        j += 1

        return np.asarray(slices), data_info

    # ------------------------------------------------------------------------------------------------------------------
    # save data images of the model
    def save_model_data(self,data_info,what_to_save):
        total_n_slices = 0
        if what_to_save == 'inverted_D':
            suffix = '_dec'
            data_to_save = [slice.D for slice in self.slices]
        elif what_to_save == 'AM':
            suffix = '_atlas_model_space'
            data_to_save = [slice.AM for slice in self.slices]
        elif what_to_save == 'DM':
            suffix = '_dec_model_space'
            data_to_save = [slice.DM for slice in self.slices]

        for subject_name in sorted(data_info.keys()):
            data_info[subject_name][what_to_save] = data_to_save[total_n_slices:total_n_slices+data_info[subject_name]['n_slices']]
            total_n_slices += data_info[subject_name]['n_slices']

        for subject_name, info in zip(data_info.keys(), data_info.values()):
            im_inverted_seg = Image(param=np.asarray(info[what_to_save]))
            im_inverted_seg.path = self.model_dic_name + '/' + subject_name + '/'

            im_inverted_seg.file_name = subject_name + suffix
            im_inverted_seg.ext = '.nii.gz'
            im_inverted_seg.save(type='minimize')
        return data_info

    # ------------------------------------------------------------------------------------------------------------------
    # Invert the segmentations of GM to get segmentation of the WM instead (more information, theoretically better results)
    def invert_seg(self):
        for slice in self.slices:
            im_a = Image(param=slice.A)
            sc = im_a.copy()
            nz_coord_sc = sc.getNonZeroCoordinates()
            im_d = Image(param=slice.D)
            nz_coord_d = im_d.getNonZeroCoordinates()
            for coord in nz_coord_sc:
                sc.data[coord.x, coord.y] = 1
            for coord in nz_coord_d:
                im_d.data[coord.x, coord.y] = 1
            #cast of the -1 values (-> GM pixel at the exterior of the SC pixels) to +1 --> WM pixel
            inverted_slice_decision = np.absolute(sc.data - im_d.data).astype(int)
            slice.set(D=inverted_slice_decision)

    # ------------------------------------------------------------------------------------------------------------------
    # return the rigid transformation (for each slice, computed on D data) to coregister all the atlas information to a common groupwise space
    def find_rigid_coregistration(self):
        # initialization
        chi = self.compute_mean_seg(np.asarray([slice.D for slice in self.slices]))
        for slice in self.slices:
            name_j_transform = 'rigid_transform_slice_' + str(slice.id) + '.mat'
            slice.set(RtoM=name_j_transform)
            decision_M = apply_ants_2D_rigid_transfo(chi, slice.D,  transfo_name=name_j_transform, path=self.model_dic_name+'/')
            slice.set(DM=decision_M.astype(int))
            slice.set(DM_flat=decision_M.flatten().astype(int))
        chi = self.compute_mean_seg([slice.DM for slice in self.slices])

        save_image(chi, 'mean_seg', path=self.model_dic_name+'/', type='uint8')

        return chi

    # ------------------------------------------------------------------------------------------------------------------
    # Compute the mean segmentation image 'chi' for a given decision dataset D
    def compute_mean_seg(self, D):
        mean_seg = []
        choose_maj_vote = {}
        for l in self.L:
            to_be_summed = []
            for slice in D:
                consistent_vox = []
                for row in slice:
                    for i in row:
                        try:
                            if i > 0.2:
                                i = 1
                        except ValueError:
                            print 'Value Error with i = ', i
                            print 'Dataset was : \n', D
                        consistent_vox.append(kronecker_delta(i, l))
                to_be_summed.append(consistent_vox)
            summed_vector = np.zeros(len(to_be_summed[0]), dtype=np.int)
            for v in to_be_summed:
                summed_vector = np.add(summed_vector, v)
            choose_maj_vote[l] = summed_vector

        for vote_tuple in zip(choose_maj_vote[0], choose_maj_vote[1]):
            if vote_tuple[0] >= vote_tuple[1]:
                mean_seg.append(0)
            elif vote_tuple[1] > vote_tuple[0]:
                mean_seg.append(1)
        n = int(sqrt(self.N))
        return np.asarray(mean_seg).reshape(n, n)


    # ------------------------------------------------------------------------------------------------------------------
    # compute the mean atlas image (used to coregister the dataset)
    def mean_dataset(self, A):
        sum=np.zeros(A[0].shape)
        for slice_A in A:
            for k,row in enumerate(slice_A):
                sum[k] = np.add(sum[k],row)
        mean = sum/self.J
        return mean

    # ------------------------------------------------------------------------------------------------------------------
    # return the coregistered data into the common groupwise space using the previously computed rigid transformation :self.RM
    def coregister_data(self):
        list_A = [slice.A for slice in self.slices]
        for slice in self.slices:
            atlas_M = apply_ants_2D_rigid_transfo(self.mean_dataset(list_A), slice.A, search_reg=False, transfo_name=slice.RtoM, binary=False, path=self.model_dic_name+'/')
            #apply_2D_rigid_transformation(self.A[j], self.RM[j]['tx'], self.RM[j]['ty'], self.RM[j]['theta'])
            slice.set(AM=atlas_M)
            slice.set(AM_flat=atlas_M.flatten())

    # ------------------------------------------------------------------------------------------------------------------
    # END OF FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # Load the model data using the dataset
    def load_model_dataset(self):
        sct.printv('\nLoading the model dataset ...', self.param.verbose, 'normal')

        j = 0
        for subject_dir in os.listdir(self.model_dic_name):
            subject_path = self.model_dic_name + '/' + subject_dir
            if os.path.isdir(subject_path) and 'transformations' not in subject_path :
                subject_atlas= ''
                subject_dec = ''
                subject_atlas_m= ''
                subject_dec_m= ''

                for file in os.listdir(subject_path):
                    if '_atlas.nii' in file:
                        subject_atlas = file
                    if '_dec.nii' in file:
                        subject_dec = file
                    if '_atlas_model_space.nii' in file:
                        subject_atlas_m = file
                    if '_dec_model_space.nii' in file:
                        subject_dec_m = file

                atlas = Image(subject_path + '/' + subject_atlas)
                dec = Image(subject_path + '/' + subject_dec)
                atlas_m = Image(subject_path + '/' + subject_atlas_m)
                dec_m = Image(subject_path + '/' + subject_dec_m)

                for atlas_slice, dec_slice, atlas_m_slice, dec_m_slice in zip(atlas.data, dec.data, atlas_m.data, dec_m.data):
                    if self.param.split_data:
                        left_slice, right_slice = split(atlas_slice)
                        left_slice_dec, right_slice_dec = split(dec_slice)
                        left_slice_m, right_slice_m = split(atlas_m_slice)
                        left_slice_dec_m, right_slice_dec_m = split(dec_m_slice)


                        self.slices.append(Slice(id=j, A=left_slice, D=left_slice_dec, AM=left_slice_m, DM=left_slice_dec_m, AM_flat=left_slice_m.flatten(), DM_flat=left_slice_dec_m.flatten(), RtoM='rigid_transform_slice_' + str(j) + '.mat'))
                        self.slices.append(Slice(id=j+1, A=right_slice, D=right_slice_dec, AM=right_slice_m, DM=right_slice_dec_m, AM_flat=right_slice_m.flatten(), DM_flat=right_slice_dec_m.flatten(), RtoM='rigid_transform_slice_' + str(j+1) + '.mat'))
                        j += 2


                    else:
                        self.slices.append(Slice(id=j, A=atlas_slice, D=dec_slice, AM=atlas_m_slice, DM=dec_m_slice, AM_flat=atlas_m_slice.flatten(), DM_flat=dec_m_slice.flatten(), RtoM='rigid_transform_slice_' + str(j) + '.mat'))
                        j += 1

        #number of atlases in the dataset
        self.J = len([slice.A for slice in self.slices])
        #dimension of the data (flatten slices)
        self.N = len(self.slices[0].A.flatten())

        #set of possible labels that can be assigned to a given voxel in the segmentation
        self.L = [0, 1] #1=WM, 0=GM or CSF

        self.mean_data = Image(self.model_dic_name + 'mean_data.nii.gz').data
        self.mean_seg = Image(self.model_dic_name + 'mean_seg.nii.gz').data


    # ------------------------------------------------------------------------------------------------------------------
    # FUNCTIONS USED TO LOAD THE MODEL
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # END OF FUNCTIONS USED TO LOAD THE MODEL
    # ------------------------------------------------------------------------------------------------------------------


    def show_data(self):
        for slice in self.slices:
            fig = plt.figure()

            d = fig.add_subplot(2,3, 1)
            d.set_title('Original space - seg')
            im_D = d.imshow(slice.D)
            im_D.set_interpolation('nearest')
            im_D.set_cmap('gray')

            dm = fig.add_subplot(2,3, 2)
            dm.set_title('Common groupwise space - seg')
            im_DM = dm.imshow(slice.DM)
            im_DM.set_interpolation('nearest')
            im_DM.set_cmap('gray')

            seg = fig.add_subplot(2,3, 3)
            seg.set_title('Mean seg')
            im_seg = seg.imshow(np.asarray(self.mean_seg))
            im_seg.set_interpolation('nearest')
            im_seg.set_cmap('gray')

            a = fig.add_subplot(2,3, 4)
            a.set_title('Original space - data ')
            im_A = a.imshow(slice.A)
            im_A.set_interpolation('nearest')
            im_A.set_cmap('gray')

            am = fig.add_subplot(2,3, 5)
            am.set_title('Common groupwise space - data ')
            im_AM = am.imshow(slice.AM)
            im_AM.set_interpolation('nearest')
            im_AM.set_cmap('gray')

            plt.suptitle('Slice ' + str(slice.id))
            plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# APPEARANCE MODEL -----------------------------------------------------------------------------------------------------
class AppearanceModel:
    def __init__(self, param=None, dataset=None):
        if param is None:
            self.param = Param()
        else:
            self.param = param

        self.dataset = dataset

        sct.printv("The shape of the dataset used for the PCA is (" + str(self.dataset.N) + ',' + str(self.dataset.J) + ')', verbose=self.param.verbose)
        # Instantiate a PCA object given the dataset just build
        sct.printv('\nCreating a reduced common space (using a PCA) ...', self.param.verbose, 'normal')
        '''
        print '######################################################################################################\n' \
              '######################################################################################################\n' \
              'TODO MODEL : ' + self.dataset.param.todo_model+ \
              ' \n######################################################################################################\n' \
              '######################################################################################################\n' \
              'DM_FLAT :\n ', self.dataset.slices[0].DM_flat, '\nDM NOT FLAT :\n', self.dataset.slices[0].DM
        '''
        self.pca = PCA(np.asarray([slice.AM_flat for slice in self.dataset.slices]).T, k=0.8) #WARNING : k usually is 0.8


# ----------------------------------------------------------------------------------------------------------------------
# RIGID REGISTRATION ---------------------------------------------------------------------------------------------------
class RigidRegistration:
    def __init__(self, appearance_model, target_image=None, tau=None):
        self.appearance_model = appearance_model

        # Get the target image
        self.target = target_image
        #save_image(self.target.data, 'target_image')

        self.target_M = self.register_target_to_model_space()
        #save_image(self.target_M.data, 'target_image_model_space')

        # coord_projected_target is a list of all the coord of the target's projected slices
        sct.printv('\nProjecting the target image in the reduced common space ...', appearance_model.param.verbose, 'normal')
        self.coord_projected_target = appearance_model.pca.project(self.target_M) if self.target_M is not None else None
        #print '----SHAPE COORD PROJECTED TARGET -----', self.coord_projected_target.shape
        #print '----COORD PROJECTED TARGET -----', self.coord_projected_target
        if tau is None :
            self.tau = self.compute_tau()
        else:
            self.tau = tau
        self.beta = self.compute_beta(self.coord_projected_target, tau=self.tau)
        #print '----------- BETAS :', self.beta
        #self.beta = self.compute_beta(self.coord_projected_target, tau=0.00114)

        sct.printv('\nSelecting the atlases closest to the target ...', appearance_model.param.verbose, 'normal')
        self.selected_K = self.select_K(self.beta)
        #print '----SELECTED K -----', self.selected_K
        #print '----SHAPE SELECTED K -----', self.selected_K.shape
        sct.printv('\nComputing the result gray matter segmentation ...', appearance_model.param.verbose, 'normal')
        self.target_GM_seg_M = self.label_fusion(self.selected_K)
        self.target_GM_seg = self.inverse_register_target()

    # ------------------------------------------------------------------------------------------------------------------
    #
    def register_target_to_model_space(self):
        target_M = []
        for i,slice in enumerate(self.target.data):
            #slice_M = apply_ants_2D_rigid_transfo(self.appearance_model.dataset.mean_data, slice, binary=False, transfo_type='Affine', transfo_name='transfo_target_to_model_space_slice_' + str(i) + '.mat')
            n = int(sqrt(self.appearance_model.pca.N))
            mean_vect = self.appearance_model.pca.mean_image.reshape(len(self.appearance_model.pca.mean_image),)
            im = mean_vect.reshape(n, n).astype(np.float)
            slice_M = apply_ants_2D_rigid_transfo(im, slice, binary=False, transfo_type='Affine', transfo_name='transfo_target_to_model_space_slice_' + str(i) + '.mat')
            target_M.append(slice_M)
        return Image(param=np.asarray(target_M))

    # ------------------------------------------------------------------------------------------------------------------
    #
    def inverse_register_target(self):
        res_seg = []
        for i,slice_M in enumerate(self.target_GM_seg_M.data):
            #slice = apply_ants_2D_rigid_transfo(self.appearance_model.dataset.mean_data, slice_M, search_reg=False ,binary=True, inverse=1, transfo_type='Affine', transfo_name='transfo_target_to_model_space_slice_' + str(i) + '.mat')
            n = int(sqrt(self.appearance_model.pca.N))
            mean_vect = self.appearance_model.pca.mean_image.reshape(len(self.appearance_model.pca.mean_image),)
            im = mean_vect.reshape(n, n).astype(np.float)
            slice = apply_ants_2D_rigid_transfo(im, slice_M, search_reg=False ,binary=True, inverse=1, transfo_type='Affine', transfo_name='transfo_target_to_model_space_slice_' + str(i) + '.mat')
            res_seg.append(slice)
        return Image(param=np.asarray(res_seg))



    # ------------------------------------------------------------------------------------------------------------------
    # beta is the model similarity between all the individual images and our input image
    # beta = (1/Z)exp(-tau*square_norm(omega-omega_j))
    # Z is the partition function that enforces the constraint tha sum(beta)=1
    def compute_beta(self, coord_target, tau=0.01):
        beta = []
        #sct.printv('----------- COMPUTING BETA --------------', 1, 'info')
        #print '------ TAU = ', tau
        #print '---------- IN BETA : coord_target ---------------->', coord_target
        #print '---------- IN BETA : shape coord_target ---------------->', coord_target.shape, ' len = ', len(coord_target.shape)
        #print '---------- IN BETA : type coord_target[0][0] ---------------->', type(coord_target[0][0])
        if isinstance(coord_target[0][0], (list, np.ndarray)): # SEE IF WE NEED TO CHECK TEH SECOND DIMENSION OF COORD TARGET OR THE FIRST ...
            for i,coord_projected_slice in enumerate(coord_target):
                beta_slice = []
                # in omega matrix, each column correspond to the projection of one of the original data image,
                # the transpose operator .T enable the loop to iterate over all the images coord
                for omega_j in self.appearance_model.pca.omega.T:
                    square_norm = np.linalg.norm((coord_projected_slice - omega_j), 2)
                    beta_slice.append(exp(-tau*square_norm))

                Z = sum(beta_slice)
                for i, b in enumerate(beta_slice):
                    beta_slice[i] = (1/Z) * b

                beta.append(beta_slice)
        else:
            # in omega matrix, each column correspond to the projection of one of the original data image,
            # the transpose operator .T enable the loop to iterate over all the images coord
            for omega_j in self.appearance_model.pca.omega.T:
                square_norm = np.linalg.norm((coord_target - omega_j), 2)
                beta.append(exp(-tau*square_norm))

                Z = sum(beta)
                for i, b in enumerate(beta):
                    beta[i] = (1/Z) * b

        return np.asarray(beta)


    # ------------------------------------------------------------------------------------------------------------------
    # decay constant associated with the geodesic distance between a given atlas and the projected target image in model space.
    def compute_tau(self):
        sct.printv('\nComputing Tau ... \n'
                   '(Tau is a parameter indicatng the decay constant associated with a geodesic distance between a given atlas and a projected target image, see Asman paper, eq (16))', 1, 'normal')
        from scipy.optimize import minimize
        def to_minimize(tau):
            sum_norm = 0
            for slice in self.appearance_model.dataset.slices:
                Kj = self.select_K(self.compute_beta(self.appearance_model.pca.project_array(slice.AM_flat), tau=tau)) #in project : Image(param=Amj)
                est_dmj = self.label_fusion(Kj)
                sum_norm += l0_norm(slice.DM, est_dmj.data)
            return sum_norm
        est_tau = minimize(to_minimize, 0, method='Nelder-Mead', options={'xtol':0.0005})
        sct.printv('Estimated tau : ' + str(est_tau.x[0]))
        if self.appearance_model.param.todo_model == 'compute':
            fic = open(self.appearance_model.dataset.model_dic_name + '/tau.txt','w')
            fic.write(str(est_tau.x[0]))
            fic.close()
        return float(est_tau.x[0])




    # ------------------------------------------------------------------------------------------------------------------
    # returns the index of the selected slices of the dataset to do label fusion and compute the graymater segmentation
    def select_K(self, beta, epsilon=0.015):#0.015
        selected = []
        #print '---------- IN SELECT_K : shape beta ---------------->', beta.shape, ' len = ', len(beta.shape)
        #print '---------- IN SELECT_K : type beta[0] ---------------->', type(beta[0])
        if isinstance(beta[0], (list, np.ndarray)):
            for beta_slice in beta:
                selected_by_slice=[]
                for j, beta_j in enumerate(beta_slice):
                    if beta_j > epsilon:
                        selected_by_slice.append(j)
                # selected.append(np.asarray(selected_by_slice))
                selected.append(selected_by_slice)
        else:
            for j, beta_j in enumerate(beta):
                if beta_j > epsilon:
                    selected.append(j)

        return np.asarray(selected)

    def label_fusion(self, selected_K):
        res_seg_M = []
        #print '---------- IN LABEL_FUSION : shape selected_K ---------------->', selected_K.shape, ' len = ', len(selected_K.shape)
        #print '---------- IN LABEL_FUSION : type selected_K[0] ---------------->', type(selected_K[0])
        if isinstance(selected_K[0], (list, np.ndarray)):
            for i, selected_by_slice in enumerate(selected_K):
                kept_decision_dataset = []
                for j in selected_by_slice:
                    kept_decision_dataset.append(self.appearance_model.dataset.slices[j].DM)
                slice_seg = self.appearance_model.dataset.compute_mean_seg(kept_decision_dataset)
                res_seg_M.append(slice_seg)
        else:
            kept_decision_dataset = []
            for j in selected_K:
                kept_decision_dataset.append(self.appearance_model.dataset.slices[j].DM)
            slice_seg = self.appearance_model.dataset.compute_mean_seg(kept_decision_dataset)
            res_seg_M = slice_seg

        res_seg_M = np.asarray(res_seg_M)
        #save_image(res_seg_M, 'res_GM_seg_model_space')

        return Image(res_seg_M)


    # ------------------------------------------------------------------------------------------------------------------
    # plot the pca and the target projection if target is provided
    def plot_omega(self,nb_modes=3):
        self.appearance_model.pca.plot_omega(nb_mode=nb_modes, target_coord=self.coord_projected_target, to_highlight=(3, self.selected_K[3])) if self.coord_projected_target is not None \
            else self.appearance_model.pca.plot_omega()

    # ------------------------------------------------------------------------------------------------------------------
    def show_projected_target(self):
        # Retrieving projected image from the mean image & its coordinates
        import copy

        index = 0
        fig1 = plt.figure()
        fig2 = plt.figure()
        # loop across all the projected slices coord
        for coord in self.coord_projected_target:
            img_reducted = copy.copy(self.appearance_model.pca.mean_image)
            # loop across coord and build projected image
            for i in range(0, coord.shape[0]):
                img_reducted += int(coord[i][0]) * self.appearance_model.pca.W.T[i].reshape(self.appearance_model.pca.N, 1)

            if self.appearance_model.param.split_data:
                n = int(sqrt(self.appearance_model.pca.N * 2))
            else:
                n = int(sqrt(self.appearance_model.pca.N))

            # Plot original image
            orig_ax = fig1.add_subplot(10, 3, index)
            orig_ax.set_title('original slice {} '.format(index))
            if self.appearance_model.param.split_data:
                imgplot = orig_ax.imshow(self.target.data[index, :, :].reshape(n / 2, n))
            else:
                imgplot = orig_ax.imshow(self.target.data[index].reshape(n, n))
            imgplot.set_interpolation('nearest')
            imgplot.set_cmap('gray')
            # plt.title('Original Image')
            # plt.show()

            index += 1
            # Plot projected image image
            proj_ax = fig2.add_subplot(10, 3, index)
            proj_ax.set_title('slice {} projected'.format(index))
            if self.appearance_model.param.split_data:
                imgplot = proj_ax.imshow(img_reducted.reshape(n / 2, n))
                #imgplot = plt.imshow(img_reducted.reshape(n / 2, n))
            else:
                # imgplot = plt.imshow(img_reducted.reshape(n, n))
                imgplot = proj_ax.imshow(img_reducted.reshape(n, n))
            imgplot.set_interpolation('nearest')
            imgplot.set_cmap('gray')
            # plt.title('Projected Image')
            # plt.show()
        plt.show()



# ----------------------------------------------------------------------------------------------------------------------
# ASMAN METHOD ---------------------------------------------------------------------------------------------------
class Asman():
    def __init__(self, param=None):

        self.dataset = Dataset(param=param)

        sct.printv('\nBuilding the appearance model...', verbose=param.verbose, type='normal')
        # build the appearance model
        self.appearance_model = AppearanceModel(param=param, dataset=self.dataset)

        sct.printv('\nConstructing target image ...', verbose=param.verbose, type='normal')
        # construct target image
        self.target_image = Image(param.target_fname)
        if param.split_data:
            splited_target = []
            for slice in self.target_image.data:
                left_slice, right_slice = split(slice)
                splited_target.append(left_slice)
                splited_target.append(right_slice)
            self.target_image = Image(np.asarray(splited_target))

        tau=0.000982421875 #0.00090625 #None
        '''
        if param.todo_model == 'load' :
            fic = open(self.appearance_model.dataset.model_dic_name + '/tau.txt','r')
            tau = float(fic.read())
            fic.close()
        '''

        #build a rigid registration
        self.rigid_reg = RigidRegistration(self.appearance_model, target_image=self.target_image, tau=tau)

        self.res_GM_seg = self.rigid_reg.target_GM_seg
        name_res = sct.extract_fname(param.target_fname)[1] + '_GM_seg'
        save_image(self.res_GM_seg.data, name_res)

        inverted_res_seg=[]
        for slice in self.res_GM_seg.data:
            inverted_res_seg.append(np.absolute(slice-1))
        inverted_res_seg_image = Image(param=np.asarray(inverted_res_seg))
        inverted_res_seg_image.file_name = name_res + '_inv'
        inverted_res_seg_image.ext = '.nii.gz'
        inverted_res_seg_image.save()

        sct.printv('Done! \nTo see the result, type : fslview ' + param.target_fname + ' ' + name_res + '.nii.gz -l Red -t 0.4 &')

    def show(self):

        sct.printv('\nShowing the pca modes ...')
        self.appearance_model.pca.show_all_modes()


        sct.printv('\nPloting Omega ...')
        self.rigid_reg.plot_omega(nb_modes=3)

        sct.printv('\nShowing PCA mode graphs ...')
        self.appearance_model.pca.show_mode()
        print 'J :',self.dataset.J

        '''
        sct.printv('\nShowing the projected target ...')
        self.rigid_reg.show_projected_target()
        '''







########################################################################################################################
######-------------------------------------------------  MAIN   --------------------------------------------------######
########################################################################################################################

if __name__ == "__main__":
    param = Param()

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
                          example=['load','compute'])
        parser.add_option(name="-split",
                          type_value="int",
                          description="1 will split all images from dictionary in the right-left direction in order to have more dataset for the PCA",
                          mandatory=False,
                          example='0')
        parser.add_option(name="-v",
                          type_value="int",
                          description="verbose",
                          mandatory=False,
                          example='1')

        arguments = parser.parse(sys.argv[1:])
        param.target_fname = arguments["-i"]
        param.path_dictionary = arguments["-dic"]
        param.todo_model = arguments["-model"]

        if "-split" in arguments:
            param.split_data = arguments["-split"]
        if "-v" in arguments:
            param.verbose = arguments["-v"]


    asman_seg = Asman(param=param)

    asman_seg.show()

