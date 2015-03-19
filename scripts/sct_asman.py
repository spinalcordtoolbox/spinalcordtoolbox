#!/usr/bin/env python
########################################################################################################################
#
# Asman et al. groupwise multi-atlas segmentation method implementation
# The name of the attributes of each class correspond to the names in Asman et al. paper
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Sara Dupont
# Modified: 2015-03-12
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

# TODO change 'target' by 'input'
#TODO : make it faster



import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from msct_pca import PCA
from msct_image import Image
from msct_parser import *
import sct_utils as sct

from math import sqrt
from math import exp
from math import fabs
#from math import log
#from math import pi



class Param:
    def __init__(self):
        self.debug = 0
        self.path_dictionary = '/Volumes/folder_shared/greymattersegmentation/data_asman/dictionary'
        self.target_fname = ''
        self.split_data = 0  # this flag enables to duplicate the image in the right-left direction in order to have more dataset for the PCA
        self.verbose = 0


########################################################################################################################
######------------------------------------------------- Classes --------------------------------------------------######
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# DATA -----------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, param=None):
        if param is None:
            self.param = Param()
        else:
            self.param = param

        # Load all the images' slices from param.path_dictionary
        sct.printv('\nLoading dictionary ...', self.param.verbose, 'normal')
        #List of atlases (A) and their label decision (D) (=segmentation of the gray matter), slice by slice
        #zip(self.A,self.D) would give a list of tuples (slice_image,slice_segmentation)
        self.A, self.D = self.load_dictionary()

        #number of atlases in the dataset
        self.J = len(self.A)
        #dimension of the data (flatten slices)
        self.N = len(self.A[0].flatten())

        #inverts the segmentations to use segmentations of the WM instead of segmentations of the GM
        self.invert_seg()

        #set of possible labels that can be assigned to a given voxel in the segmentation
        self.L = [0, 1] #1=WM, 0=GM or CSF

        sct.printv('\nComputing the rigid transformation to coregister all the data into a common groupwise space ...', self.param.verbose, 'normal')
        #list of rigid transformation for each slice to coregister the data into the common groupwise space
        self.R_to_M, self.mean_seg, self.D_M, self.D_M_flat = self.find_rigid_coregistration()

        sct.printv('\nCoregistering all the data into the common groupwise space ...', self.param.verbose, 'normal')
        #List of atlases (A_M) and their label decision (D_M) (=segmentation of the gray matter), slice by slice in the common groupwise space
        #zip(self.A_M,self.D_M) would give a list of tuples (slice_image,slice_segmentation)
        self.A_M, self.A_M_flat = self.coregister_data()

        self.mean_data = self.mean_dataset(self.A_M)

        #self.show_data()

    # ------------------------------------------------------------------------------------------------------------------
    # Load the dictionary:
    # each slice of each patient will be load separately in A with its corresponding GM segmentation in D
    def load_dictionary(self):
        # init
        atlas_slices = []
        decision_slices = []
        # loop across all the volume
        #TODO: change the name of files to find to a more general structure
        for subject_dir in os.listdir(self.param.path_dictionary):
            subject_path = self.param.path_dictionary + '/' + subject_dir
            if os.path.isdir(subject_path):
                subject_seg_in = ''
                subject_GM = ''
                for file in os.listdir(subject_path):
                    if 'seg_in.nii' in file:
                        subject_seg_in = file
                    if 'GM.nii' in file:
                        subject_GM = file
                atlas = Image(subject_path + '/' + subject_seg_in)
                seg = Image(subject_path + '/' + subject_GM)

                for atlas_slice, seg_slice in zip(atlas.data, seg.data):
                    if self.param.split_data:
                        left_slice, right_slice = split(atlas_slice)
                        atlas_slices.append(left_slice)
                        atlas_slices.append(right_slice)
                        left_slice_seg, right_slice_seg = split(seg_slice)
                        decision_slices.append(left_slice_seg)
                        decision_slices.append(right_slice_seg)
                    else:
                        atlas_slices.append(atlas_slice)
                        decision_slices.append(seg_slice)

        return np.asarray(atlas_slices), np.asarray(decision_slices)

    # ------------------------------------------------------------------------------------------------------------------
    # Invert the segmentations of GM to get segmentation of the WM instead (more information, theoretically better results)
    def invert_seg(self):
        for j,a in enumerate(self.A):
            im_a = Image(param=a)
            sc = im_a.copy()
            nz_coord_sc = sc.getNonZeroCoordinates()
            im_d = Image(param=self.D[j])
            nz_coord_d = im_d.getNonZeroCoordinates()
            for coord in nz_coord_sc:
                sc.data[coord.x, coord.y] = 1
            for coord in nz_coord_d:
                im_d.data[coord.x, coord.y] = 1
            #cast of the -1 values (-> GM pixel at the exterior of the SC pixels) to +1 --> WM pixel
            self.D[j] = np.absolute(sc.data - im_d.data)

    # ------------------------------------------------------------------------------------------------------------------
    # return the rigid transformation (for each slice, computed on D data) to coregister all the atlas information to a common groupwise space
    def find_rigid_coregistration(self):
        # initialization
        R = []
        Dm = []
        Dm_flat = []
        chi = self.compute_mean_seg(self.D)
        for j,Dj in enumerate(self.D):
            name_j_transform = 'rigid_transform_slice_' + str(j) + '.mat'
            R.append(name_j_transform)
            decision_M = apply_ants_2D_rigid_transfo(chi, Dj,  transfo_name=name_j_transform)
            Dm.append(decision_M)
            Dm_flat.append(decision_M.flatten())
        chi = self.compute_mean_seg(Dm)

        save_image(chi, 'mean_seg', path=param.path_dictionary, type='uint8')

        return R, chi, np.asarray(Dm), np.asarray(Dm_flat)

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
                        if i > 0.2:
                            i = 1
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
        return np.asarray(mean_seg).reshape(n,n)


    # ------------------------------------------------------------------------------------------------------------------
    # compute the mean atlas image (used to coregister the dataset)
    def mean_dataset(self, A):
        sum=np.zeros(A[0].shape)
        for slice in A:
            for k,row in enumerate(slice):
                sum[k] = np.add(sum[k],row)
        mean = sum/self.J
        return mean

    # ------------------------------------------------------------------------------------------------------------------
    # return the coregistered data into the common groupwise space using the previously computed rigid transformation :self.RM
    def coregister_data(self):
        A_M = []
        A_M_flat = []
        for j in range(self.J):
            atlas_M = apply_ants_2D_rigid_transfo(self.mean_dataset(self.A), self.A[j], search_reg=False, transfo_name=self.R_to_M[j], binary=False)
            #apply_2D_rigid_transformation(self.A[j], self.RM[j]['tx'], self.RM[j]['ty'], self.RM[j]['theta'])
            A_M.append(atlas_M)
            A_M_flat.append(atlas_M.flatten())

        return np.asarray(A_M), np.asarray(A_M_flat)

    def show_data(self):
        for j in range(self.J):
            fig = plt.figure()

            d = fig.add_subplot(2,3, 1)
            d.set_title('Original space - seg')
            im_D = d.imshow(self.D[j])
            im_D.set_interpolation('nearest')
            im_D.set_cmap('gray')

            dm = fig.add_subplot(2,3, 2)
            dm.set_title('Common groupwise space - seg')
            im_DM = dm.imshow(self.D_M[j])
            im_DM.set_interpolation('nearest')
            im_DM.set_cmap('gray')

            seg = fig.add_subplot(2,3, 3)
            seg.set_title('Mean seg')
            im_seg = seg.imshow(np.asarray(self.mean_seg))
            im_seg.set_interpolation('nearest')
            im_seg.set_cmap('gray')

            a = fig.add_subplot(2,3, 4)
            a.set_title('Original space - data ')
            im_A = a.imshow(self.A[j])
            im_A.set_interpolation('nearest')
            im_A.set_cmap('gray')

            am = fig.add_subplot(2,3, 5)
            am.set_title('Common groupwise space - data ')
            im_AM = am.imshow(self.A_M[j])
            im_AM.set_interpolation('nearest')
            im_AM.set_cmap('gray')

            plt.suptitle('Slice ' + str(j))
            plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# APPEARANCE MODEL -----------------------------------------------------------------------------------------------------
class AppearanceModel:
    def __init__(self, param=None, data=None):
        if param is None:
            self.param = Param()
        else:
            self.param = param

        self.data = data

        sct.printv("The shape of the dataset used for the PCA is {}".format(self.data.A_M_flat.T.shape), verbose=self.param.verbose)
        # Instantiate a PCA object given the dataset just build
        sct.printv('\nCreating a reduced common space (using a PCA) ...', self.param.verbose, 'normal')
        self.pca = PCA(self.data.A_M_flat.T, k=0.8) #WARNING : k usually is 0.8


# ----------------------------------------------------------------------------------------------------------------------
# RIGID REGISTRATION ---------------------------------------------------------------------------------------------------
class RigidRegistration:
    def __init__(self, appearance_model, target_image=None):
        self.appearance_model = appearance_model
        # Get the target image
        self.target = target_image
        save_image(self.target.data, 'target_image')
        self.target_M = self.register_target_to_model_space()
        save_image(self.target_M.data, 'target_image_model_space')

        # coord_projected_target is a list of all the coord of the target's projected slices
        sct.printv('\nProjecting the target image in the reduced common space ...', appearance_model.param.verbose, 'normal')
        self.coord_projected_target = appearance_model.pca.project(self.target_M) if self.target_M is not None else None

        self.beta = self.compute_beta()
        self.selected_K = self.select_K()
        self.target_GM_seg_M = self.label_fusion()
        self.target_GM_seg = self.inverse_register_target()

    # ------------------------------------------------------------------------------------------------------------------
    #
    def register_target_to_model_space(self):
        target_M = []
        for i,slice in enumerate(self.target.data):
            slice_M = apply_ants_2D_rigid_transfo(self.appearance_model.data.mean_data, slice, binary=False, transfo_name='transfo_target_to_model_space_slice_' + str(i) + '.mat')
            target_M.append(slice_M)
        return Image(param=np.asarray(target_M))

    # ------------------------------------------------------------------------------------------------------------------
    #
    def inverse_register_target(self):
        res_seg = []
        for i,slice_M in enumerate(self.target_GM_seg_M.data):
            slice = apply_ants_2D_rigid_transfo(self.appearance_model.data.mean_data, slice_M, search_reg=False ,binary=True, inverse=1, transfo_name='transfo_target_to_model_space_slice_' + str(i) + '.mat')
            res_seg.append(slice)
        return Image(param=np.asarray(res_seg))



    # ------------------------------------------------------------------------------------------------------------------
    # beta is the model similarity between all the individual images and our input image
    # beta = (1/Z)exp(-tau*square_norm(omega-omega_j))
    # Z is the partition function that enforces the constraint tha sum(beta)=1
    def compute_beta(self):
        beta = []
        tau = 0.005 #decay constant associated with the geodesic distance between a given atlas and the projected target image in model space.
        if self.coord_projected_target is not None:
            for i,coord_projected_slice in enumerate(self.coord_projected_target):
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
            return np.asarray(beta)
        else:
            raise Exception("No projected input in the appearance model")


    # ------------------------------------------------------------------------------------------------------------------
    # returns the index of the selected slices of the dataset to do label fusion and compute the graymater segmentation
    def select_K(self, epsilon=0.055): # 0.035
        selected = []
        for beta_slice in self.beta:
            selected_by_slice=[]
            for j, beta_j in enumerate(beta_slice):
                if beta_j > epsilon:
                    selected_by_slice.append(j)
            selected.append(selected_by_slice)
        return selected

    def label_fusion(self):
        res_seg_M = []
        for i, selected_by_slice in enumerate(self.selected_K):
            kept_decision_dataset = []
            for j in selected_by_slice:
                kept_decision_dataset.append(self.appearance_model.data.D_M[j])
            slice_seg = self.appearance_model.data.compute_mean_seg(kept_decision_dataset)
            res_seg_M.append(slice_seg)
        res_seg_M = np.asarray(res_seg_M)
        save_image(res_seg_M, 'res_GM_seg_model_space')
        return Image(res_seg_M)


    # ------------------------------------------------------------------------------------------------------------------
    # plot the pca and the target projection if target is provided
    def plot_omega(self):
        self.appearance_model.pca.plot_omega(nb_mode=3, target_coord=self.coord_projected_target, to_highlight=(3, self.selected_K[3])) if self.coord_projected_target is not None \
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

        self.data = Data(param=param)

        sct.printv('\nBuilding the appearance model...', verbose=param.verbose, type='normal')
        # build the appearance model
        self.appearance_model = AppearanceModel(param=param, data=self.data)

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

        #build a rigid registration
        self.rigid_reg = RigidRegistration(self.appearance_model, target_image=self.target_image)

        self.res_GM_seg = self.rigid_reg.target_GM_seg
        save_image(self.res_GM_seg.data, 'res_GM_seg')

    def show(self):

        '''
        sct.printv('\nShowing the PCA space ...')
        appearance_model.pca.show(split=param.split_data)
        '''

        sct.printv('\nPloting Omega ...')
        self.rigid_reg.plot_omega()

        '''
        sct.printv('\nShowing the projected target ...')
        self.rigid_reg.show_projected_target()
        '''






########################################################################################################################
######------------------------------------------------ FUNCTIONS -------------------------------------------------######
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
# Split a slice in two slices, used to deal with actual loss of data
def split(slice):
    left_slice = []
    right_slice = []
    column_length = slice.shape[1]
    i = 0
    for column in slice:
        if i < column_length / 2:
            left_slice.append(column)
        else:
            right_slice.insert(0, column)
        i += 1
    left_slice = np.asarray(left_slice)
    right_slice = np.asarray(right_slice)
    assert (left_slice.shape == right_slice.shape), \
        str(left_slice.shape) + '==' + str(right_slice.shape) + \
        'You should check that the first dim of your image (or slice) is an odd number'
    return left_slice, right_slice


# ----------------------------------------------------------------------------------------------------------------------
def show(coord_projected_img, pca, target):
    # Retrieving projected image from the mean image & its coordinates
    import copy

    img_reducted = copy.copy(pca.mean_image)
    for i in range(0, coord_projected_img.shape[0]):
        img_reducted += int(coord_projected_img[i][0]) * pca.W.T[i].reshape(pca.N, 1)

    if param.split_data:
        n = int(sqrt(pca.N * 2))
    else:
        n = int(sqrt(pca.N))
    if param.split_data:
        imgplot = plt.imshow(pca.mean_image.reshape(n, n / 2))
    else:
        imgplot = plt.imshow(pca.mean_image.reshape(n, n))
    imgplot.set_interpolation('nearest')
    imgplot.set_cmap('gray')
    plt.title('Mean Image')
    plt.show()
    if param.split_data:
        imgplot = plt.imshow(target.reshape(n, n / 2))
    else:
        imgplot = plt.imshow(target.reshape(n, n))
    imgplot.set_interpolation('nearest')
    #imgplot.set_cmap('gray')
    plt.title('Original Image')
    plt.show()
    if param.split_data:
        imgplot = plt.imshow(img_reducted.reshape(n, n / 2))
    else:
        imgplot = plt.imshow(img_reducted.reshape(n, n))
    imgplot.set_interpolation('nearest')
    #imgplot.set_cmap('gray')
    plt.title('Projected Image')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# save an image from an array, if the array correspond to a flatten image, the saved image will be square shaped
def save_image(im_array, im_name, path='', type='', verbose=1):
    if isinstance(im_array, list):
        n = int(sqrt(len(im_array)))
        im_data = np.asarray(im_array).reshape(n,n)
    else:
        im_data = np.asarray(im_array)
    im = Image(param=im_data,verbose=verbose)
    im.file_name = im_name
    im.ext = '.nii.gz'
    if path != '':
        im.path = path
    im.save(type=type)

def apply_ants_2D_rigid_transfo(fixed_im, moving_im, search_reg=True, apply_transfo=True, transfo_name='', binary = True, inverse=0, verbose=0):
    import time
    try:
        if 'rigidTransformations' not in os.listdir('.'):
            sct.run('mkdir ./rigidTransformations')
        dir_name = 'tmp_reg_' +str(time.time())
        sct.run('mkdir ' + dir_name, verbose=verbose)
        os.chdir('./'+ dir_name)

        if binary:
            t = 'uint8'
        else :
            t = ''

        fixed_im_name = 'fixed_im'
        save_image(fixed_im, fixed_im_name, type=t, verbose=verbose)
        moving_im_name = 'moving_im'
        save_image(moving_im, moving_im_name, type=t, verbose=verbose)

        if search_reg:
            reg_interpolation = 'BSpline'
            gradientstep = 0.3  # 0.5
            metric = 'MeanSquares'
            metric_params = ',5'
            #metric = 'MI'
            #metric_params = ',1,2'
            niter = 20
            smooth = 0
            shrink = 1
            cmd_reg = 'antsRegistration -d 2 -n ' + reg_interpolation + ' -t Rigid[' + str(gradientstep) + '] ' \
                      '-m ' + metric + '[' + fixed_im_name +'.nii.gz,' + moving_im_name + '.nii.gz ' + metric_params  + '] -o reg  -c ' + str(niter) + \
                      ' -s ' + str(smooth) + ' -f ' + str(shrink)

            sct.runProcess(cmd_reg, verbose=verbose)

            sct.run('cp reg0GenericAffine.mat ../rigidTransformations/'+transfo_name, verbose=verbose)


        if apply_transfo:
            if not search_reg:
                sct.run('cp ../rigidTransformations/'+transfo_name +  ' ./reg0GenericAffine.mat ', verbose=verbose)

            if binary:
                applyTransfo_interpolation = 'NearestNeighbor'
            else:
                applyTransfo_interpolation = 'BSpline'

            cmd_apply = 'sct_antsApplyTransforms -d 2 -i ' + moving_im_name +'.nii.gz -o ' + moving_im_name + '_moved.nii.gz ' \
                        '-n ' + applyTransfo_interpolation + ' -t [reg0GenericAffine.mat,'+ str(inverse) +']  -r ' + fixed_im_name + '.nii.gz'

            status, output = sct.runProcess(cmd_apply, verbose=verbose)

            res_im = Image(moving_im_name + '_moved.nii.gz')
    except Exception, e:
        sct.printv('WARNING: AN ERROR OCCURRED WHEN DOING RIGID REGISTRATION USING ANTs',1 ,'warning')
        print e
    else:
        sct.printv('Removing temporary files ...',verbose = verbose, type='normal')
        os.chdir('..')
        sct.run('rm -rf ' + dir_name + '/', verbose=verbose)

    if apply_transfo:
        return res_im.data

# ----------------------------------------------------------------------------------------------------------------------
# Kronecker delta function
def kronecker_delta(x, y):
    if x == y:
        return 1
    else:
        return 0


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

        if "-split" in arguments:
            param.split_data = arguments["-split"]
        if "-v" in arguments:
            param.verbose = arguments["-v"]


    asman_seg = Asman(param=param)

    asman_seg.show()
