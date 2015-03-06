#!/usr/bin/env python
########################################################################################################################
#
# Asman et al. groupwise multi-atlas segmentation method implementation
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Sara Dupont
# Modified: 2014-11-20
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

# TODO change 'target' by 'input'

#TODO: is scipy.misc.toimage really needed ?
#from scipy.misc import toimage
from msct_pca import PCA
import numpy as np
from math import sqrt
from math import exp
from msct_image import Image
from msct_parser import *
import matplotlib.pyplot as plt
import sct_utils as sct
import os



class Param:
    def __init__(self):
        self.debug = 0
        self.path_dictionary = '/Volumes/folder_shared/greymattersegmentation/data_asman/dictionary'
        #self.patient_id = ['09', '24', '30', '31', '32', '25', '10', '08', '11', '16', '17', '18']
        self.include_GM = 0
        self.split_data = 1  # this flag enables to duplicate the image in the right-left direction in order to have more dataset for the PCA
        self.verbose = 0



########################################################################################################################
######------------------------------------------------- Classes --------------------------------------------------######
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# APPEARANCE MODEL -----------------------------------------------------------------------------------------------------
class AppearanceModel:
    def __init__(self, param=None):
        if param is None:
            self.param = Param()
        else:
            self.param = param
        # Load all the images' slices from param.path_dictionary
        sct.printv('\nLoading dictionary ...', self.param.verbose, 'normal')
        self.list_atlas_seg = self.load_dictionary(self.param.split_data)
        # Construct a dataset composed of all the slices
        sct.printv('\nConstructing the data set ...', self.param.verbose, 'normal')
        dataset = self.construct_dataset()
        sct.printv("The shape of the dataset used for the PCA is {}".format(dataset.shape), verbose=self.param.verbose)
        # Instantiate a PCA object given the dataset just build
        sct.printv('\nCreating a reduced common space (using a PCA) ...', self.param.verbose, 'normal')
        self.pca = PCA(dataset, k=0.6) #WARNING : k usually is 0.8 not 0.6


    # ------------------------------------------------------------------------------------------------------------------
    # Load the dictionary:
    # each slice of each patient will be load separately with its corresponding GM segmentation
    # they will be stored as tuples in list_atlas_seg
    def load_dictionary(self, split_data):
        # init
        list_atlas_seg = []
        # loop across all the volume
        #TODO: change the name of files to find to a more general structure
        #for id in param.patient_id:
        for subject_dir in os.listdir(self.param.path_dictionary):
            subject_path = self.param.path_dictionary + '/' + subject_dir
            if os.path.isdir(subject_path):
                for file in os.listdir(subject_path):
                    if 'seg_in.nii' in file:
                        subject_seg_in = file
                    if 'GMr.nii' in file:
                        subject_GMr = file

                #atlas = Image(self.param.path_dictionary + 'errsm_' + id + '.nii.gz')
                atlas = Image(subject_path + '/' + subject_seg_in)

                if split_data:
                    if self.param.include_GM:
                        #seg = Image(self.param.path_dictionary + 'errsm_' + id + '_GMr.nii.gz')
                        seg = Image(subject_path + '/' + subject_GMr)
                        index_s = 0
                        for slice in atlas.data:
                            left_slice, right_slice = split(slice)
                            seg_slice = seg.data[index_s]
                            left_slice_seg, right_slice_seg = split(seg_slice)
                            list_atlas_seg.append((left_slice, left_slice_seg))
                            list_atlas_seg.append((right_slice, right_slice_seg))
                            index_s += 1
                    else:
                        index_s = 0
                        for slice in atlas.data:
                            left_slice, right_slice = split(slice)
                            list_atlas_seg.append((left_slice, None))
                            list_atlas_seg.append((right_slice, None))
                            index_s += 1
                else:
                    if param.include_GM:
                        #seg = Image(param.path_dictionary + 'errsm_' + id + '_GMr.nii.gz')
                        seg = Image(subject_path + '/' + subject_GMr)
                        index_s = 0
                        for slice in atlas.data:
                            seg_slice = seg.data[index_s]
                            list_atlas_seg.append((slice, seg_slice))
                            index_s += 1
                    else:
                        for slice in atlas.data:
                            list_atlas_seg.append((slice, None))
        return list_atlas_seg

    # ------------------------------------------------------------------------------------------------------------------
    # in order to build the PCA from all the J atlases, we must construct a matrix of J columns and N rows,
    # with N the dimension of flattened images
    def construct_dataset(self):
        dataset = []
        for atlas_slice in self.list_atlas_seg:
            dataset.append(atlas_slice[0].flatten())
        return np.asarray(dataset).T






# ----------------------------------------------------------------------------------------------------------------------
# RIGID REGISTRATION ---------------------------------------------------------------------------------------------------
class RigidRegistration:
    def __init__(self, appearance_model, target_image=None):
        self.appearance_model = appearance_model
        # Get the target image
        self.target = target_image
        # coord_projected_target is a list of all the coord of the target's projected slices
        sct.printv('\nProjecting the target image in the reduced common space ...', appearance_model.param.verbose, 'normal')
        self.coord_projected_target = appearance_model.pca.project(target_image) if target_image is not None else None
        self.beta = self.compute_beta()
        self.mu = []
        for beta_slice in self.beta:
            self.mu.append(appearance_model.pca.omega.dot(beta_slice))
        self.sigma = self.compute_sigma()

    # ------------------------------------------------------------------------------------------------------------------
    # beta is the model similarity between all the individual images and our input image
    # beta = (1/Z)exp(-theta*square_norm(omega-omega_j))
    # Z is the partition function that enforces the constraint tha sum(beta)=1
    def compute_beta(self):
        beta = []
        tau = 0.05 #1 #decay constant associated with the geodesic distance between a given atlas and the projected target image in model space.
        if self.coord_projected_target is not None:
            for coord_projected_slice in self.coord_projected_target:
                beta_slice = []
                # in omega matrix, each column correspond to the projection of one of the original data image,
                # the transpose operator .T enable the loop to iterate over all the images coord
                for omega_j in self.appearance_model.pca.omega.T:
                    square_norm = np.linalg.norm((omega_j - coord_projected_slice), 2)
                    print 'square_norm', square_norm
                    print 'exp(-0,5*square_norm)', exp(-theta*square_norm)
                    beta_slice.append(exp(-tau*square_norm))

                Z = sum(beta_slice)
                beta_slice = np.asarray((1/Z)*beta_slice)

            beta.append(beta_slice)
            return beta
        else:
            raise Exception("No projected input in the appearance model")

    def compute_sigma(self):
        sigma = []
        j = 0
        for beta_slice in self.beta:
            for w_v in self.appearance_model.pca.omega.T:
                sigma_slice = []
                sig = 0
                for w_j in w_v:
                    sig += beta_slice[j]*(w_j - self.mu[j])
                sigma_slice.append(sig)
            sigma.append(sigma_slice)
        return sigma

    # ------------------------------------------------------------------------------------------------------------------
    # plot the pca and the target projection if target is provided
    def plot_omega(self):
        self.pca.plot_omega(target_coord=self.coord_projected_target) if self.coord_projected_target is not None \
            else self.pca.plot_omega()

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

            # plot mean image
            # if self.param.split_data:
            #     imgplot = plt.imshow(self.pca.mean_image.reshape(n / 2, n))
            # else:
            #     imgplot = plt.imshow(self.pca.mean_image.reshape(n, n))
            # imgplot.set_interpolation('nearest')
            # imgplot.set_cmap('gray')
            # plt.title('Mean Image')
            # plt.show()
            #
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


    #TODO:
    # TODO calculate geodesic distance between the target image and the dataset
    # ------------------------------------------------------------------------------------------------------------------
    # Must return all the geodesic distances between self.appearance_model.pca.omega and the projected target image
    #def compute_geodesic_distances(self):

    # ------------------------------------------------------------------------------------------------------------------
    # Must return the geodesic distance between two arrays
    def geodesic_dist(self, array1, array2):
        print self.appearance_model.pca.omega[0]








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
    import matplotlib.pyplot as plt

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
# This little loop save projection through several pcas with different k i.e. different number of modes
def save(dataset, list_atlas_seg):
    import scipy
    import copy

    betas = [0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
    target = list_atlas_seg[8][0].flatten()
    for beta in betas:
        pca = PCA(dataset, beta)
        coord_projected_img = pca.project(target)
        img_reducted = copy.copy(pca.mean_image)
        n = int(sqrt(pca.N * 2))
        for i in range(0, coord_projected_img.shape[0]):
            img_reducted += int(coord_projected_img[i][0]) * pca.W.T[i].reshape(pca.N, 1)
        scipy.misc.imsave("/home/django/aroux/Desktop/pca_modesInfluence/" + str(pca.kept) + "modes.jpeg",
                          img_reducted.reshape(n, n / 2))

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
        parser.add_option(name="-gm",
                          type_value="int",
                          description="1 will include the gray matter data, default is 0",
                          mandatory=False,
                          example='1')
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
        fname_input = arguments["-i"]
        param.path_dictionary = arguments["-dic"]

        if "-gm" in arguments:
            param.include_GM = arguments["-gm"]
        if "-split" in arguments:
            param.split_data = arguments["-split"]
        if "-v" in arguments:
            param.verbose = arguments["-v"]

    # build the appearance model
    appearance_model = AppearanceModel(param=param)

    '''
    sct.printv('\nShowing the PCA space ...')
    appearance_model.pca.show(split=param.split_data)
    '''

    # construct target image
    target_image = Image(fname_input, split=param.split_data)

    #build a rigid registration
    rigid_reg = RigidRegistration(appearance_model, target_image=target_image)

    '''
    sct.printv('\nPloting Omega ...')
    rigid_reg.plot_omega()

    sct.printv('\nShowing the projected target ...')
    rigid_reg.show_projected_target()
    '''
    tab1 = np.asarray([1,2,3,4,5,6])
    tab2 = np.asarray([1,32,43,42,5,1])
    rigid_reg.geodesic_dist(self,tab1, tab2)
