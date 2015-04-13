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
import matplotlib.pyplot as plt


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
        self.verbose = 1


########################################################################################################################
# ----------------------------------------------------- Classes ------------------------------------------------------ #
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# MODEL DICTIONARY -----------------------------------------------------------------------------------------------------
class ModelDictionary:
    """
    Dictionary used by the supervised gray matter segmentation method
    """
    def __init__(self, param=None):
        if param is None:
            self.param = Param()
        else:
            self.param = param

        # list of the slices of the dictionary
        self.slices = []  # type: list of slices
        # number of slices
        self.J = 0  # type: int
        # dimension of the slices (flattened)
        self.N = 0  # type: int
        # list of the possible label decisions in a segmentation image (if only 1 label L=[0,1])
        self.L = []  # type: list
        # mean segmentation image of the dictionary
        self.mean_seg = None  # type: numpy array
        # mean image of the dictionary
        self.mean_image = None  # type: numpy array

        # folder containing the saved model
        self.model_dic_name = ''
        if self.param.todo_model == 'compute':
            self.model_dic_name = './gmseg_model_dictionary'
            self.compute_model_dictionary()
        elif self.param.todo_model == 'load':
            self.model_dic_name = self.param.path_dictionary # TODO change the path by the name of the dic ?? ...
            self.load_model_dictionary()

        # save all the dictionary slices
        ##### --> not used by the model, only for visualization
        if 'data_by_slice' not in os.listdir('.'):
            sct.run('mkdir ./data_by_slice')
        os.chdir('./data_by_slice')
        for j in range(self.J):
            save_image(self.slices[j].im,'slice_'+str(j) + '_im')
            save_image(self.slices[j].im_M,'slice_'+str(j) + '_registered_im')

            save_image(self.slices[j].seg,'slice_'+str(j) + '_seg')
            save_image(self.slices[j].seg_M,'slice_'+str(j) + '_registered_seg')
        os.chdir('..')

        if self.param.verbose == 2:
            self.show_data()

    # ------------------------------------------------------------------------------------------------------------------
    def compute_model_dictionary(self):
        """
        Compute the model dictionary using the provided data set
        :return:
        """
        sct.printv('\nComputing the model dictionary ...', self.param.verbose, 'normal')
        # Load all the images' slices from param.path_dictionary
        sct.printv('\nLoading data dictionary ...', self.param.verbose, 'normal')
        # List of T2star images (im) and their label decision (seg) (=segmentation of the gray matter), slice by slice

        sct.run('mkdir ' + self.model_dic_name)

        self.slices, dic_data_info = self.load_data_dictionary()

        # number of slices in the data set
        self.J = len([slice.im for slice in self.slices])
        # dimension of the data (flatten slices)
        self.N = len(self.slices[0].im.flatten())

        # inverts the segmentation slices : the model uses segmentation of the WM instead of segmentation of the GM
        self.invert_seg()
        dic_data_info = self.save_model_data(dic_data_info, 'inverted_seg')

        # set of possible labels that can be assigned to a given voxel in the segmentation
        self.L = [0, 1]  # 1=WM, 0=GM or CSF

        sct.printv('\nComputing the transformation to co-register all the data into a common groupwise space ...', self.param.verbose, 'normal')

        # list of transformation to apply to each slice to co-register the data into the common groupwise space
        coregistration_transfos = ['Affine'] #'Rigid',
        # coregistration_transfos = ['SyN']

        self.mean_seg = self.seg_coregistration(transfo_to_apply=coregistration_transfos)
        dic_data_info = self.save_model_data(dic_data_info, 'seg_M')

        sct.printv('\nCo-registering all the data into the common groupwise space ...', self.param.verbose, 'normal')

        # List of images (im_M) and their label decision (seg_M) (=segmentation of the gray matter), slice by slice in the common groupwise space
        self.coregister_data(transfo_to_apply=coregistration_transfos)

        '''
        TESTING
        '''
        self.crop_data()

        dic_data_info = self.save_model_data(dic_data_info, 'im_M')

        self.mean_image = self.compute_mean_dic_image(np.asarray([slice.im_M for slice in self.slices]))
        save_image(self.mean_image, 'mean_image', path=self.model_dic_name+'/', type='uint8')

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
        :return:
        """
        # initialization
        slices = []
        j = 0
        data_info = {}
        #TODO: change the name of files to find to a more general structure
        for subject_dir in os.listdir(self.param.path_dictionary):
            subject_path = self.param.path_dictionary + '/' + subject_dir
            if os.path.isdir(subject_path):
                data_info[subject_dir] = {'n_slices':0,'inverted_seg':[],'im_M':[], 'seg_M':[]}

                subject_seg_in = ''
                subject_gm_seg = ''
                sct.run('mkdir ' + self.model_dic_name + '/' + subject_dir, verbose=self.param.verbose)
                for file_name in os.listdir(subject_path):
                    if 'GM' in file_name or 'gmseg' in file_name:
                        subject_gm_seg = file_name
                        sct.run('cp ./' + self.param.path_dictionary + subject_dir + '/' + file_name + ' ' + self.model_dic_name + '/' + subject_dir + '/' + subject_dir + '_seg.nii.gz')
                    if 'seg_in' in file_name and 'gm' not in file_name.lower():
                        subject_seg_in = file_name
                        sct.run('cp ./' + self.param.path_dictionary + subject_dir + '/' + file_name + ' ' + self.model_dic_name + '/' + subject_dir + '/' + subject_dir + '_im.nii.gz')

                im = Image(subject_path + '/' + subject_seg_in)
                seg = Image(subject_path + '/' + subject_gm_seg)

                for im_slice, seg_slice in zip(im.data, seg.data):
                    data_info[subject_dir]['n_slices'] += 1
                    if self.param.split_data:
                        left_slice, right_slice = split(im_slice)

                        left_slice_seg, right_slice_seg = split(seg_slice)

                        slices.append(Slice(id=j, im=left_slice, seg=left_slice_seg))
                        slices.append(Slice(id=j+1, im=right_slice, seg=right_slice_seg))
                        j += 2

                    else:
                        slices.append(Slice(id=j, im=im_slice, seg=seg_slice))
                        j += 1

        return np.asarray(slices), data_info

    # ------------------------------------------------------------------------------------------------------------------
    def save_model_data(self,data_info,what_to_save):
        """
        save images of the model dictionary
        :param data_info: dictionary containing information about the data
        :param what_to_save: type of data to be saved
        :return:
        """
        suffix = ''
        data_to_save = []
        total_n_slices = 0
        if what_to_save == 'inverted_seg':
            suffix = '_seg'
            data_to_save = [slice.seg for slice in self.slices]
        elif what_to_save == 'im_M':
            suffix = '_im_model_space'
            data_to_save = [slice.im_M for slice in self.slices]
        elif what_to_save == 'seg_M':
            suffix = '_seg_model_space'
            data_to_save = [slice.seg_M for slice in self.slices]

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
    def invert_seg(self):
        """
        Invert the gray matter segmentation to get segmentation of the white matter instead
        keeps more information, theoretically better results
        :return:
        """
        for slice in self.slices:
            im_a = Image(param=slice.im)
            sc = im_a.copy()
            nz_coord_sc = sc.getNonZeroCoordinates()
            im_d = Image(param=slice.seg)
            nz_coord_d = im_d.getNonZeroCoordinates()
            for coord in nz_coord_sc:
                sc.data[coord.x, coord.y] = 1
            for coord in nz_coord_d:
                im_d.data[coord.x, coord.y] = 1
            # cast of the -1 values (-> GM pixel at the exterior of the SC pixels) to +1 --> WM pixel
            inverted_slice_decision = np.absolute(sc.data - im_d.data).astype(int)
            slice.set(seg=inverted_slice_decision)

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
        current_mean_seg = self.compute_mean_seg(np.asarray([slice.seg for slice in self.slices]))
        for transfo in transfo_to_apply:
            sct.printv('Doing a ' + transfo + ' registration of each segmentation slice to the mean segmentation ...', self.param.verbose, 'normal')
            current_mean_seg = self.find_coregistration(mean_seg=current_mean_seg, transfo_type=transfo)

        '''
        fig=plt.figure()
        plt.imshow(current_mean_seg)
        plt.title('Current mean segmentation ...')
        plt.plot()

        fig=plt.figure()
        plt.imshow(self.slices[0].seg_M)
        plt.title('slice 0...')
        plt.plot()


        mean_white_pix = self.nb_w_pixels() #mean number of white pixels in the manual segmentation slices of the dictionary
        print mean_white_pix


        i=1
        while np.sum(current_mean_seg) < 0.8*mean_white_pix:
            print '--> Affine registration number ',i
            print 'number of white pixels in the current mean seg', np.sum(current_mean_seg)
            current_mean_seg = self.compute_mean_seg(np.asarray([slice.seg_M for slice in self.slices]))
            current_mean_seg = self.find_coregistration(mean_seg=current_mean_seg, transfo_type='Affine')
            i+=10
        #current_mean_seg = self.compute_mean_seg(np.asarray([slice.seg_M for slice in self.slices]))
        current_mean_seg = self.find_coregistration(mean_seg=current_mean_seg, transfo_type='Affine')
        '''
        resulting_mean_seg = current_mean_seg

        return resulting_mean_seg

    '''
    def nb_w_pixels(self):
        s_w_pix = 0
        for slice in self.slices:
            s_w_pix += np.sum(slice.seg)
        return s_w_pix/self.J
    '''

    # ------------------------------------------------------------------------------------------------------------------
    def find_coregistration(self, mean_seg=None, transfo_type='Rigid'):
        """
        For each segmentation slice, apply and save a registration of the specified type of transformation
        the name of the registration file (usually a matlab matrix) is saved in self.RtoM
        :param mean_seg: current mean segmentation
        :param transfo_type: type of transformation for the registration
        :return mean seg: updated mean segmentation
        """
        # initialization
        for slice in self.slices:
            name_j_transform = 'transform_slice_' + str(slice.id) + '.mat'
            slice.set(RtoM=name_j_transform)
            decision_M = apply_ants_transfo(mean_seg, slice.seg,  transfo_name=name_j_transform, path=self.model_dic_name + '/', transfo_type=transfo_type)
            slice.set(seg_M=decision_M.astype(int))
            slice.set(seg_M_flat=decision_M.flatten().astype(int))
        mean_seg = self.compute_mean_seg([slice.seg_M for slice in self.slices])

        save_image(mean_seg, 'mean_seg', path=self.model_dic_name+'/', type='uint8')

        return mean_seg

    # ------------------------------------------------------------------------------------------------------------------
    def compute_mean_seg(self, seg_data_set):
        """
        Compute the mean segmentation image for a given segmentation data set seg_data_set
        :param seg_data_set:
        :return:
        """
        mean_seg = []
        choose_maj_vote = {}
        for l in self.L:
            to_be_summed = []
            for slice in seg_data_set:
                consistent_vox = []
                for row in slice:
                    for i in row:
                        try:
                            if i > 0.2:
                                i = 1
                        except ValueError:
                            print 'Value Error with i = ', i
                            print 'Dataset was : \n', seg_data_set
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
    def compute_mean_dic_image(self, im_data_set):
        """
        Compute the mean image of the dictionary
        Used to co-register the dictionary images into teh common groupwise space
        :param im_data_set:
        :return:
        """
        sum = np.zeros(im_data_set[0].shape)
        for slice_im in im_data_set:
            for k,row in enumerate(slice_im):
                sum[k] = np.add(sum[k],row)
        mean = sum/self.J
        return mean

    # ------------------------------------------------------------------------------------------------------------------
    def coregister_data(self,  transfo_to_apply=None):
        """
        Apply to each image slice of the dictionary the transformations found registering the segmentation slices
        The co_registered images are saved for each slice as im_M
        :param transfo_to_apply: list of string
        :return:
        """
        list_im = [slice.im for slice in self.slices]
        for slice in self.slices:
            for transfo in transfo_to_apply:
                im_M = apply_ants_transfo(self.compute_mean_dic_image(list_im), slice.im, search_reg=False, transfo_name=slice.RtoM, binary=False, path=self.model_dic_name+'/', transfo_type=transfo)
                #apply_2D_rigid_transformation(self.im[j], self.RM[j]['tx'], self.RM[j]['ty'], self.RM[j]['theta'])
            slice.set(im_M=im_M)
            slice.set(im_M_flat=im_M.flatten())

    '''
    def crop_data_new_ellipse(self):
        #croping the im_M images

        #mean_seg dimensions :
        #height
        down = True
        above = False
        height_min = 0
        height_max = 0
        for h,row in enumerate(self.mean_seg.T):
            if sum(row) == 0:
                if down :
                    height_min = h
                elif not above:
                    height_max = h
                    above = True
            else:
                down = False
        height_min += 1
        height_max -= 1

        #width
        left = True
        right = False
        width_min = 0
        width_max = 0
        for w,row in enumerate(self.mean_seg):
            if sum(row) == 0:
                if left :
                    width_min = w
                elif not right:
                    width_max = w
                    right = True
            else:
                left = False
        width_min += 1
        width_max -= 1

        a = (width_max - width_min)/2.0
        b = (height_max - height_min)/2.0

        range_x = np.asarray(range(int(-10*a),int(10*a)+1)) /10.0
        top_points = [(b * sqrt(1-(x**2/a**2)),x) for x in range_x]
        n = int(sqrt(self.N))

        top_points.sort()

        ellipse_mask = np.zeros((n,n))
        done_rows = []
        for point in top_points:
            y_plus = int(round((n/2)+point[0]))
            y_minus = int(round((n/2)-point[0]))
            print 'y_plus', y_plus, 'y_minus', y_minus
            if y_plus not in done_rows:
                x_plus = int(round((n/2)+abs(point[1])))
                x_minus = int(round((n/2)-abs(point[1])))
                for x in range(x_minus, x_plus+1):
                    ellipse_mask[x, y_plus] = 1
                    ellipse_mask[x, y_minus] = 1
                done_rows.append(y_plus)


        print 'axis', a, b
        print 'done_rows', done_rows
        print 'center : ', int(n/2)
        print ellipse_mask
        save_image(ellipse_mask, 'ellipse_mask', path=self.model_dic_name+'/', type='uint8')
        '''

    # ------------------------------------------------------------------------------------------------------------------
    def crop_data(self):
        """
        Crop the model images (im_M) to an ellipse shape to get rid of the size/shape variability between slices
        :return:
        """
        im_mean_seg = Image(param=self.mean_seg)

        nz_coord = im_mean_seg.getNonZeroCoordinates()
        nz_coord_dic = {}
        for coord in nz_coord:
            nz_coord_dic[coord.x] = []
        for coord in nz_coord:
            nz_coord_dic[coord.x].append(coord.y)

        ellipse_mask = im_mean_seg.copy().data

        for x, y_list in nz_coord_dic.items():
            full_y_list = range(min(y_list), max(y_list)+1)
            if y_list != full_y_list:
                for y in full_y_list:
                    ellipse_mask[x, y] = 1

        save_image(ellipse_mask, 'ellipse_mask', path=self.model_dic_name+'/', type='uint8')

        for slice in self.slices:
            new_im_m = np.einsum('ij,ij->ij', ellipse_mask, slice.im_M)
            slice.set(im_M=new_im_m)

    # ------------------------------------------------------------------------------------------------------------------
    # END OF FUNCTIONS USED TO COMPUTE THE MODEL
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def load_model_dictionary(self):
        """
        Load the model dictionary from a saved one
        :return:
        """

        sct.printv('\nLoading the model dictionary ...', self.param.verbose, 'normal')

        j = 0
        for subject_dir in os.listdir(self.model_dic_name):
            subject_path = self.model_dic_name + subject_dir
            if os.path.isdir(subject_path) and 'transformations' not in subject_path:
                subject_im = ''
                subject_seg = ''
                subject_im_m = ''
                subject_seg_m = ''

                for file_name in os.listdir(subject_path):
                    if '_im.nii' in file_name:
                        subject_im = file_name
                    if '_seg.nii' in file_name:
                        subject_seg = file_name
                    if '_im_model_space.nii' in file_name:
                        subject_im_m = file_name
                    if '_seg_model_space.nii' in file_name:
                        subject_seg_m = file_name

                im = Image(subject_path + '/' + subject_im)
                seg = Image(subject_path + '/' + subject_seg)
                im_m = Image(subject_path + '/' + subject_im_m)
                seg_m = Image(subject_path + '/' + subject_seg_m)

                for im_slice, seg_slice, im_m_slice, seg_m_slice in zip(im.data, seg.data, im_m.data, seg_m.data):
                    if self.param.split_data:
                        left_slice, right_slice = split(im_slice)
                        left_slice_seg, right_slice_seg = split(seg_slice)
                        left_slice_m, right_slice_m = split(im_m_slice)
                        left_slice_seg_m, right_slice_seg_m = split(seg_m_slice)

                        self.slices.append(Slice(id=j, im=left_slice, seg=left_slice_seg, im_M=left_slice_m, seg_M=left_slice_seg_m, im_M_flat=left_slice_m.flatten(), seg_M_flat=left_slice_seg_m.flatten(), RtoM='transform_slice_' + str(j) + '.mat'))
                        self.slices.append(Slice(id=j+1, im=right_slice, seg=right_slice_seg, im_M=right_slice_m, seg_M=right_slice_seg_m, im_M_flat=right_slice_m.flatten(), seg_M_flat=right_slice_seg_m.flatten(), RtoM='transform_slice_' + str(j+1) + '.mat'))
                        j += 2

                    else:
                        self.slices.append(Slice(id=j, im=im_slice, seg=seg_slice, im_M=im_m_slice, seg_M=seg_m_slice, im_M_flat=im_m_slice.flatten(), seg_M_flat=seg_m_slice.flatten(), RtoM='transform_slice_' + str(j) + '.mat'))
                        j += 1

        # number of atlases in the dictionary
        self.J = len(self.slices)  # len([slice.im for slice in self.slices])

        # dimension of the data (flatten slices)
        self.N = len(self.slices[0].im.flatten())

        # set of possible labels that can be assigned to a given pixel in the segmentation
        self.L = [0, 1]  # 1=WM, 0=GM or CSF

        self.mean_image = Image(self.model_dic_name + 'mean_image.nii.gz').data
        self.mean_seg = Image(self.model_dic_name + 'mean_seg.nii.gz').data

    # ------------------------------------------------------------------------------------------------------------------
    def show_data(self):
        """
        show the 10 first slices of the model dictionary
        :return:
        """
        for slice in self.slices[:10]:
            fig = plt.figure()

            seg_subplot = fig.add_subplot(2,3, 1)
            seg_subplot.set_title('Original space - seg')
            im_seg = seg_subplot.imshow(slice.seg)
            im_seg.set_interpolation('nearest')
            im_seg.set_cmap('gray')

            seg_m_subplot = fig.add_subplot(2,3, 2)
            seg_m_subplot.set_title('Common groupwise space - seg')
            im_seg_m = seg_m_subplot.imshow(slice.seg_M)
            im_seg_m.set_interpolation('nearest')
            im_seg_m.set_cmap('gray')

            mean_seg_subplot = fig.add_subplot(2,3, 3)
            mean_seg_subplot.set_title('Mean seg')
            im_mean_seg = mean_seg_subplot.imshow(np.asarray(self.mean_seg))
            im_mean_seg.set_interpolation('nearest')
            im_mean_seg.set_cmap('gray')

            slice_im_subplot = fig.add_subplot(2,3, 4)
            slice_im_subplot.set_title('Original space - data ')
            im_slice_im = slice_im_subplot.imshow(slice.im)
            im_slice_im.set_interpolation('nearest')
            im_slice_im.set_cmap('gray')

            slice_im_m_subplot = fig.add_subplot(2,3, 5)
            slice_im_m_subplot.set_title('Common groupwise space - data ')
            im_slice_im_m = slice_im_m_subplot.imshow(slice.im_M)
            im_slice_im_m.set_interpolation('nearest')
            im_slice_im_m.set_cmap('gray')

            plt.suptitle('Slice ' + str(slice.id))
            plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# MODEL ---------------------------------------------------------------------------------------------------------------
class Model:
    """
    Model used by the supervised gray matter segmentation method

    """
    def __init__(self, param=None, dictionary=None, k=0.8):
        if param is None:
            self.param = Param()
        else:
            self.param = param

        self.dictionary = dictionary

        sct.printv("The shape of the dictionary used for the PCA is (" + str(self.dictionary.N) + ',' + str(self.dictionary.J) + ')', verbose=self.param.verbose)

        # Instantiate a PCA object given the dictionary just build
        sct.printv('\nCreating a reduced common space (using a PCA) ...', self.param.verbose, 'normal')

        '''
        print '######################################################################################################\n' \
              '######################################################################################################\n' \
              'TODO MODEL : ' + self.dictionary.param.todo_model+ \
              ' \n######################################################################################################\n' \
              '######################################################################################################\n' \
              'seg_M_FLAT :\n ', self.dictionary.slices[0].seg_M_flat, '\nseg_M NOT FLAT :\n', self.dictionary.slices[0].seg_M
        '''

        if self.param.todo_model == 'compute':
            self.pca = PCA(np.asarray([slice.im_M_flat for slice in self.dictionary.slices]).T, k=k)
            self.dictionary.mean_image = self.pca.mean_image

            save_image(self.pca.mean_image, 'mean_image', path=self.dictionary.model_dic_name+'/')#, type='uint8')
            self.pca.save_data(self.dictionary.model_dic_name)

        elif self.param.todo_model == 'load':
            pca_mean_data, pca_eig_pairs = self.load_pca_file()

            self.pca = PCA(np.asarray([slice.im_M_flat for slice in self.dictionary.slices]).T, mean_vect=pca_mean_data, eig_pairs=pca_eig_pairs, k=k)

    # ----------------------------------------------------------------------------------------------------------------------
    def load_pca_file(self, file_name='data_pca.txt'):
        """
        Load a PCA from a text file containing the appropriate information (previously saved)
        :param file_name: name of the PCA text file
        :return:
        """
        fic_data_pca = open(self.param.path_dictionary + file_name, 'r')
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
            for i,v in enumerate(eig_vect_str):
                 if v != '' and v != '\n':
                    eig_vect.append(float(v))
            eig_pairs_vect.append((float(eig_val_str), eig_vect))

        return mean_data_vect, eig_pairs_vect


# ----------------------------------------------------------------------------------------------------------------------
# TARGET SEGMENTATION ---------------------------------------------------------------------------------------------------
class TargetSegmentation:
    """
    Contains all the function to segment the gray matter an a target image given a model
        - registration of the target to the model space
        - projection of the target slices on the reduced model space
        - selection of the model data most similar to the target slices

    """
    def __init__(self, model, target_image=None, tau=None):
        self.model = model

        # Get the target image
        self.target = target_image
        '''
        save_image(self.target.data, 'target_image')
        print '---TARGET IMAGE IN CLASS TargetSegmentation : ', self.target.data
        save_image(self.target.data[0],'target_slice0_targetSeg_'+self.model.param.todo_model)
        '''

        self.target_M = self.register_target_to_model_space()
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

        self.epsilon = round(1.0/self.model.dataset.J,4) - 0.001
        print 'epsilon : ', self.epsilon

        if tau is None :
            self.tau = self.compute_tau()
        else:
            self.tau = tau
        self.beta = self.compute_beta(self.coord_projected_target, tau=self.tau)
        #print '----------- BETAS :', self.beta
        #self.beta = self.compute_beta(self.coord_projected_target, tau=0.00114)

        sct.printv('\nSelecting the atlases closest to the target ...', model.param.verbose, 'normal')

        self.selected_K = self.select_K(self.beta)
        #print '----SELECTED K -----', self.selected_K
        #print '----SHAPE SELECTED K -----', self.selected_K.shape
        sct.printv('\nComputing the result gray matter segmentation ...', model.param.verbose, 'normal')
        self.target_GM_seg_M = self.label_fusion(self.selected_K)
        self.target_GM_seg = self.inverse_register_target()

    # ------------------------------------------------------------------------------------------------------------------
    #
    def register_target_to_model_space(self):
        target_M = []
        for i,slice in enumerate(self.target.data):
            #slice_M = apply_ants_transfo(self.model.dictionary.mean_image, slice, binary=False, transfo_type='Affine', transfo_name='transfo_target_to_model_space_slice_' + str(i) + '.mat')
            n = int(sqrt(self.model.pca.N))
            mean_vect = self.model.pca.mean_data_vect.reshape(len(self.model.pca.mean_data_vect),)
            im = mean_vect.reshape(n, n).astype(np.float)
            slice_M = apply_ants_transfo(im, slice, binary=False, transfo_type='Affine', transfo_name='transfo_target_to_model_space_slice_' + str(i) + '.mat')
            target_M.append(slice_M)
        return Image(param=np.asarray(target_M))

    # ------------------------------------------------------------------------------------------------------------------
    #
    def inverse_register_target(self):
        res_seg = []
        for i,slice_M in enumerate(self.target_GM_seg_M.data):
            #slice = apply_ants_transfo(self.model.dictionary.mean_image, slice_M, search_reg=False ,binary=True, inverse=1, transfo_type='Affine', transfo_name='transfo_target_to_model_space_slice_' + str(i) + '.mat')
            n = int(sqrt(self.model.pca.N))
            mean_vect = self.model.pca.mean_data_vect.reshape(len(self.model.pca.mean_data_vect),)
            im = mean_vect.reshape(n, n).astype(np.float)
            slice = apply_ants_transfo(im, slice_M, search_reg=False ,binary=True, inverse=1, transfo_type='Affine', transfo_name='transfo_target_to_model_space_slice_' + str(i) + '.mat')
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
                for omega_j in self.model.pca.omega.T:
                    square_norm = np.linalg.norm((coord_projected_slice - omega_j), 2)
                    beta_slice.append(exp(-tau*square_norm))

                Z = sum(beta_slice)
                '''
                print 'beta case 1 :', beta
                print '--> sum beta ', Z
                '''
                for i, b in enumerate(beta_slice):
                    beta_slice[i] = (1/Z) * b

                beta.append(beta_slice)
        else:
            # in omega matrix, each column correspond to the projection of one of the original data image,
            # the transpose operator .T enable the loop to iterate over all the images coord
            for omega_j in self.model.pca.omega.T:
                square_norm = np.linalg.norm((coord_target - omega_j), 2)
                beta.append(exp(-tau*square_norm))

            Z = sum(beta)
            '''
            print 'beta case 2 :', beta
            print '--> sum beta ', Z
            '''
            for i, b in enumerate(beta):
                beta[i] = (1/Z) * b

        return np.asarray(beta)


    # ------------------------------------------------------------------------------------------------------------------
    # decay constant associated with the geodesic distance between a given atlas and the projected target image in model space.
    def compute_tau(self):
        sct.printv('\nComputing Tau ... \n'
                   '(Tau is a parameter indicating the decay constant associated with a geodesic distance between a given atlas and a projected target image, see Asman paper, eq (16))', 1, 'normal')
        from scipy.optimize import minimize
        def to_minimize(tau):
            sum_norm = 0
            for slice in self.model.dataset.slices:
                Kj = self.select_K(self.compute_beta(self.model.pca.project_array(slice.im_M_flat), tau=tau)) #in project : Image(param=Amj)
                est_dmj = self.label_fusion(Kj)
                sum_norm += l0_norm(slice.seg_M, est_dmj.data)
            return sum_norm
        est_tau = minimize(to_minimize, 0, method='Nelder-Mead', options={'xtol':0.0005})
        sct.printv('Estimated tau : ' + str(est_tau.x[0]))
        if self.model.param.todo_model == 'compute':
            fic = open(self.model.dataset.model_dic_name + '/tau.txt','w')
            fic.write(str(est_tau.x[0]))
            fic.close()
        return float(est_tau.x[0])




    # ------------------------------------------------------------------------------------------------------------------
    # returns the index of the selected slices of the dictionary to do label fusion and compute the graymater segmentation
    def select_K(self, beta):#0.015
        selected = []
        #print '---------- IN SELECT_K : shape beta ---------------->', beta.shape, ' len = ', len(beta.shape)
        #print '---------- IN SELECT_K : type beta[0] ---------------->', type(beta[0])
        if isinstance(beta[0], (list, np.ndarray)):
            for beta_slice in beta:
                selected_by_slice=[]
                for j, beta_j in enumerate(beta_slice):
                    if beta_j > self.epsilon:
                        selected_by_slice.append(j)
                # selected.append(np.asarray(selected_by_slice))
                selected.append(selected_by_slice)
        else:
            for j, beta_j in enumerate(beta):
                if beta_j > self.epsilon:
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
                    kept_decision_dataset.append(self.model.dataset.slices[j].seg_M)
                slice_seg = self.model.dataset.compute_mean_seg(kept_decision_dataset)
                res_seg_M.append(slice_seg)
        else:
            kept_decision_dataset = []
            for j in selected_K:
                kept_decision_dataset.append(self.model.dataset.slices[j].seg_M)
            slice_seg = self.model.dataset.compute_mean_seg(kept_decision_dataset)
            res_seg_M = slice_seg

        res_seg_M = np.asarray(res_seg_M)
        #save_image(res_seg_M, 'res_GM_seg_model_space')

        return Image(res_seg_M)


    # ------------------------------------------------------------------------------------------------------------------
    # plot the pca and the target projection if target is provided
    def plot_omega(self,nb_modes=3):
        self.model.pca.plot_omega(nb_mode=nb_modes, target_coord=self.coord_projected_target) if self.coord_projected_target is not None \
            else self.model.pca.plot_omega()
        self.model.pca.plot_omega(nb_mode=nb_modes, target_coord=self.coord_projected_target, to_highlight=(5, self.selected_K[5])) if self.coord_projected_target is not None \
            else self.model.pca.plot_omega()

    # ------------------------------------------------------------------------------------------------------------------
    def show_projected_target(self):
        # Retrieving projected image from the mean image & its coordinates
        import copy

        index = 0
        fig1 = plt.figure()
        fig2 = plt.figure()
        # loop across all the projected slices coord
        for coord in self.coord_projected_target:
            img_reducted = copy.copy(self.model.pca.mean_data_vect)
            # loop across coord and build projected image
            for i in range(0, coord.shape[0]):
                img_reducted += int(coord[i][0]) * self.model.pca.W.T[i].reshape(self.model.pca.N, 1)

            if self.model.param.split_data:
                n = int(sqrt(self.model.pca.N * 2))
            else:
                n = int(sqrt(self.model.pca.N))

            # Plot original image
            orig_ax = fig1.add_subplot(10, 3, index)
            orig_ax.set_title('original slice {} '.format(index))
            if self.model.param.split_data:
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
            if self.model.param.split_data:
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

        self.dataset = ModelDictionary(param=param)

        sct.printv('\nBuilding the appearance model...', verbose=param.verbose, type='normal')
        # build the appearance model
        self.appearance_model = Model(param=param, dataset=self.dataset, k=0.8) #WARNING : K IS USUALLY 0.8

        sct.printv('\nConstructing target image ...', verbose=param.verbose, type='normal')
        # construct target image
        self.target_image = Image(param.target_fname)

        #print '---TARGET IMAGE IN CLASS ASMAN : ', self.target_image.data
        #save_image(self.target_image.data[0],'target_slice0_asman_'+param.todo_model)


        if param.split_data:
            splited_target = []
            for slice in self.target_image.data:
                left_slice, right_slice = split(slice)
                splited_target.append(left_slice)
                splited_target.append(right_slice)
            self.target_image = Image(np.asarray(splited_target))

        tau = None #0.000765625 #0.00025 #0.000982421875 #0.00090625 #None

        if param.todo_model == 'load' :
            fic = open(self.appearance_model.dictionary.model_dic_name + '/tau.txt','r')
            tau = float(fic.read())
            fic.close()


        #build a rigid registration
        self.rigid_reg = TargetSegmentation(self.appearance_model, target_image=self.target_image, tau=tau)

        self.res_GM_seg = self.rigid_reg.target_GM_seg
        name_res = sct.extract_fname(param.target_fname)[1] + '_graymatterseg'
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
                          description="1 will split all images from dictionary in the right-left direction in order to have more dictionary for the PCA",
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

