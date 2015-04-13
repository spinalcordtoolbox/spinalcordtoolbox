#!/usr/bin/env python
########################################################################################################################
#
# Implementation of Principal Component Analysis using scatter matrix inspired by Sebastian Raschka
# http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#sc_matrix
#
#
# Step 1: Take the whole dictionary consisting of J N-dimensional flattened images [NxJ], N=nb of vox
#
# Step 2: Compute the mean image: PSI (called mean_data_vect in the code)
#
# Step 3: Compute the covariance matrix of the dictionary
#
# Step 4: Compute the eigenvectors and corresponding eigenvalues (from the covariance matrix)
#
# Step 5: Sort the eigenvectors by decreasing eigenvalues and choose k in order to keep V of them such as:
#       sum(kept eigenvalues)/sum(all eigenvalues) < k (kappa in asman article)
#           This gives W, a NxV matrix with N=nb_vox and V=nb of kept eigenvectors
#
# Step 6: Transform the input image onto the new subspace, this can be done by:
#       y = W.T*(x - mean)  where:
#           x is the input N*1 flatened image
#           .T: transpose operator
#           W [NxV]
#           y is x projected in the PCA space
#
#   TODO: add datashape arg in __init__ in order to remove if split etc...
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux
# Modified: 2014-12-05
#
# About the license: see the file LICENSE.TXT
########################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
import sct_utils as sct
from math import sqrt
import os


class PCA:

    def __init__(self, dataset, mean_vect=None, eig_pairs=None, k=0.80):
        # STEP 1
        self.dataset = dataset  # This should be a J*N dimensional matrix of J N-dimensional flattened images
        self.N = dataset.shape[0]  # The number of rows is the dimension of flattened images
        self.J = dataset.shape[1]  # The number of columns is the number of images
        # STEP 2
        if mean_vect is not None:
            self.mean_data_vect = mean_vect
        else:
            self.mean_data_vect = self.mean()


        n = int(sqrt(self.N))
        self.mean_image = self.mean_data_vect.reshape(n,n)

        fig=plt.figure()
        plt.imshow(self.mean_image.astype(np.float))
        plt.plot()

        if eig_pairs == None:
            # STEP 3
            self.covariance_matrix = self.covariance_matrix()
            # STEP 4 eigpairs consist of a list of tuple (eigenvalue, eigenvector) already sorted by decreasing eigenvalues
            self.eig_pairs = self.sorted_eig()
        else:
             # STEP 3
            self.covariance_matrix = None
            # STEP 4 eigpairs consist of a list of tuple (eigenvalue, eigenvector) already sorted by decreasing eigenvalues
            self.eig_pairs = eig_pairs
        # STEP 5
        self.k = k
        self.W, self.kept = self.generate_W(modes_to_ignore=0)
        print '\n\n-------> IN PCA : '
        print '\n-> W:', self.W
        print '\n-> kept:', self.kept
        # omega is a matrix of k rows and J columns, each columns correspond to a vector projection of an image from
        # the dictionary
        self.omega = self.project_dataset()

    # STEP 2
    def mean(self):
        mean_im = []
        for row in self.dataset:
            m = sum(row)/self.J
            mean_im.append(m)
        mean_im = np.array([mean_im]).T
        return mean_im

    # STEP 3
    def covariance_matrix(self):
        covariance_matrix = np.zeros((self.N, self.N))
        for j in range(0, self.J):
            covariance_matrix += float(1)/self.J*(self.dataset[:, j].reshape(self.N, 1) - self.mean_data_vect)\
                .dot((self.dataset[:, j].reshape(self.N, 1) - self.mean_data_vect).T)
        return covariance_matrix

    # STEP 4
    def sorted_eig(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix)

        eigenvectors = eigenvectors.astype(np.float)

        # Create a list of (eigenvalue, eigenvector) tuple
        eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))
                     if np.abs(eigenvalues[i]) > 0.0000001]
        # Sort the (eigenvalue, eigenvector) tuples from high to low

        def getKey(item):
            return item[0] #sorting by eigenvalues
        eig_pairs = sorted(eig_pairs, key=getKey, reverse=True)
        return eig_pairs

    # STEP 5
    def generate_W(self, modes_to_ignore=0):
        eigenvalues_kept = []
        s = sum([eig[0] for eig in self.eig_pairs])
        print '\n####################################\n ---> sum of eigenvalues : ', s
        first = 1
        start=modes_to_ignore
        for eig in self.eig_pairs[start:]:
            if first:
                W = np.asarray(eig[1]).reshape(self.N, 1)
                eigenvalues_kept.append(eig[0])
                first = 0
            else:
                if (sum(eigenvalues_kept) + eig[0])/s <= self.k:
                    eigenvalues_kept.append(eig[0])
                    W = np.hstack((W, np.asarray(eig[1]).reshape(self.N, 1)))
                else:
                    break
        kept = len(eigenvalues_kept)
        print 'kept eigenvalues (PCA space dimension)  : ', kept
        return W, kept

    # STEP 6
    # Project the all dictionary in order to get images "close" to the target & to plot the dictionary
    def project_dataset(self):
        omega = []
        for column in self.dataset.T:
            omega.append(self.project_array(column).reshape(self.kept,))
        return np.asarray(omega).T

    def project(self, target):
        # check if the target is a volume or a flat image
        coord_projected_slices = []
        # loop across all the slice
        for slice in target.data:
            coord_projected_slices.append(self.project_array(slice.flatten()))
        return np.asarray(coord_projected_slices)

    def project_array(self, target_as_array):
        if target_as_array.shape == (self.N,):
            target = target_as_array.reshape(self.N, 1)
            coord_projected_img = self.W.T.dot(target - self.mean_data_vect)
            return coord_projected_img
        else:
            print "target dimension is {}, must be {}.\n".format(target_as_array.shape, self.N)


    def save_data(self, path):
        previous_path = os.getcwd()
        os.chdir(path)
        fic_data = open('data_pca.txt', 'w')
        for i,m in enumerate(self.mean_data_vect):
            if i == len(self.mean_data_vect) - 1:
                fic_data.write(str(m[0]))
            else:
                fic_data.write(str(m[0]) + ' , ')
        fic_data.write('\n')
        for i,eig in enumerate(self.eig_pairs):
            eig_vect_string = ''
            for v in eig[1]:
                eig_vect_string += str(v) + ' '
            if i == len(self.eig_pairs) - 1:
                fic_data.write(str(eig[0]) + ' ; ' + eig_vect_string)
            else:

                fic_data.write(str(eig[0]) + ' ; ' + eig_vect_string + ' , ')
        fic_data.write('\n')
        fic_data.close()
        os.chdir(previous_path)


    #
    # Show all the mode
    def show_all_modes(self, split=0):
        from math import sqrt
        if split:
            n = int(sqrt(2*self.N))
        else:
            n = int(sqrt(self.N))
        fig = plt.figure()
        for i_fig in range(0, self.kept):
            eigen_V = self.W.T[i_fig, :]
            #dimensions of the subfigure
            x = int(sqrt(self.kept))
            y = int(self.kept/x)
            x += 1
            a = fig.add_subplot(x, y, i_fig + 1)
            a.set_title('Mode {}'.format(i_fig))
            if split:
                #TODO: check if casting complex values to float isn't a too big loss of information ...
                imgplot = a.imshow(eigen_V.reshape(n/2, n).astype(np.float))
            else:
                imgplot = a.imshow(eigen_V.reshape(n, n).astype(np.float))
            imgplot.set_interpolation('nearest')
            imgplot.set_cmap('gray')
        plt.show()

        #
    # Show one mode
    def show_mode(self, mode=0, split=0):
        from math import sqrt
        if split:
            n = int(sqrt(2*self.N))
        else:
            n = int(sqrt(self.N))
        fig = plt.figure()
        eigen_V = self.W.T[mode, :]
        eigen_value = self.eig_pairs[mode][0]
        mean_vect = self.mean_data_vect.reshape(len(self.mean_data_vect),)


        minus_3vect = mean_vect -  3 * eigen_value *eigen_V


        print '\n\n-------> minus_3vect min :', min(minus_3vect), ' max :', max(minus_3vect)
        print 'mean  min :', min(mean_vect), ' max : ', max(mean_vect)
        print 'eig  min :', min(eigen_V), ' max : ', max(eigen_V)


        print 'eig Val', self.eig_pairs[mode][0]

        plot_minus3 = fig.add_subplot(1, 5, 1)
        plot_minus3.set_title('Mean - 3 lambda * eigen vector')
        if split:
            #TODO: check if casting complex values to float isn't a too big loss of information ...
            im_minus3 = plot_minus3.imshow(minus_3vect.reshape(n/2, n).astype(np.float))
        else:
            im_minus3 = plot_minus3.imshow(minus_3vect.reshape(n, n).astype(np.float))
        im_minus3.set_interpolation('nearest')
        im_minus3.set_cmap('gray')

        minus_3vect /= (max(minus_3vect) - min(minus_3vect))


        minus_vect = mean_vect - eigen_value *eigen_V

        minus_vect /= (max(minus_vect) - min(minus_vect))
        '''
        print '\n\n-------> minus_vect', minus_vect
        print 'mean ', mean_vect
        print 'eig ', eigen_V
        '''

        plot_minus = fig.add_subplot(1, 5, 2)
        plot_minus.set_title('Mean - 1 lambda * eigen vector')
        if split:
            #TODO: check if casting complex values to float isn't a too big loss of information ...
            im_minus = plot_minus.imshow(minus_vect.reshape(n/2, n).astype(np.float))
        else:
            im_minus = plot_minus.imshow(minus_vect.reshape(n, n).astype(np.float))
        im_minus.set_interpolation('nearest')
        im_minus.set_cmap('gray')

        plot_mean = fig.add_subplot(1, 5, 3)
        plot_mean.set_title('Mean')
        if split:
            #TODO: check if casting complex values to float isn't a too big loss of information ...
            im_mean = plot_mean.imshow(mean_vect.reshape(n/2, n).astype(np.float))
        else:
            im_mean = plot_mean.imshow(mean_vect.reshape(n, n).astype(np.float))
        im_mean.set_interpolation('nearest')
        im_mean.set_cmap('gray')

        plus_vect = mean_vect + eigen_value*eigen_V
        plot_plus = fig.add_subplot(1, 5, 4)
        plot_plus.set_title('Mean + 1 lambda * eigen vector')
        if split:
            #TODO: check if casting complex values to float isn't a too big loss of information ...
            im_plus = plot_plus.imshow(plus_vect.reshape(n/2, n).astype(np.float))
        else:
            im_plus = plot_plus.imshow(plus_vect.reshape(n, n).astype(np.float))
        im_plus.set_interpolation('nearest')
        im_plus.set_cmap('gray')

        plus_3vect = mean_vect +  3 * eigen_value *eigen_V

        plot_plus3 = fig.add_subplot(1, 5, 5)
        plot_plus3.set_title('Mean + 3 lambda * eigen vector')
        if split:
            #TODO: check if casting complex values to float isn't a too big loss of information ...
            im_plus3 = plot_plus3.imshow(plus_3vect.reshape(n/2, n).astype(np.float))
        else:
            im_plus3 = plot_plus3.imshow(plus_3vect.reshape(n, n).astype(np.float))
        im_plus3.set_interpolation('nearest')
        im_plus3.set_cmap('gray')



        plt.suptitle('Mode ' + str(mode))
        plt.show()



    # plot the projected dictionary on nb_mode modes, if target is provided then it will also add its coord in the graph
    def plot_omega(self, nb_mode=1, target_coord=None, to_highlight=None):
        if self.kept < nb_mode:
            print "Can't plot {} modes, not enough modes kept. " \
                  "Try to increase k, which is curently {}".format(nb_mode, self.k)
            exit(2)
        assert self.omega.shape == (self.kept, self.J), "The matrix is {}".format(self.omega.shape)
        for i in range(nb_mode):
            for j in range(i,nb_mode):
                # Plot the dictionary
                if j != i:
                    fig = plt.figure()

                    graph = fig.add_subplot(1,1,1)
                    graph.plot(self.omega[i, 0:self.J], self.omega[j, 0:self.J],
                             'o', markersize=7, color='blue', alpha=0.5, label='dictionary')

                    if to_highlight is not None:
                        graph.plot(self.omega[i, to_highlight[1]], self.omega[j, to_highlight[1]],
                             'o', markersize=7, color='black', alpha=0.5, label='chosen dictionary')

                    # Plot the projected image's coord
                    if target_coord is not None:
                        # target coord is a numpy array of either dimension of all the slices or just one slice
                        if len(target_coord.shape) == 2:
                            graph.plot(target_coord[i], target_coord[j],
                                     '^', markersize=7, color='red', alpha=0.5, label='target')

                        elif len(target_coord.shape) == 3:
                            for j_slice,slice_coord in enumerate(target_coord):
                                graph.plot(slice_coord[i], slice_coord[j],
                                         '^', markersize=7, color='red', alpha=0.5, label='target')

                                if to_highlight is not None and j_slice == to_highlight[0]:
                                    graph.plot(slice_coord[i], slice_coord[j],
                                             '^', markersize=7, color='black', alpha=0.5, label='this target')

                        else:
                            sct.printv('Cannot plot projected target.', 1, 'error')

                    plt.title('Atlas and target slices in the PCA space. (' + str(self.kept) + ' modes in total)')
                    plt.xlabel('Mode ' + str(i))
                    plt.ylabel('Mode ' + str(j))
        plt.show()


