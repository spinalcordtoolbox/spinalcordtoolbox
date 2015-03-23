#!/usr/bin/env python
########################################################################################################################
#
# Implementation of Principal Component Analysis using scatter matrix inspired by Sebastian Raschka
# http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#sc_matrix
#
#
# Step 1: Take the whole dataset consisting of J N-dimensional flattened images [NxJ], N=nb of vox
#
# Step 2: Compute the mean image: PSI (called mean_image in the code)
#
# Step 3: Compute the covariance matrix of the dataset
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


class PCA:

    def __init__(self, dataset, datashape=None, k=0.80):
        # STEP 1
        self.dataset = dataset  # This should be a J*N dimensional matrix of J N-dimensional flattened images
        self.N = dataset.shape[0]  # The number of rows is the dimension of flattened images
        self.J = dataset.shape[1]  # The number of columns is the number of images
        # STEP 2
        self.mean_image = self.mean()
        # STEP 3
        self.covariance_matrix = self.covariance_matrix()
        # STEP 4 eigpairs consist of a list of tuple (eigenvalue, eigenvector) already sorted by decreasing eigenvalues
        self.eig_pairs = self.sorted_eig()
        # STEP 5
        self.k = k
        self.W, self.kept = self.generate_W()
        # omega is a matrix of k rows and J columns, each columns correspond to a vector projection of an image from
        # the dataset
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
            covariance_matrix += float(1)/self.J*(self.dataset[:, j].reshape(self.N, 1) - self.mean_image)\
                .dot((self.dataset[:, j].reshape(self.N, 1) - self.mean_image).T)
        return covariance_matrix

    # STEP 4
    def sorted_eig(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix)
        # Create a list of (eigenvalue, eigenvector) tuple
        eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))
                     if np.abs(eigenvalues[i]) > 0.0000001]
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort()
        eig_pairs.reverse()
        return eig_pairs

    # STEP 5
    def generate_W(self):
        eigenvalues_kept = []
        s = sum([eig[0] for eig in self.eig_pairs])
        first = 1
        for eig in self.eig_pairs:
            if first:
                W = eig[1].reshape(self.N, 1)
                eigenvalues_kept.append(eig[0])
                first = 0
            else:
                if (sum(eigenvalues_kept) + eig[0])/s <= self.k:
                    eigenvalues_kept.append(eig[0])
                    W = np.hstack((W, eig[1].reshape(self.N, 1)))
                else:
                    break
        kept = len(eigenvalues_kept)
        print 'kept eigenvalues (PCA space dimension)  : ', kept
        return W, kept

    # STEP 6
    # Project the all dataset in order to get images "close" to the target & to plot the dataset
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
            coord_projected_img = self.W.T.dot(target - self.mean_image)
            return coord_projected_img
        else:
            print "target dimension is {}, must be {}.\n".format(target_as_array.shape, self.N)

    #
    # Show all the mode
    def show(self, split=0):
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
            #imgplot.set_cmap('gray')

        plt.show()



    # plot the projected dataset on nb_mode modes, if target is provided then it will also add its coord in the graph
    def plot_omega(self, nb_mode=1, target_coord=None, to_highlight=None):
        if self.kept < nb_mode:
            print "Can't plot {} modes, not enough modes kept. " \
                  "Try to increase k, which is curently {}".format(nb_mode, self.k)
            exit(2)
        assert self.omega.shape == (self.kept, self.J), "The matrix is {}".format(self.omega.shape)
        for i in range(0, nb_mode):
            # Plot the dataset
            fig = plt.figure()

            graph = fig.add_subplot(1,1,1)
            graph.plot(self.omega[i, 0:self.J], self.omega[i+1, 0:self.J],
                     'o', markersize=7, color='blue', alpha=0.5, label='dataset')

            if to_highlight is not None:
                graph.plot(self.omega[i, to_highlight[1]], self.omega[i+1, to_highlight[1]],
                     'o', markersize=7, color='black', alpha=0.5, label='chosen dataset')

            # Plot the projected image's coord
            if target_coord is not None:
                # target coord is a numpy array of either dimension of all the slices or just one slice
                if len(target_coord.shape) == 2:
                    graph.plot(target_coord[i], target_coord[i+1],
                             '^', markersize=7, color='red', alpha=0.5, label='target')

                elif len(target_coord.shape) == 3:
                    for j_slice,slice_coord in enumerate(target_coord):
                        graph.plot(slice_coord[i], slice_coord[i+1],
                                 '^', markersize=7, color='red', alpha=0.5, label='target')

                        if to_highlight is not None and j_slice == to_highlight[0]:
                            graph.plot(slice_coord[i], slice_coord[i+1],
                                     '^', markersize=7, color='black', alpha=0.5, label='this target')

                else:
                    sct.printv('Cannot plot projected target.', 1, 'error')

            plt.title('Atlas and target slices in the PCA space. (' + str(self.kept) + ' modes in total)')
            plt.xlabel('Mode ' + str(i))
            plt.ylabel('Mode ' + str(i+1))
        plt.show()


