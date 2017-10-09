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
#       sum(kept_eigenval eigenvalues)/sum(all eigenvalues) < k (kappa in asman article)
#           This gives kept_modes, a NxV matrix with N=nb_vox and V=nb of kept_eigenval eigenvectors
#
# Step 6: Transform the input image onto the new subspace, this can be done by:
#       y = kept_modes.T*(x - mean)  where:
#           x is the input N*1 flatened image
#           .T: transpose operator
#           kept_modes [NxV]
#           y is x projected in the PCA space
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Sara Dupont
# Modified: 2014-12-05
#
# About the license: see the file LICENSE.TXT
########################################################################################################################
import numpy as np
import sct_utils as sct
from math import sqrt
import os


class PCA:
    """
    Principal Component Analysis for a data set of images

    Each dimension of the original data set space represent the variation of intensity of one pixel among the data set.
    The new space (PCA space) axes are linear combinations of the original space axes.
    The axes (or modes) of the PCA are the eigenvectors of the covariance matrix of the data set
    Each eigenvector is associated with an eigenvalue,
    the eigenvalue indicates the amount of the data set variability explained by this mode

    (eigenvalue/sum(eigenvalues)) = percentage of variability explained by this axis

    The reduced PCA space is the space composed of the eigenvectors that explain in total k% of the variability
    """

    def __init__(self, slices, mean_vect=None, eig_pairs=None, k=0.80, verbose=1):
        """
        Principal Components Analysis constructor

        :param dataset: data set used to do the PCA

        :param mean_vect: flatten mean image -> if you want to load a PCA instead of compute it

        :param eig_pairs: list of tuples (eigenvalue, eigenvector) -> if you want to load a PCA instead of compute it

        :param k: percentage of variability to keep in the reduced PCA space

        :param verbose: 0: displays nothing, 1: displays text, 2: displays text and figures

        """
        self.verbose = verbose
        # STEP 1
        # The data set should be a J*N dimensional matrix of J N-dimensional flattened images
        self.slices = slices

        self.dataset = np.asarray([dic_slice.im_M_flat for dic_slice in self.slices]).T  # type: list of flatten images
        # The number of rows in self.dataset is the dimension of flattened images
        self.N = self.dataset.shape[0]  # type: int
        # The number of columns in self.datatset is the number of images
        self.J = self.dataset.shape[1]  # type: int

        # STEP 2
        if mean_vect is not None:
            self.mean_data_vect = mean_vect
        else:
            self.mean_data_vect = self.mean()

        # unflatten mean image
        n = int(sqrt(self.N))
        self.mean_image = self.mean_data_vect.reshape(n, n)

        if self.verbose == 2:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(self.mean_image.astype(np.float))
            plt.set_cmap('gray')
            plt.plot()
            plt.show()

        # compute the the eigenvalues and eigenvectors from the data
        if eig_pairs is None:
            # STEP 3
            self.covariance_matrix = self.covariance_matrix()

            # STEP 4
            # eigpairs consist of a list of tuple (eigenvalue, eigenvector) already sorted by decreasing eigenvalues
            self.eig_pairs = self.sort_eig()

        # or get the eigenvalues and eigenvectors from the arguments
        else:
            # STEP 3
            self.covariance_matrix = None
            # STEP 4
            # eigpairs consist of a list of tuple (eigenvalue, eigenvector) already sorted by decreasing eigenvalues
            self.eig_pairs = eig_pairs

        # STEP 5
        self.k = k  # type: float
        self.kept_modes, self.kept_eigenval = self.select_kept_modes(modes_to_ignore=0)

        sct.printv('\n\n-------> IN PCA : ', self.verbose, 'normal')
        sct.printv('\n-> kept_modes:' + str(self.kept_modes), self.verbose, 'normal')
        sct.printv('\n-> kept_eigenval:' + str(len(self.kept_eigenval)), self.verbose, 'normal')

        # --> The eigenvectors are the modes of the PCA
        if self.verbose == 2:
            import matplotlib.pyplot as plt
            eig_val = [pair[0] for pair in self.eig_pairs]
            eig_val = 100 * np.asarray(eig_val) / float(np.sum(eig_val))
            n = 100
            index = range(n)
            eig_val_to_plot = np.cumsum(eig_val[:n])
            width = 0.5
            plt.figure()
            plt.bar(index, eig_val_to_plot, width, color='b')
            plt.axvline(len(self.kept_eigenval), color='r')
            plt.ylabel('Eigenvalues (in %)')
            plt.axis([-1, n, 0, max(eig_val_to_plot) + 10])
            plt.plot()
            plt.show()

        # dataset_coord is a matrix of len(self.kept_eigenval) rows and J columns,
        # each columns correspond to a vector projection of an image from the dictionary
        self.dataset_coord = self.project_dataset()

    # ------------------------------------------------------------------------------------------------------------------
    # FUNCTIONS
    # ------------------------------------------------------------------------------------------------------------------

    # STEP 2
    # ------------------------------------------------------------------------------------------------------------------
    def mean(self):
        """
        Compute the mean data set image as a vector (flatten image)

        :return mean_im: mean image flatten (vector)
        """
        mean_im = []
        for row in self.dataset:
            m = sum(row) / self.J
            mean_im.append(m)
        mean_im = np.array([mean_im]).T
        return mean_im

    # STEP 3
    # ------------------------------------------------------------------------------------------------------------------
    def covariance_matrix(self):
        """
        Compute the covariance matrix of the data set

        :return covariance_matrix:
        """
        covariance_matrix = np.zeros((self.N, self.N))
        for j in range(0, self.J):
            covariance_matrix += float(1) / self.J * (self.dataset[:, j].reshape(self.N, 1) - self.mean_data_vect)\
                .dot((self.dataset[:, j].reshape(self.N, 1) - self.mean_data_vect).T)
        return covariance_matrix

    # STEP 4
    # ------------------------------------------------------------------------------------------------------------------
    def sort_eig(self):
        """
        Sort the eigenvalues and eigenvectors by decreasing eigenvalues

        :return eig_pairs: sorted list of tuples (eigenvalue, eigenvector)
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix)

        eigenvectors = eigenvectors.astype(np.float)

        # Create a list of (eigenvalue, eigenvector) tuple
        eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))
                     if np.abs(eigenvalues[i]) > 0.0000001]

        # Sort the (eigenvalue, eigenvector) tuples from high eigenvalues to low eigenvalues
        def get_key(item):
            """
            indicate the key to use to sort a tuple list 9here : first element
            :param item: tuple
            :return item[0]: first element of tuple
            """
            return item[0]  # sorting by eigenvalues
        eig_pairs = sorted(eig_pairs, key=get_key, reverse=True)
        return eig_pairs

    # STEP 5
    # ------------------------------------------------------------------------------------------------------------------
    def select_kept_modes(self, modes_to_ignore=0):
        """
        select the modes to keep according the percentage of variability to keep (self.k)

        :param modes_to_ignore: if not 0 the specified number of first modes will be ignored

        :return kept_modes, kept_eigenvalues: list of the kept modes vectors and list of the kept eigenvalues
        """
        kept_eigenvalues = []
        s = sum([eig[0] for eig in self.eig_pairs])
        sct.printv('\n ---> sum of eigenvalues : ' + str(s), self.verbose, 'normal')
        first = 1
        start = modes_to_ignore
        kept_modes = []
        for eig in self.eig_pairs[start:]:
            if first:
                kept_modes = np.asarray(eig[1]).reshape(self.N, 1)
                kept_eigenvalues.append(eig[0])
                first = 0
            else:
                if (sum(kept_eigenvalues) + eig[0]) / s <= self.k:
                    kept_eigenvalues.append(eig[0])
                    kept_modes = np.hstack((kept_modes, np.asarray(eig[1]).reshape(self.N, 1)))
                else:
                    break

        sct.printv('kept eigenvalues (PCA space dimension)  : ' + str(len(kept_eigenvalues)), self.verbose, 'normal')
        return kept_modes, kept_eigenvalues

    # STEP 6
    # ------------------------------------------------------------------------------------------------------------------
    def project_dataset(self):
        """
        project all the images of the data set into the PCA reduced space

        :return dataset_coord: coordinates of the data set as a list of vectors
        """
        dataset_coord = []
        for column in self.dataset.T:
            dataset_coord.append(self.project_array(column).reshape(len(self.kept_eigenval),))
        return np.asarray(dataset_coord).T

    # ------------------------------------------------------------------------------------------------------------------
    def project(self, image_list):
        """
        project a 3D image into the PCA reduced space

        :param image_list: image to project

        :return coord_projected_img: numpy array containing the coordinates of the image in the PCA reduced space
        """
        # check if the target is a volume or a flat image
        coord_projected_slices = []
        # loop across all the slice
        for image_slice in image_list:
            coord_projected_slices.append(self.project_array(image_slice.flatten()))
        return np.asarray(coord_projected_slices)

    # ------------------------------------------------------------------------------------------------------------------
    def project_array(self, image_as_array):
        """
        project a flatten image into the PCA reduced space

        :param image_as_array: flatten image to project

        :return coord_projected_img: coordinates of the image in the PCA reduced space
        """
        if image_as_array.shape == (self.N,):
            target = image_as_array.reshape(self.N, 1)
            coord_projected_img = self.kept_modes.T.dot(target - self.mean_data_vect)
            return coord_projected_img.reshape(coord_projected_img.shape[:-1])
        else:
            sct.printv("target dimension is {}, must be {}.\n".format(image_as_array.shape, self.N))

    # ------------------------------------------------------------------------------------------------------------------
    def save_data(self, path):
        """
        Save the PCA data (vector of the mean image, eigenvalues and eigenvectors) into a text file

        :param path: path where save the data
        """
        import pickle, gzip
        previous_path = os.getcwd()
        os.chdir(path)

        pca_data = np.asarray([self.mean_data_vect, self.eig_pairs])
        pickle.dump(pca_data, gzip.open('./pca_data.pklz', 'wb'), protocol=2)

        os.chdir(previous_path)

    # ------------------------------------------------------------------------------------------------------------------
    def show_all_modes(self):
        """
        Displays the kept PCA modes as images
        """
        from math import sqrt
        import matplotlib.pyplot as plt

        n = int(sqrt(self.N))
        fig = plt.figure()
        for i_fig in range(0, len(self.kept_eigenval)):
            eigenvect = self.kept_modes.T[i_fig, :]

            # dimensions of the subfigure
            x = int(sqrt(len(self.kept_eigenval)))
            y = int(len(self.kept_eigenval) / x)
            x += 1
            a = fig.add_subplot(x, y, i_fig + 1)
            a.set_title('Mode {}'.format(i_fig))

            imgplot = a.imshow(eigenvect.reshape(n, n).astype(np.float))
            imgplot.set_interpolation('nearest')
            imgplot.set_cmap('gray')
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    def show_mode_variation(self, mode=0):
        """
        Displays a mode variation as 5 images

        WARNING: NOT WORKING PROPERLY FOR NOW

        :param mode: mode to display
        """
        import matplotlib.pyplot as plt
        from math import sqrt
        # TODO: improve mode visualization
        n = int(sqrt(self.N))
        fig = plt.figure()
        eigen_vector = self.kept_modes.T[mode, :]
        eigen_value = self.eig_pairs[mode][0]
        mean_vect = self.mean_data_vect.reshape(len(self.mean_data_vect),)

        minus_3vect = np.asarray(mean_vect - 3 * eigen_value * eigen_vector)
        minus_3vect /= (max(minus_3vect) - min(minus_3vect))

        plot_minus3 = fig.add_subplot(1, 5, 1)
        plot_minus3.set_title('Mean - 3 lambda * eigen vector')
        im_minus3 = plot_minus3.imshow(minus_3vect.reshape(n, n).astype(np.float))
        im_minus3.set_interpolation('nearest')
        im_minus3.set_cmap('gray')

        minus_vect = mean_vect - eigen_value * eigen_vector
        minus_vect /= (max(minus_vect) - min(minus_vect))

        plot_minus = fig.add_subplot(1, 5, 2)
        plot_minus.set_title('Mean - 1 lambda * eigen vector')
        im_minus = plot_minus.imshow(minus_vect.reshape(n, n).astype(np.float))
        im_minus.set_interpolation('nearest')
        im_minus.set_cmap('gray')

        plot_mean = fig.add_subplot(1, 5, 3)
        plot_mean.set_title('Mean')
        im_mean = plot_mean.imshow(mean_vect.reshape(n, n).astype(np.float))
        im_mean.set_interpolation('nearest')
        im_mean.set_cmap('gray')

        plus_vect = mean_vect + eigen_value * eigen_vector
        plot_plus = fig.add_subplot(1, 5, 4)
        plot_plus.set_title('Mean + 1 lambda * eigen vector')
        im_plus = plot_plus.imshow(plus_vect.reshape(n, n).astype(np.float))
        im_plus.set_interpolation('nearest')
        im_plus.set_cmap('gray')

        plus_3vect = np.asarray(mean_vect + 3 * eigen_value * eigen_vector)
        plot_plus3 = fig.add_subplot(1, 5, 5)
        plot_plus3.set_title('Mean + 3 lambda * eigen vector')
        im_plus3 = plot_plus3.imshow(plus_3vect.reshape(n, n).astype(np.float))
        im_plus3.set_interpolation('nearest')
        im_plus3.set_cmap('gray')

        plt.suptitle('Mode ' + str(mode))
        plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    def plot_projected_dic(self, nb_modes=None, target_coord=None, target_levels=None, to_highlight=None):
        """
        plot the projected dictionary (data set) in a graph mode_x=f(mode_x-1)

        if a target is provided, the target slices will be plotted in the same graph

        :param nb_modes: maximum number of modes to display graph of

        :param target_coord: coordinates of a target image (ie. that wasn't in the PCA data set)

        :param to_highlight: indexes of some points to highlight as a tuple (target_slice, [list of data slices])
        """
        import matplotlib.pyplot as plt
        cmap = 'gist_ncar'
        cmin = 0
        cmax = 9
        marker_size = 7
        if target_coord is not None:
            im_dir = 'mode_images_with_target'
            if to_highlight is not None:
                im_dir += '_slice' + str(to_highlight[0])
        else:
            im_dir = 'mode_images_without_target'
        if im_dir not in os.listdir('.'):
            sct.run('mkdir ./' + im_dir)
        if nb_modes is None:
            nb_modes = int(round(len(self.kept_eigenval) / 3))

        elif len(self.kept_eigenval) < nb_modes:
            sct.printv("Can't plot {} modes, not enough modes kept. " 
                       "Try to increase k, which is curently {}".format(nb_modes, self.k))
            exit(2)
        assert self.dataset_coord.shape == (len(self.kept_eigenval), self.J), \
            "The matrix is {}".format(self.dataset_coord.shape)

        first = True
        for i in range(nb_modes):
            for j in range(i, nb_modes):

                # Plot the PCA data set slices
                if j != i:
                    fig = plt.figure()

                    graph = fig.add_subplot(1, 1, 1)
                    i_dat = self.dataset_coord[i, 0:self.J]
                    j_dat = self.dataset_coord[j, 0:self.J]
                    c_dat = np.asarray([dic_slice.level for dic_slice in self.slices])
                    # graph.plot(i_dat, j_dat, 'o', markersize=7, color='blue', alpha=0.5, label='dictionary')

                    graph.scatter(i_dat, j_dat, c=c_dat, cmap=cmap, vmin=cmin, vmax=cmax, s=marker_size**2, alpha=0.5)  # , alpha=0.5, label='dictionary')

                    if to_highlight is not None:
                        graph.plot(self.dataset_coord[i, to_highlight[1]], self.dataset_coord[j, to_highlight[1]], 'o', markersize=marker_size, color='black', alpha=0.6, label='chosen dictionary')

                    # Plot the projected image's coord
                    if target_coord is not None:
                        # target coord is a numpy array of either dimension of all the slices or just one slice
                        # target_cord.shape[0] == n_slices - target_cord.shape[1] == n PCA kept dimensions
                        if target_levels is None:
                            for j_slice, slice_coord in enumerate(target_coord):
                                graph.plot(slice_coord[i], slice_coord[j], '^', markersize=marker_size, color='black', alpha=0.5, label='target')
                        else:
                            target_levels = np.asarray(target_levels).reshape(len(target_levels), 1)
                            i_slice_dat = target_coord[0:len(target_coord), i]
                            j_slice_dat = target_coord[0:len(target_coord), j]
                            graph.scatter(i_slice_dat, j_slice_dat, marker=u'^', s=marker_size**2, c=target_levels, cmap=cmap, vmin=cmin, vmax=cmax, alpha=0.5)

                        for j_slice, slice_coord in enumerate(target_coord):
                            if to_highlight is not None and j_slice == to_highlight[0]:
                                if target_levels is None:
                                    graph.plot(slice_coord[i], slice_coord[j], '^', markersize=marker_size, color='red', alpha=0.6, label='this target')
                                else:
                                    graph.plot(slice_coord[i], slice_coord[j], '^', markersize=marker_size, color='black', alpha=0.6, label='this target')

                    title = 'Dictionary images'
                    if target_coord is not None:
                        title += ' and target slices'
                    title += ' in the PCA space.'
                    plt.title(title + ' (' + str(len(self.kept_eigenval))
                              + ' modes in total)')
                    plt.xlabel('Mode ' + str(i))
                    plt.ylabel('Mode ' + str(j))
                    plt.savefig(im_dir + '/modes_' + str(i) + '_' + str(j) + '.png')
                    if not first:
                        plt.close()
                    else:
                        first = False
        plt.show()
