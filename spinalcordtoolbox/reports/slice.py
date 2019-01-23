# -*- coding: utf-8 -*-

# TODO: Replace slice by spinalcordtoolbox.image.Slicer


from __future__ import print_function, absolute_import, division

import abc
import logging
import math

import numpy as np
from scipy import ndimage

from .. import image as msct_image

logger = logging.getLogger("sct.{}".format(__file__))


class Slice(object):
    """Abstract class representing slicing applied to >=1 volumes for the purpose
    of generating ROI slices.

    The many volumes that are worked on are usually an original MRI volume, then
    other ones which can be processed or segmentations of the first volumes; the ROIs
    are computed on the last volume by default.

    For convenience, the volumes are all brought in the SAL reference frame.

    Functions with the suffix `_slice` gets a slice cut in the desired axis at the
    "i" position of the data of the 3D image. While the functions with the suffix
    `_dim` gets the size of the desired dimension of the 3D image.

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, images):
        """
        :param images: list of 3D volumes to be separated into slices.
        """
        self._images = list()
        for image in images:
            img = msct_image.change_orientation(image, "SAL")
            self._images.append(img)

    @staticmethod
    def axial_slice(data, i):
        return data[i, :, :]

    @staticmethod
    def axial_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nx

    @staticmethod
    def axial_aspect(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return py / pz

    @staticmethod
    def sagittal_slice(data, i):
        return data[:, :, int(i)]

    @staticmethod
    def sagittal_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nz

    @staticmethod
    def sagittal_aspect(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return px / py

    @staticmethod
    def coronal_slice(data, i):
        return data[:, i, :]

    @staticmethod
    def coronal_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return ny

    @staticmethod
    def coronal_aspect(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return px / pz

    @abc.abstractmethod
    def get_aspect(self, image):
        return

    @staticmethod
    def crop(matrix, x, y, width, height):
        """Crops the matrix to width and height from the center

        Select the size of the matrix if the calculated crop `width` or `height`
        are larger than the size of the matrix.

        TODO : Move this into the Axial class

        :param matrix: Array representation of the image
        :param x: The center of the crop area in the x axis
        :param y: The center of the crop area in the y axis
        :param width: The width from the center
        :param height: The height from the center
        :returns: cropped matrix
        """
        if width * 2 > matrix.shape[0]:
            width = matrix.shape[0] // 2
        if height * 2 > matrix.shape[1]:
            height = matrix.shape[1] // 2

        if x < width:
            x = width
        if y < height:
            y = height

        start_row = x - width
        end_row = start_row + width * 2

        start_col = y - height
        end_col = start_col + height * 2

        return matrix[start_row:end_row, start_col:end_col]

    @staticmethod
    def add_slice(matrix, i, column, size, patch):
        """Adds a slice to the canvas containing all the slices

        TODO : Move this to the Axial class

        :param matrix: input/output "big canvas"
        :param i: slice position
        :param column: number of columns in mosaic
        :param size:
        :param patch: patch to insert
        :return: matrix
        """
        start_col = (i % column) * size * 2
        end_col = start_col + patch.shape[1]

        start_row = int(i / column) * size * 2
        end_row = start_row + patch.shape[0]

        matrix[start_row:end_row, start_col:end_col] = patch
        return matrix

    @staticmethod
    def nan_fill(A):
        """Interpolate NaN values with neighboring values in array (in-place)
        If only NaNs, return an array of zeros.
        """
        nans = np.isnan(A)
        if ~np.any(nans):
            return A
        elif np.all(nans):
            A[:] = np.zeros_like(A)
            return A
        xp = (~nans).ravel().nonzero()[0]
        fp = A[~nans]
        x  = nans.ravel().nonzero()[0]
        A[nans] = np.interp(x, xp, fp)
        return A

    @abc.abstractmethod
    def get_name(self):
        """Get the class name"""
        return

    @abc.abstractmethod
    def get_slice(self, data, i):
        """Abstract method to obtain a slice of a 3d matrix

        :param data: volume
        :param i: position to slice
        :return: 2D slice
        """
        return

    @abc.abstractmethod
    def get_dim(self, image):
        """Abstract method to obtain the depth of the 3d matrix.

        :param image: input msct_image.Image
        :returns: numpy.ndarray
        """
        return

    def _axial_center(self, image):
        """Gets the center of mass in the axial plan

        :param image : input msct_image.Image
        :returns: centers of mass in the x and y axis (tuple of numpy.ndarray of int)
            .
        """
        axial_dim = self.axial_dim(image)
        centers_x = np.zeros(axial_dim)
        centers_y = np.zeros(axial_dim)
        for i in range(axial_dim):
            aslice = self.axial_slice(image.data, i)
            centers_x[i], centers_y[i] = ndimage.measurements.center_of_mass(aslice)
        try:
            Slice.nan_fill(centers_x)
            Slice.nan_fill(centers_y)
        except ValueError as err:
            logger.error("Axial center of the spinal cord is not found: %s", err)
            raise
        return centers_x, centers_y

    def mosaic(self, nb_column=0, size=15):
        """Obtain matrices of the mosaics

        Calculates how many squares will fit in a row based on the column and the size
        Multiply by 2 because the sides are of size*2. Central point is size +/-.

        :param nb_column: number of mosaic columns
        :param size: each column size
        :return: tuple of numpy.ndarray containing the mosaics of each slice pixels
        """
        image = self._images[0]

        # Calculate number of columns to display on the report
        dim = self.get_dim(image)  # dim represents the 3rd dimension of the 3D matrix
        if nb_column == 0:
            nb_column = 600 // (size * 2)

        nb_row = math.ceil(dim // nb_column) + 1

        # Compute the matrix size of the final mosaic image
        matrix_sz = (int(size * 2 * nb_row), int(size * 2 * nb_column))

        # Get center of mass for each slice of the image. If the input is the cord segmentation, these coordinates are
        # used to center the image on each panel of the mosaid.
        centers_x, centers_y = self.get_center()

        matrices = list()
        for image in self._images:
            matrix = np.zeros(matrix_sz)
            for i in range(dim):
                x = int(centers_x[i])
                y = int(centers_y[i])
                # crop slice around center of mass and add slice to the matrix layout
                self.add_slice(matrix, i, nb_column, size, self.crop(self.get_slice(image.data, i), x, y, size, size))

            matrices.append(matrix)

        return matrices

    def single(self):
        """Obtain the matrices of the single slices

        :returns: tuple of numpy.ndarray, matrix of the input 3D MRI
                  containing the slices and matrix of the transformed 3D RMI
                  to output containing the slices
        """
        assert len(set([x.data.shape for x in self._images])) == 1, "Volumes don't have the same size"

        image = self._images[0]
        dim = self.get_dim(image)

        matrices = list()
        for image in self._images:
            matrix = self.get_slice(image.data, dim / 2)
            index = self.get_center_spit()
            for j in range(len(index)):
                matrix[j] = self.get_slice(image.data, int(np.round(index[j])))[j]
            matrices.append(matrix)

        return matrices

    def aspect(self):
        return [ self.get_aspect(x) for x in self._images ]


class Axial(Slice):
    """The axial representation of a slice"""
    def get_name(self):
        return Axial.__name__

    def get_aspect(self, image):
        return Slice.axial_aspect(image)

    def get_slice(self, data, i):
        return self.axial_slice(data, i)

    def get_dim(self, image):
        return self.axial_dim(image)

    def get_center_spit(self, img_idx=-1):
        image = self._images[img_idx]
        size = self.axial_dim(image)
        return np.ones(size) * size / 2

    def get_center(self, img_idx=-1):
        """Get the center of mass of each slice. By default, it assumes that self._images is a list of images, and the
        last item is the segmentation from which the center of mass is computed."""
        image = self._images[img_idx]
        return self._axial_center(image)


class Sagittal(Slice):
    """The sagittal representation of a slice"""

    def get_name(self):
        return Sagittal.__name__

    def get_aspect(self, image):
        return Slice.sagittal_aspect(image)

    def get_slice(self, data, i):
        return self.sagittal_slice(data, i)

    def get_dim(self, image):
        return self.sagittal_dim(image)

    def get_center_spit(self, img_idx=-1):
        image = self._images[img_idx]
        x, y = self._axial_center(image)
        return y

    def get_center(self, img_idx=-1):
        image = self._images[img_idx]
        dim = self.get_dim(image)
        size_y = self.axial_dim(image)
        size_x = self.coronal_dim(image)
        return np.ones(dim) * size_x / 2, np.ones(dim) * size_y / 2


class Coronal(Slice):
    """The coronal representation of a slice"""
    def get_name(self):
        return Coronal.__name__

    def get_aspect(self, image):
        return Slice.coronal_aspect(image)

    def get_slice(self, data, i):
        return self.coronal_slice(data, i)

    def get_dim(self, image):
        return self.coronal_dim(image)

    def get_center_spit(self, img_idx=-1):
        image = self._images[img_idx]
        x, y = self._axial_center(image)
        return x

    def get_center(self, img_idx=-1):
        image = self._images[img_idx]
        dim = self.get_dim(image)
        size_y = self.axial_dim(image)
        size_x = self.sagittal_dim(image)
        return np.ones(dim) * size_x / 2, np.ones(dim) * size_y / 2
