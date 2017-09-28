# -*- coding: utf-8 -*-
from __future__ import division

import abc
import logging
import math

import numpy as np
from scipy import ndimage


logger = logging.getLogger("sct.{}".format(__file__))


class Slice(object):
    """Abstract class represents the slice object that will be transformed in 2D image file.

Functions with the suffix `_slice` gets a slice cut in the desired axis at the
"i" position of the data of the 3D image. While the functions with the suffix
`_dim` gets the size of the desired dimension of the 3D image.

Attributes
----------
image : msct_image.Image
    The Image object of the original image
image_seg : msct_image.Image
    The Image object of the outputed segmneted image
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, image_name, seg_image_name):
        """
        Parameters
        ----------
        image_name : msct_image.Image
            Input 3D MRI to be separated into slices.
        seg_image_name : msct_image.Image
            Output name for the 3D MRI to be produced.
        """
        self.image = image_name
        self.image_seg = seg_image_name
        self.image.change_orientation('SAL')
        self.image_seg.change_orientation('SAL')

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
        """Crops the matrix to width and heigth from the center

        Select the size of the matrix if the calulated crop `width` or `height`
        are larger then the size of the matrix.

        TODO : Move this into the Axial class

        Parameters
        ----------
        matrix : ndarray
            Array representation of the image
        x : int
            The center of the crop area in the x axis
        y : int
            The center of the crop area in the y axis
        width : int
            The width from the center
        height : int
            The height from the center

        Returns
        -------
        ndarray
            returns the cropped matrix
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
        """Adds a slice to the Matrix containing all the slices

        TODO : Move this to the Axial class
        Parameters
        ----------
        matrix : ndarray
        i : int
        column : int
        size : int
        patch : ndarray

        Returns
        -------
        ndarray
            returns the matrix with the additional slice
        """
        start_col = (i % column) * size * 2
        end_col = start_col + patch.shape[1]

        start_row = int(i / column) * size * 2
        end_row = start_row + patch.shape[0]

        matrix[start_row:end_row, start_col:end_col] = patch
        return matrix

    @staticmethod
    def nan_fill(array):
        """TODO """
        array[np.isnan(array)] = np.interp(np.isnan(array).ravel().nonzero()[0],
                                           (-np.isnan(array)).ravel().nonzero()[0],
                                           array[-np.isnan(array)])
        return array

    @abc.abstractmethod
    def get_name(self):
        """Get the class name"""
        return

    @abc.abstractmethod
    def get_slice(self, data, i):
        """Abstract method to obtain a slice of a 3d matrix

        Parameters
        ----------
        data: numpy.ndarray
        i: int

        Returns
        -------
        numpy.ndarray
            The 2D representation of the selected slice
        """
        return

    @abc.abstractmethod
    def get_dim(self, image):
        """Abstract method to obtain the depth of the 3d matrix.

        Parameters
        ----------
        image : msct_image.Image

        Returns
        -------
        numpy.ndarray
        """
        return

    def _axial_center(self, image):
        """Gets the center of mass in the axial plan

        Parameters
        ----------
        image : msct_image.Image

        Returns
        -------
        tuple of numpy.ndarray of int
            centers of mass in the x and y axis.
        """
        axial_dim = self.axial_dim(image)
        centers_x = np.zeros(axial_dim)
        centers_y = np.zeros(axial_dim)
        for i in range(axial_dim):
            centers_x[i], centers_y[i] = ndimage.measurements.center_of_mass(self.axial_slice(image.data, i))
        try:
            Slice.nan_fill(centers_x)
            Slice.nan_fill(centers_y)
        except ValueError as err:
            logger.error("Axial center of the spinal cord is not found", err)
            raise err
        return centers_x, centers_y

    def mosaic(self, nb_column=0, size=15):
        """Obtain matrices of the mosaics

        Calculates how many squares will fit in a row based on the column and the size
        Multiply by 2 because the sides are of size*2. Central point is size +/-.

        Parameters
        ----------
        nb_column : int
        size : int

        Returns
        -------
        tuple of numpy.ndarray
            matrix of the input 3D RMI containing the mosaics of slices' "pixels"
            and matrix of the transformed 3D RMI to output containing the mosaics
            of slices' "pixels"
        """
        dim = self.get_dim(self.image)
        if nb_column == 0:
            nb_column = 600 // (size * 2)

        nb_row = math.ceil(dim // nb_column) + 1

        length, width = self.get_slice(self.image.data, 1).shape

        matrix_sz = (int(size * 2 * nb_row), int(size * 2 * nb_column))
        matrix0 = np.ones(matrix_sz)
        matrix1 = np.zeros(matrix_sz)
        centers_x, centers_y = self.get_center()

        for i in range(dim):
            x = int(centers_x[i])
            y = int(centers_y[i])

            matrix0 = self.add_slice(matrix0, i, nb_column, size,
                                     self.crop(self.get_slice(self.image.data, i), x, y, size, size))
            matrix1 = self.add_slice(matrix1, i, nb_column, size,
                                     self.crop(self.get_slice(self.image_seg.data, i), x, y, size, size))

        return matrix0, matrix1

    def single(self):
        """Obtain the matrices of the single slices

        Returns
        -------
        tuple of numpy.ndarray
            matrix of the input 3D RMI containing the slices and matrix of the
            transformed 3D RMI to output containing the slices
        """
        assert self.image.data.shape == self.image_seg.data.shape

        dim = self.get_dim(self.image)
        matrix0 = self.get_slice(self.image.data, dim / 2)
        matrix1 = self.get_slice(self.image_seg.data, dim / 2)
        index = self.get_center_spit()
        for j in range(len(index)):
            matrix0[j] = self.get_slice(self.image.data, int(round(index[j])))[j]
            matrix1[j] = self.get_slice(self.image_seg.data, int(round(index[j])))[j]

        return matrix0, matrix1

    def aspect(self):
        return self.get_aspect(self.image), self.get_aspect(self.image_seg)


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

    def get_center_spit(self):
        size = self.axial_dim(self.image_seg)
        return np.ones(size) * size / 2

    def get_center(self):
        return self._axial_center(self.image_seg)


class AxialTemplate(Axial):
    """The axial template representation of a slice"""
    def get_dim(self, image):
        return min(self.axial_dim(image), self.axial_dim(self.image_seg))

    def get_size(self, image):
        return min(image.data.shape + self.image_seg.data.shape) / 2

    def get_center(self):
        size = self.get_size(self.image)
        dim = self.get_dim(self.image)
        return np.ones(dim) * size, np.ones(dim) * size

    def mosaic(self, nb_column=10, size=15):
        return super(AxialTemplate, self).mosaic(size=self.get_size(self.image), nb_column=nb_column)

    def single(self):
        dim = self.get_dim(self.image)
        matrix0 = self.get_slice(self.image.data, dim / 2)
        matrix1 = self.get_slice(self.image_seg.data, dim / 2)

        return matrix0, matrix1


class AxialTemplate2Anat(AxialTemplate):
    """The axial template to anat representation of a slice"""
    def __init__(self, image_name, template2anat_image_name, seg_image_name):
        super(AxialTemplate2Anat, self).__init__(image_name, template2anat_image_name)
        self.image_seg2 = seg_image_name  # transformed input the one segmented
        self.image_seg2.change_orientation('SAL')  # reorient to SAL

    def get_center(self):
        return self._axial_center(self.image_seg2)


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

    def get_center_spit(self):
        x, y = self._axial_center(self.image_seg)
        return y

    def get_center(self):
        dim = self.get_dim(self.image_seg)
        size_y = self.axial_dim(self.image_seg)
        size_x = self.coronal_dim(self.image_seg)
        return np.ones(dim) * size_x / 2, np.ones(dim) * size_y / 2


class SagittalTemplate(Sagittal):
    """The sagittal template representation of a slice"""

    def get_dim(self, image):
        return min([self.sagittal_dim(image), self.sagittal_dim(self.image_seg)])

    def get_size(self, image):
        return min(image.data.shape + self.image_seg.data.shape) / 2

    def get_center(self):
        size = self.get_size(self.image)
        dim = self.get_dim(self.image)
        return np.ones(dim) * size, np.ones(dim) * size

    def mosaic(self, nb_column=10, size=15):
        return super(SagittalTemplate, self).mosaic(size=self.get_size(self.image), nb_column=nb_column)

    def single(self):
        dim = self.get_dim(self.image)
        matrix0 = self.get_slice(self.image.data, dim / 2)
        matrix1 = self.get_slice(self.image_seg.data, dim / 2)

        return matrix0, matrix1


class SagittalTemplate2Anat(Sagittal):
    """The sagittal template to Anat representation of a slice"""

    def __init__(self, image_name, template2anat_name, seg_image_name):
        super(SagittalTemplate2Anat, self).__init__(image_name, template2anat_name)
        self.image_seg2 = seg_image_name  # transformed input the one segmented
        self.image_seg2.change_orientation('SAL')  # reorient to SAL

    def get_center(self):
        return self._axial_center(self.image_seg2)


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

    def get_center_spit(self):
        x, y = self._axial_center(self.image_seg)
        return x

    def get_center(self):
        dim = self.get_dim(self.image_seg)
        size_y = self.axial_dim(self.image_seg)
        size_x = self.sagittal_dim(self.image_seg)
        return np.ones(dim) * size_x / 2, np.ones(dim) * size_y / 2
