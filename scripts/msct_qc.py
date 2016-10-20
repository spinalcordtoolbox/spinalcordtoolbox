#!/usr/bin/env python
#########################################################################################
#
# Qc class implementation
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Frederic Cloutier
# Modified: 2016-10-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from msct_image import Image
from scipy import ndimage
import abc


class Qc(object):
    """
    Create a .png file from a 2d image.
    """

    def __init__(self, alpha, dpi=300, interpolation='none'):
        self.interpolation = interpolation
        self.alpha = alpha
        self.dpi = dpi

    def __call__(self, f):
        def wrapped_f(slice,*args, **kargs):
            name = slice.name
            self.mkdir()
            img, mask = f(slice,*args, **kargs)
            assert isinstance(img, np.ndarray)
            assert isinstance(mask, np.ndarray)
            fig = plt.imshow(img, cmap='gray', interpolation=self.interpolation)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            self.save('{}_gray'.format(name))
            mask = np.ma.masked_where(mask < 1, mask)
            plt.imshow(img, cmap='gray', interpolation=self.interpolation)
            plt.imshow(mask, cmap=cm.hsv, interpolation=self.interpolation, alpha=self.alpha, )
            self.save(name)

            plt.close()

        return wrapped_f

    def save(self, name, format='png', bbox_inches='tight', pad_inches=0):
        plt.savefig('{}.png'.format(name), format=format, bbox_inches=bbox_inches,
                    pad_inches=pad_inches, dpi=self.dpi)

    def mkdir(self):
        # TODO : implement function
        # make a new.or update Qc directory
        return  0


class slices(object):

    def __init__(self,name):
        self.name = name

    __metaclass__ = abc.ABCMeta

    @staticmethod
    def crop(matrix, center_x, center_y, ray_x, ray_y):
        start_row = center_x - ray_x
        end_row = center_x + ray_x
        start_col = center_y - ray_y
        end_col = center_y + ray_y

        if matrix.shape[ 0 ] < end_row:
            if matrix.shape[ 0 ] < (end_row - start_row):# TODO: throw/raise an exception
                raise OverflowError
            return slices.crop(matrix, center_x - 1, center_y, ray_x, ray_y)
        if matrix.shape[ 1 ] < end_col:
            if matrix.shape[ 1 ] < (end_col - start_col):# TODO: throw/raise an exception
                raise OverflowError
            return slices.crop(matrix, center_x, center_y - 1, ray_x, ray_y)
        if start_row < 0:
            return slices.crop(matrix, center_x + 1 , center_y, ray_x , ray_y)
        if start_col < 0:
            return slices.crop(matrix, center_x, center_y + 1, ray_x, ray_y )

        return matrix[ start_row:end_row, start_col:end_col ]

    @staticmethod
    def add_slice(matrix, i, column, size, patch):
        startCol = (i % column) * size * 2
        endCol = startCol + patch.shape[ 1 ]
        startRow = int(math.ceil(i / column)) * size * 2
        endRow = startRow + patch.shape[ 0 ]
        matrix[ startRow:endRow, startCol:endCol ] = patch
        return matrix

    @staticmethod
    def nan_fill(array):
        array[ np.isnan(array) ] = np.interp(np.isnan(array).ravel().nonzero()[0]
                                             , (-np.isnan(array)).ravel().nonzero()[0]
                                             , array[ -np.isnan(array) ])
        return array

    @abc.abstractmethod
    def getSlice(self, data, i):
        """
        Abstract method to obtain a slice of a 3d matrix.
        :param data: 3d numpy.ndarray
        :param i: int
        :return: 2d numpy.ndarray
        """
        return

    @abc.abstractmethod
    def getDim(self, image):
        """
        Abstract method to obtain the depth of the 3d matrix.
        :param image: 3d numpy.ndarray
        :return: int
        """
        return

    def getImage(self, imageName, segImageName):
        image = Image(imageName)
        image_seg = Image(segImageName)
        image.change_orientation('RPI')  # reorient to RPI
        image_seg.change_orientation('RPI')  # reorient to RPI
        return image, image_seg, self.getDim(image)

    def mosaic(self, imageName, segImageName, nb_column, size):
        print segImageName
        image, image_seg, dim = self.getImage(imageName, segImageName)
        matrix0 = np.ones((size * 2 * int((dim / nb_column) + 1), size * 2 * nb_column))
        matrix1 = np.empty((size * 2 * int((dim / nb_column) + 1), size * 2 * nb_column))
        centers_x = np.zeros(dim)
        centers_y = np.zeros(dim)
        for i in range(dim):
            centers_x[i], centers_y[i] = ndimage.measurements.center_of_mass(self.getSlice(image_seg.data, i))
        try:
            slices.nan_fill(centers_x)
            slices.nan_fill(centers_y)
        except ValueError:
            print "Oops! There are no trace of that spinal cord." # TODO : raise error
            raise

        for i in range(dim):
            x = int(round(centers_x[ i ]))
            y = int(round(centers_y[ i ]))
            matrix0 = slices.add_slice(matrix0, i, nb_column, size,
                                       slices.crop(self.getSlice(image.data, i), x, y, size, size))
            matrix1 = slices.add_slice(matrix1, i, nb_column, size,
                                       slices.crop(self.getSlice(image_seg.data, i), x, y, size, size))

        return matrix0, matrix1

    def single(self, imageName, segImageName):
        image, image_seg, dim = self.getImage(imageName, segImageName)
        index = 0
        sum = 0
        for i in range(dim):
            seg_img = self.getSlice(image_seg.data, i)
            tmp = seg_img.sum()
            if tmp > sum:
                sum = tmp
                index = i
        matrix0 = self.getSlice(image.data, index)
        matrix1 = self.getSlice(image_seg.data, index)

        return matrix0, matrix1

    @Qc(1)
    def save(self, imageName, segImageName, nb_column=0, size=10):
        if nb_column > 0:
            return self.mosaic(imageName, segImageName, nb_column, size)
        else:
            return self.single(imageName, segImageName)



class axial(slices):
    def getSlice(self, data, i):
        return data[ :, :, i ]

    def getDim(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nz


class sagital(slices):
    def getSlice(self, data, i):
        return data[ i, :, : ]

    def getDim(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nx


class coronal(slices):
    def getSlice(self, data, i):
        return data[ :, i, : ]

    def getDim(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return ny
