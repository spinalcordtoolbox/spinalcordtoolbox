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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from msct_image import Image
from scipy import ndimage
import abc


class Qc(object):
    """
    Create a .png file from a 2d image.
    """

    def __init__(self, alpha, dpi=600, interpolation='none'):
        self.interpolation = interpolation
        self.alpha = alpha
        self.dpi = dpi

    def show(self,name, img, mask):
        assert isinstance(img, np.ndarray)
        assert isinstance(mask, np.ndarray)
        fig = plt.imshow(img, cmap='gray', interpolation = self.interpolation)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        self.save('{}_gray'.format(name))
        mask = np.ma.masked_where(mask < 1, mask)
        plt.imshow(img, cmap='gray', interpolation=self.interpolation)
        plt.imshow(mask, cmap=cm.hsv, interpolation= self.interpolation, alpha=self.alpha,)
        self.save(name)

        plt.close()

    def save(self, name, format='png', bbox_inches='tight', pad_inches=0):
        plt.savefig('{}.png'.format(name), format=format, bbox_inches=bbox_inches,
                    pad_inches=pad_inches, dpi=self.dpi)



class slices(object):

    __metaclass__ = abc.ABCMeta

    @staticmethod
    def add_slice(matrix, i, column, size, patch):
        startCol = (i % column) * size * 2
        endCol = startCol + patch.shape[1]
        startRow = i / column * size * 2
        endRow = startRow + patch.shape[0]
        matrix[startRow:endRow, startCol:endCol] = patch
        return matrix

    @abc.abstractmethod
    def getSlice(self,data, i):
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

    def mosaic(self, imageName, segImageName, nb_column, size):
        image = Image(imageName)
        image_seg = Image(segImageName)
        dim = self.getDim(image)
        matrix0 = np.zeros((size * 2 * int((dim / nb_column) + 1), size * 2 * nb_column))
        matrix1 = np.zeros((size * 2 * int((dim / nb_column) + 1), size * 2 * nb_column))
        for i in range(dim):
            seg_img = self.getSlice(image_seg.data,i)
            if seg_img.sum() > 0:
                img = self.getSlice(image.data,i)
                center = ndimage.measurements.center_of_mass(seg_img)
                x = int(round(center[0]))
                y = int(round(center[1]))
                matrix0 = slices.add_slice(matrix0, i, nb_column, size, crop(img, x, y, size))
                matrix1 = slices.add_slice(matrix1, i, nb_column, size, crop(seg_img, x, y, size))
        return matrix0, matrix1

    def single(self,imageName,segImageName):
        image = Image(imageName)
        imageSeg = Image(segImageName)
        dim = self.getDim(image)
        index = 0
        sum = 0

        for i in range(dim):
            seg_img = self.getSlice(imageSeg.data,i)
            tmp = seg_img.sum()
            if tmp > sum:
                sum = tmp
                index = i
        matrix0 = self.getSlice(image.data, index)
        matrix1 = self.getSlice(imageSeg.data, index)

        return  matrix0, matrix1

    def save(self, name, imageName, segImageName, nb_column=1, size=10):
        if nb_column > 1:
            img, mask = self.mosaic(imageName, segImageName, nb_column, size)
        else:
            img, mask = self.single(imageName, segImageName)

        Qc(1).show(name, img, mask)


def crop(matrix, centerX, centerY, ray):
    start_row = centerX - ray
    end_row = centerX + ray
    start_col = centerY - ray
    end_col = centerY + ray

    if matrix.shape < (end_row,end_col):
        return matrix

    return matrix[start_row:end_row,start_col:end_col]


class axial(slices):

    def getSlice(self, data, i):
        return data[:, i, :]

    def getDim(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return ny


class sagital(slices):

    def getSlice(self, data, i):
        return data[:, :, i]

    def getDim(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nz

class coronal(slices):

    def getSlice(self, data, i):
        return data[i, :, :]

    def getDim(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nx



