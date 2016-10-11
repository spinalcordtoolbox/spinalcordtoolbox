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
import math
import matplotlib.cm as cm
from msct_image import Image
from scipy import ndimage
import abc

class Qc(object):
    def __init__(self,name,interpolation='none', alpha = 0.8 ):
        self.name = name
        self.interpolation = interpolation
        self.alpha = alpha

    def __call__(self, f):
        def wrapped_f(*args):
            img, mask= f(*args)
            fig = plt.imshow(img, cmap='gray', interpolation = self.interpolation)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig('{}_gray.png'.format(self.name), format='png', bbox_inches='tight', pad_inches=0)
            my_cmap = cm.hsv
            my_cmap.set_under('k', alpha=0)  # how the color map deals with clipping values
            #  you can change your threshold in clim values
            plt.imshow(mask, cmap=my_cmap, interpolation= self.interpolation, clim=[0.9, 1], alpha= self.alpha)
            plt.savefig('{}.png'.format(self.name), format='png', bbox_inches='tight', pad_inches=0)

            plt.close()

        return wrapped_f


class slices(object):

    __metaclass__ = abc.ABCMeta

    @staticmethod
    def crop(matrix, centerX, centerY, ray):
        startRow = centerX - ray
        endRow = centerX + ray
        startCol = centerY - ray
        endCol = centerY + ray
        return matrix[startRow:endRow,startCol:endCol]

    @staticmethod
    def addSlice(matrix, i, colum, size, patch):
        startCol = (i % colum) * size * 2
        endCol = startCol + patch.shape[1]
        startRow = i / colum * size * 2
        endRow = startRow + patch.shape[0]
        matrix[startRow:endRow, startCol:endCol] = patch
        return matrix

    @abc.abstractproperty
    def name(self):
        return "default"

    @abc.abstractmethod
    def getSlice(self,data, i):
        """
        Make a slice from a 3d image data array.
        ----------
        data : A 3d matrix.
        i : Index.
        Returns
        -------
        2d image data array.
        """
        return

    @abc.abstractmethod
    def getDim(self, image):
        """method"""
        return

    @abc.abstractmethod
    def getCenter(self, image):
        """method"""
        return

    def matrix(self,imageName,segImageName,nbcolum, size):
        image = Image(imageName)
        imageSeg = Image(segImageName)
        dim = self.getDim(image)
        matrix0 = np.zeros((size * 2 * int((dim / nbcolum)+1), size * 2 * nbcolum))
        matrix1 = np.empty((size * 2 * int((dim / nbcolum)+1), size * 2 * nbcolum))
        for i in range(dim):
            img = self.getSlice(image.data,i)
            segImg = self.getSlice(imageSeg.data,i)
            center = self.getCenter(segImg)
            x = int(round(center[0]))
            y = int(round(center[1]))
            matrix0 = slices.addSlice(matrix0, i, nbcolum, size,slices.crop(img, x, y, size))
            matrix1 = slices.addSlice(matrix1, i, nbcolum, size,slices.crop(segImg, x, y, size))

        return  matrix0, matrix1


class axial(slices):

    _name = "axial"
    @property
    def name(self):
        return self.name
    @name.setter
    def name(self,newName):
        return

    def getSlice(self,data, i):
        return data[:, i, :]

    def getDim(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return ny

    def getCenter(self, image):
        if(image.sum() != 0):
            return ndimage.measurements.center_of_mass(image)
        else:
            return image.shape[0]/2, image.shape[1]/2


    @Qc(_name)
    def save(self, nbcolum, size, imageName, segImageName):
        return self.matrix(imageName,segImageName, nbcolum, size)

class sagital(slices):

    _name = "sagital"
    @property
    def name(self):
        return self.name
    @name.setter
    def name(self,newName):
        return

    def getSlice(self,data, i):
        return data[:, :, i]

    def getDim(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nz

    def getCenter(self, image):
        return image.shape[0]/2, image.shape[1]/2


    @Qc(_name)
    def save(self, nbcolum, size, imageName, segImageName):
        return self.matrix(imageName, segImageName, nbcolum, size)





