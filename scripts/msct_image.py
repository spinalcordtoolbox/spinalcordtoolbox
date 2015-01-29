#!/usr/bin/env python
#########################################################################################
#
# Image class implementation
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux
# Modified: 2014-11-28
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import nibabel as nib
import sct_utils as sct
import numpy as np
import matplotlib.pyplot as plt
from sct_orientation import get_orientation


class Image:

    def __init__(self, path=None, verbose=0, np_array=None, split=False):
        if path is not None:
            sct.check_file_exist(path, verbose=verbose)
            try:
                im_file = nib.load(path)
            except nib.spatialimages.ImageFileError:
                sct.printv('Error: make sure ' + path + ' is an image.')
            self.orientation = get_orientation(path)
            self.data = im_file.get_data()
            self.hdr = im_file.get_header()
            self.path, self.file_name, self.ext = sct.extract_fname(path)
        elif np_array is not None:
            self.data = np_array
            self.path = None
            self.orientation = None
        else:
            raise TypeError(' Image constructor takes at least one argument.')
        if split:
            self.data = self.split_data()
        self.dim = self.data.shape

    def save(self):
        #hdr.set_data_dtype(img_type) # set imagetype to uint8 #TODO: maybe use int32
        self.hdr.set_data_shape(self.data.shape)
        img = nib.Nifti1Image(self.data, None, self.hdr)
        print 'saving ' + self.path + self.file_name + self.ext + '\n'
        print self.hdr.get_data_shape()
        nib.save(img, self.path + self.file_name + self.ext)

    # flatten the array in a single dimension vector, its shape will be (d, 1) compared to the flatten built in method
    # which would have returned (d,)
    def flatten(self):
#        return self.data.flatten().reshape(self.data.flatten().shape[0], 1)
        return self.data.flatten()

    # return a list of the image slices flattened
    def slices(self):
        slices = []
        for slc in self.data:
            slices.append(slc.flatten())
        return slices

    # return an empty image of the same size as the image self
    def empty_image(self):
        import copy
        im_buf = copy.copy(self)
        im_buf.data *= 0
        return im_buf

    # crop the image in order to keep only voxels in the mask, therefore the mask's slices must be squares or
    # rectangles of the same size
    # This method is called in sct_crop_over_mask script
    def crop_from_square_mask(self, mask):
        array = self.data
        data_mask = mask.data
        print 'ORIGINAL SHAPE: ', array.shape, '   ==   ', data_mask.shape
        array = np.asarray(array)
        data_mask = np.asarray(data_mask)
        new_data = []
        buffer = []
        buffer_mask = []
        s = 0
        r = 0
        ok = 0
        for slice in data_mask:
            #print 'SLICE ', s, slice
            for row in slice:
                if sum(row) > 0:
                    buffer_mask.append(row)
                    buffer.append(array[s][r])
                    #print 'OK1', ok
                    ok += 1
                r += 1
            new_slice_mask = np.asarray(buffer_mask).T
            new_slice = np.asarray(buffer).T
            r = 0
            buffer = []
            for row in new_slice_mask:
                if sum(row) != 0:
                    buffer.append(new_slice[r])
                r += 1
            #print buffer
            new_slice = np.asarray(buffer).T
            r = 0
            buffer_mask = []
            buffer = []
            new_data.append(new_slice)
            s += 1
        new_data = np.asarray(new_data)
        #print data_mask
        print 'SHAPE ', new_data.shape
        self.data = new_data

    def show(self):
        imgplot = plt.imshow(self.data)
        imgplot.set_cmap('gray')
        imgplot.set_interpolation('nearest')
        plt.show()

    def split_data(self):
        from sct_asman import split
        new_data = []
        for slice in self.data:
            left, right = split(slice)
            new_data.append(left)
            new_data.append(right)
        new_data = np.asarray(new_data)
        return new_data