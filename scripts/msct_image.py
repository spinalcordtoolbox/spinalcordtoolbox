#!/usr/bin/env python
#########################################################################################
#
# Image class implementation
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Benjamin De Leener
# Modified: 2015-02-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import nibabel as nib
import sct_utils as sct
import numpy as np
import matplotlib.pyplot as plt
from sct_orientation import get_orientation


class Image(object):
    """

    """
    def __init__(self, path=None, verbose=0, np_array=None, shape=None, im_ref=None, im_ref_zero=None, split=False):
        # initialization
        self.absolutepath = ""
        self.path = ""
        self.file_name = ""
        self.ext = ""

        # load an image from file
        if path is not None:
            self.loadFromPath(path, verbose)
        # create an empty image (full of zero) of dimension [dim]. dim must be [x,y,z] or (x,y,z). No header.
        elif shape is not None:
            self.data = np.zeros(shape)
        # create a copy of im_ref
        elif im_ref is not None:
            self.data = im_ref.data
            self.hdr = im_ref.hdr
            self.orientation = im_ref.orientation
            self.absolutepath = im_ref.absolutepath
            self.path = im_ref.path
            self.file_name = im_ref.file_name
            self.ext = im_ref.ext
        # create an empty image (full of zero) with the same header than ref. Ref is an Image.
        elif im_ref_zero is not None:
            self.data = np.zeros(im_ref_zero.data.shape)
            self.hdr = im_ref_zero.hdr
            self.orientation = im_ref_zero.orientation
        # create an image from an array. No header.
        elif np_array is not None:
            self.data = np_array
            self.orientation = None
        else:
            raise TypeError(' Image constructor takes at least one argument.')
        if split:
            self.data = self.split_data()
        self.dim = self.data.shape

    def loadFromPath(self, path, verbose):
        """
        This function load an image from an absolute path using nibabel library
        :param path: path of the file from which the image will be loaded
        :return:
        """
        sct.check_file_exist(path, verbose=verbose)
        try:
            im_file = nib.load(path)
        except nib.spatialimages.ImageFileError:
            sct.printv('Error: make sure ' + path + ' is an image.')
        self.orientation = get_orientation(path)
        self.data = im_file.get_data()
        self.hdr = im_file.get_header()
        self.absolutepath = path
        self.path, self.file_name, self.ext = sct.extract_fname(path)

    def setFileName(self, filename):
        self.absolutepath = filename
        self.path, self.file_name, self.ext = sct.extract_fname(filename)

    def changeType(self, type=''):
        from numpy import uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64
        """
        Change the voxel type of the image
        :param type:    if not set, the image is saved in standard type
                        if 'minimize', image space is minimize
                        if 'minimize_int', image space is minimize and values are approximated to integers
                        (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),
                        (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),
                        (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),
                        (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),
                        (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),
                        (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),
                        (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),
                        (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),
                        (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),
                        (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),
                        (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),
                        (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),
                        (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),
                        (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),
        :return:
        """
        if type == '':
            type = self.hdr.get_data_dtype()

        if type == 'minimize' or type == 'minimize_int':
            # compute max value in the image and choose the best pixel type to represent all the pixels within smallest memory space
            # warning: does not take intensity resolution into account, neither complex voxels
            max_vox = np.nanmax(self.data)
            min_vox = np.nanmin(self.data)

            # check if voxel values are real or integer
            isInteger = True
            if type == 'minimize':
                for vox in self.data.flatten():
                    if int(vox)!=vox:
                        isInteger = False
                        break

            if isInteger:
                if min_vox >= 0: # unsigned
                    if max_vox <= np.iinfo(np.uint8).max:
                        type = 'uint8'
                    elif max_vox <= np.iinfo(np.uint16):
                        type = 'uint16'
                    elif max_vox <= np.iinfo(np.uint32).max:
                        type = 'uint32'
                    elif max_vox <= np.iinfo(np.uint64).max:
                        type = 'uint64'
                    else:
                        raise ValueError("Maximum value of the image is to big to be represented.")
                else:
                    if max_vox <= np.iinfo(np.int8).max and min_vox >= np.iinfo(np.int8).min:
                        type = 'int8'
                    elif max_vox <= np.iinfo(np.int16).max and min_vox >= np.iinfo(np.int16).min:
                        type = 'int16'
                    elif max_vox <= np.iinfo(np.int32).max and min_vox >= np.iinfo(np.int32).min:
                        type = 'int32'
                    elif max_vox <= np.iinfo(np.int64).max and min_vox >= np.iinfo(np.int64).min:
                        type = 'int64'
                    else:
                        raise ValueError("Maximum value of the image is to big to be represented.")
            else:
                #if max_vox <= np.finfo(np.float16).max and min_vox >= np.finfo(np.float16).min:
                #    type = 'np.float16' # not supported by nibabel
                if max_vox <= np.finfo(np.float32).max and min_vox >= np.finfo(np.float32).min:
                    type = 'float32'
                elif max_vox <= np.finfo(np.float64).max and min_vox >= np.finfo(np.float64).min:
                    type = 'float64'

        #print "The image has been set to "+type+" (previously "+str(self.hdr.get_data_dtype())+")"
        # change type of data in both numpy array and nifti header
        type_build = eval(type)
        self.data = type_build(self.data)
        self.hdr.set_data_dtype(type)

    def save(self, type=''):
        """
        Write an image in a nifti file
        :param type:    if not set, the image is saved in standard type
                        if 'minimize', image space is minimize
                        (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),
                        (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),
                        (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),
                        (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),
                        (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),
                        (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),
                        (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),
                        (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),
                        (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),
                        (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),
                        (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),
                        (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),
                        (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),
                        (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),
        """
        if type != '':
            self.changeType(type)

        self.hdr.set_data_shape(self.data.shape)
        img = nib.Nifti1Image(self.data, None, self.hdr)
        print 'saving ' + self.path + self.file_name + self.ext + '\n'
        nib.save(img, self.path + self.file_name + self.ext)

    # flatten the array in a single dimension vector, its shape will be (d, 1) compared to the flatten built in method
    # which would have returned (d,)
    def flatten(self):
        #return self.data.flatten().reshape(self.data.flatten().shape[0], 1)
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

    """
    def split_data(self):
        from sct_asman import split
        new_data = []
        for slice in self.data:
            left, right = split(slice)
            new_data.append(left)
            new_data.append(right)
        new_data = np.asarray(new_data)
        return new_data
    """


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    from msct_parser import Parser
    import sys
    parser = Parser(__file__)
    parser.usage.set_description('Image')
    parser.add_option("-i", "file", "file", True)
    arguments = parser.parse(sys.argv[1:])

    image = Image(arguments["-i"])
    image.changeType('minimize')