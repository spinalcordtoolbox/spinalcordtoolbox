#!/usr/bin/env python
#########################################################################################
#
# Image class implementation
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Benjamin De Leener
# Modified: 2015-02-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: update function to reflect the new get_dimension

import numpy as np
from scipy.ndimage import map_coordinates
import math


def striu2mat(striu):
    """
    Construct shear matrix from upper triangular vector
    Parameters
    ----------
    striu : array, shape (N,)
       vector giving triangle above diagonal of shear matrix.
    Returns
    -------
    SM : array, shape (N, N)
       shear matrix
    Notes
    -----
    Shear lengths are triangular numbers.
    See http://en.wikipedia.org/wiki/Triangular_number
    This function has been taken from https://github.com/matthew-brett/transforms3d/blob/39a1b01398f1d932630f722a540a5020c6c07422/transforms3d/affines.py
    """
    # Caching dictionary for common shear Ns, indices
    _shearers = {}
    for n in range(1, 11):
        x = (n ** 2 + n) / 2.0
        i = n + 1
        _shearers[x] = (i, np.triu(np.ones((i, i)), 1).astype(bool))

    n = len(striu)
    # cached case
    if n in _shearers:
        N, inds = _shearers[n]
    else: # General case
        N = ((-1+math.sqrt(8*n+1))/2.0)+1 # n+1 th root
        if N != math.floor(N):
            raise ValueError('%d is a strange number of shear elements' %
                             n)
        inds = np.triu(np.ones((N,N)), 1).astype(bool)
    M = np.eye(N)
    M[inds] = striu
    return M


def compose(T, R, Z, S=None):
    """
    Compose translations, rotations, zooms, [shears]  to affine
    Parameters
    ----------
    T : array-like shape (N,)
        Translations, where N is usually 3 (3D case)
    R : array-like shape (N,N)
        Rotation matrix where N is usually 3 (3D case)
    Z : array-like shape (N,)
        Zooms, where N is usually 3 (3D case)
    S : array-like, shape (P,), optional
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix.  P is the (N-2)th Triangular
       number, which happens to be 3 for a 4x4 affine (3D case)
    Returns
    -------
    A : array, shape (N+1, N+1)
        Affine transformation matrix where N usually == 3
        (3D case)
    This function has been taken from https://github.com/matthew-brett/transforms3d/blob/39a1b01398f1d932630f722a540a5020c6c07422/transforms3d/affines.py
    """
    n = len(T)
    R = np.asarray(R)
    if R.shape != (n,n):
        raise ValueError('Expecting shape (%d,%d) for rotations' % (n,n))
    A = np.eye(n+1)
    if S is not None:
        Smat = striu2mat(S)
        ZS = np.dot(np.diag(Z), Smat)
    else:
        ZS = np.diag(Z)
    A[:n,:n] = np.dot(R, ZS)
    A[:n,n] = T[:]
    return A


def decompose_affine_transform(A44):
    """
    Decompose 4x4 homogenous affine matrix into parts.
    The parts are translations, rotations, zooms, shears.
    This is the same as :func:`decompose` but specialized for 4x4 affines.
    Decomposes `A44` into ``T, R, Z, S``, such that::
       Smat = np.array([[1, S[0], S[1]],
                        [0,    1, S[2]],
                        [0,    0,    1]])
       RZS = np.dot(R, np.dot(np.diag(Z), Smat))
       A44 = np.eye(4)
       A44[:3,:3] = RZS
       A44[:-1,-1] = T
    The order of transformations is therefore shears, followed by
    zooms, followed by rotations, followed by translations.
    This routine only works for shape (4,4) matrices
    Parameters
    ----------
    A44 : array shape (4,4)
    Returns
    -------
    T : array, shape (3,)
       Translation vector
    R : array shape (3,3)
        rotation matrix
    Z : array, shape (3,)
       Zoom vector.  May have one negative zoom to prevent need for negative
       determinant R matrix above
    S : array, shape (3,)
       Shear vector, such that shears fill upper triangle above
       diagonal to form shear matrix (type ``striu``).
    Notes
    -----
    The implementation inspired by:
    *Decomposing a matrix into simple transformations* by Spencer
    W. Thomas, pp 320-323 in *Graphics Gems II*, James Arvo (editor),
    Academic Press, 1991, ISBN: 0120644819.
    The upper left 3x3 of the affine consists of a matrix we'll call
    RZS::
       RZS = R * Z *S
    where R is a rotation matrix, Z is a diagonal matrix of scalings::
       Z = diag([sx, sy, sz])
    and S is a shear matrix of form::
       S = [[1, sxy, sxz],
            [0,   1, syz],
            [0,   0,   1]])
    Running all this through sympy (see 'derivations' folder) gives
    ``RZS`` as ::
       [R00*sx, R01*sy + R00*sx*sxy, R02*sz + R00*sx*sxz + R01*sy*syz]
       [R10*sx, R11*sy + R10*sx*sxy, R12*sz + R10*sx*sxz + R11*sy*syz]
       [R20*sx, R21*sy + R20*sx*sxy, R22*sz + R20*sx*sxz + R21*sy*syz]
    ``R`` is defined as being a rotation matrix, so the dot products between
    the columns of ``R`` are zero, and the norm of each column is 1.  Thus
    the dot product::
       R[:,0].T * RZS[:,1]
    that results in::
       [R00*R01*sy + R10*R11*sy + R20*R21*sy + sx*sxy*R00**2 + sx*sxy*R10**2 + sx*sxy*R20**2]
    simplifies to ``sy*0 + sx*sxy*1`` == ``sx*sxy``.  Therefore::
       R[:,1] * sy = RZS[:,1] - R[:,0] * (R[:,0].T * RZS[:,1])
    allowing us to get ``sy`` with the norm, and sxy with ``R[:,0].T *
    RZS[:,1] / sx``.
    Similarly ``R[:,0].T * RZS[:,2]`` simplifies to ``sx*sxz``, and
    ``R[:,1].T * RZS[:,2]`` to ``sy*syz`` giving us the remaining
    unknowns.
    This function has been taken from https://github.com/matthew-brett/transforms3d/blob/39a1b01398f1d932630f722a540a5020c6c07422/transforms3d/affines.py
    """
    A44 = np.asarray(A44)
    T = A44[:-1,-1]
    RZS = A44[:-1,:-1]
    # compute scales and shears
    M0, M1, M2 = np.array(RZS).T
    # extract x scale and normalize
    sx = math.sqrt(np.sum(M0**2))
    M0 /= sx
    # orthogonalize M1 with respect to M0
    sx_sxy = np.dot(M0, M1)
    M1 -= sx_sxy * M0
    # extract y scale and normalize
    sy = math.sqrt(np.sum(M1**2))
    M1 /= sy
    sxy = sx_sxy / sx
    # orthogonalize M2 with respect to M0 and M1
    sx_sxz = np.dot(M0, M2)
    sy_syz = np.dot(M1, M2)
    M2 -= (sx_sxz * M0 + sy_syz * M1)
    # extract z scale and normalize
    sz = math.sqrt(np.sum(M2**2))
    M2 /= sz
    sxz = sx_sxz / sx
    syz = sy_syz / sy
    # Reconstruct rotation matrix, ensure positive determinant
    Rmat = np.array([M0, M1, M2]).T
    if np.linalg.det(Rmat) < 0:
        sx *= -1
        Rmat[:,0] *= -1
    return T, Rmat, np.array([sx, sy, sz]), np.array([sxy, sxz, syz])


class Image(object):
    """

    """
    def __init__(self, param=None, hdr=None, orientation=None, absolutepath="", dim=None, verbose=1):
        from numpy import zeros, ndarray, generic
        from sct_utils import extract_fname
        from nibabel import AnalyzeHeader

        # initialization of all parameters
        self.im_file = None
        self.data = None
        self.orientation = None
        self.absolutepath = ""
        self.path = ""
        self.file_name = ""
        self.ext = ""
        self.dim = None

        if hdr is None:
            hdr = self.hdr = AnalyzeHeader()  # an empty header
        else:
            self.hdr = hdr

        self.verbose = verbose

        # load an image from file
        if type(param) is str:
            self.loadFromPath(param, verbose)
        # copy constructor
        elif isinstance(param, type(self)):
            self.copy(param)
        # create an empty image (full of zero) of dimension [dim]. dim must be [x,y,z] or (x,y,z). No header.
        elif type(param) is list:
            self.data = zeros(param)
            self.dim = param
            self.hdr = hdr
            self.orientation = orientation
            self.absolutepath = absolutepath
            self.path, self.file_name, self.ext = extract_fname(absolutepath)
        # create a copy of im_ref
        elif isinstance(param, (ndarray, generic)):
            self.data = param
            self.dim = dim
            self.hdr = hdr
            self.orientation = orientation
            self.absolutepath = absolutepath
            self.path, self.file_name, self.ext = extract_fname(absolutepath)
        else:
            raise TypeError('Image constructor takes at least one argument.')

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return type(self)(deepcopy(self.data, memo), deepcopy(self.hdr, memo), deepcopy(self.orientation, memo), deepcopy(self.absolutepath, memo), deepcopy(self.dim, memo))

    def copy(self, image=None):
        from copy import deepcopy
        from sct_utils import extract_fname
        if image is not None:
            self.im_file = deepcopy(image.im_file)
            self.data = deepcopy(image.data)
            self.dim = deepcopy(image.dim)
            self.hdr = deepcopy(image.hdr)
            self.orientation = deepcopy(image.orientation)
            self.absolutepath = deepcopy(image.absolutepath)
            self.path, self.file_name, self.ext = extract_fname(self.absolutepath)
        else:
            return deepcopy(self)

    def loadFromPath(self, path, verbose):
        """
        This function load an image from an absolute path using nibabel library
        :param path: path of the file from which the image will be loaded
        :return:
        """
        from nibabel import load, spatialimages
        from sct_utils import check_file_exist, printv, extract_fname, run
        from sct_image import get_orientation

        # check_file_exist(path, verbose=verbose)
        try:
            self.im_file = load(path)
        except spatialimages.ImageFileError:
            printv('Error: make sure ' + path + ' is an image.', 1, 'error')
        self.data = self.im_file.get_data()
        self.hdr = self.im_file.get_header()
        self.orientation = get_orientation(self)
        self.absolutepath = path
        self.path, self.file_name, self.ext = extract_fname(path)
        self.dim = get_dimension(self.im_file)
        # nx, ny, nz, nt, px, py, pz, pt = get_dimension(path)
        # self.dim = [nx, ny, nz]


    def setFileName(self, filename):
        """
        :param filename: file name with extension
        :return:
        """
        from sct_utils import extract_fname
        self.absolutepath = filename
        self.path, self.file_name, self.ext = extract_fname(filename)

    def changeType(self, type=''):
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
        from numpy import uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64

        if type == '':
            type = self.hdr.get_data_dtype()

        if type == 'minimize' or type == 'minimize_int':
            from numpy import nanmax, nanmin
            # compute max value in the image and choose the best pixel type to represent all the pixels within smallest memory space
            # warning: does not take intensity resolution into account, neither complex voxels
            max_vox = nanmax(self.data)
            min_vox = nanmin(self.data)

            # check if voxel values are real or integer
            isInteger = True
            if type == 'minimize':
                for vox in self.data.flatten():
                    if int(vox) != vox:
                        isInteger = False
                        break

            if isInteger:
                from numpy import iinfo, uint8, uint16, uint32, uint64
                if min_vox >= 0:  # unsigned
                    if max_vox <= iinfo(uint8).max:
                        type = 'uint8'
                    elif max_vox <= iinfo(uint16):
                        type = 'uint16'
                    elif max_vox <= iinfo(uint32).max:
                        type = 'uint32'
                    elif max_vox <= iinfo(uint64).max:
                        type = 'uint64'
                    else:
                        raise ValueError("Maximum value of the image is to big to be represented.")
                else:
                    if max_vox <= iinfo(int8).max and min_vox >= iinfo(int8).min:
                        type = 'int8'
                    elif max_vox <= iinfo(int16).max and min_vox >= iinfo(int16).min:
                        type = 'int16'
                    elif max_vox <= iinfo(int32).max and min_vox >= iinfo(int32).min:
                        type = 'int32'
                    elif max_vox <= iinfo(int64).max and min_vox >= iinfo(int64).min:
                        type = 'int64'
                    else:
                        raise ValueError("Maximum value of the image is to big to be represented.")
            else:
                from numpy import finfo, float32, float64
                # if max_vox <= np.finfo(np.float16).max and min_vox >= np.finfo(np.float16).min:
                #    type = 'np.float16' # not supported by nibabel
                if max_vox <= finfo(float32).max and min_vox >= finfo(float32).min:
                    type = 'float32'
                elif max_vox <= finfo(float64).max and min_vox >= finfo(float64).min:
                    type = 'float64'

        # print "The image has been set to "+type+" (previously "+str(self.hdr.get_data_dtype())+")"
        # change type of data in both numpy array and nifti header
        type_build = eval(type)
        self.data = type_build(self.data)
        self.hdr.set_data_dtype(type)

    def save(self, type='', squeeze_data=True,  verbose=1):
        """
        Write an image in a nifti file
        :param type:    if not set, the image is saved in the same type as input data
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
        from nibabel import Nifti1Image, save
        from sct_utils import printv
        from numpy import squeeze
        from os import path, remove
        if squeeze_data:
            # remove singleton
            self.data = squeeze(self.data)
        if type != '':
            self.changeType(type)
        # update header
        if self.hdr:
            self.hdr.set_data_shape(self.data.shape)
        img = Nifti1Image(self.data, None, self.hdr)
        fname_out = self.path + self.file_name + self.ext
        if path.isfile(fname_out):
            printv('WARNING: File '+fname_out+' already exists. Deleting it.', verbose, 'warning')
            remove(fname_out)
        # save file
        save(img, fname_out)

    # flatten the array in a single dimension vector, its shape will be (d, 1) compared to the flatten built in method
    # which would have returned (d,)
    def flatten(self):
        # return self.data.flatten().reshape(self.data.flatten().shape[0], 1)
        return self.data.flatten()

    # return a list of the image slices flattened
    def slices(self):
        slices = []
        for slc in self.data:
            slices.append(slc.flatten())
        return slices

    def getNonZeroCoordinates(self, sorting=None, reverse_coord=False, coordValue=False):
        """
        This function return all the non-zero coordinates that the image contains.
        Coordinate list can also be sorted by x, y, z, or the value with the parameter sorting='x', sorting='y', sorting='z' or sorting='value'
        If reverse_coord is True, coordinate are sorted from larger to smaller.
        """
        from msct_types import Coordinate
        from sct_utils import printv
        n_dim = 1
        if self.dim[3] == 1:
            n_dim = 3
        else:
            n_dim = 4
        if self.dim[2] == 1:
            n_dim = 2

        try:
            if n_dim == 3:
                X, Y, Z = (self.data > 0).nonzero()
                list_coordinates = [Coordinate([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
            elif n_dim == 2:
                X, Y = (self.data > 0).nonzero()
                list_coordinates = [Coordinate([X[i], Y[i], self.data[X[i], Y[i]]]) for i in range(0, len(X))]
        except Exception, e:
            print 'ERROR', e
            printv('ERROR: Exception ' + str(e) + ' caught while geting non Zeros coordinates', 1, 'error')

        if coordValue:
            from msct_types import CoordinateValue
            if n_dim == 3:
                list_coordinates = [CoordinateValue([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
            else:
                list_coordinates = [CoordinateValue([X[i], Y[i], self.data[X[i], Y[i]]]) for i in range(0, len(X))]
        else:
            from msct_types import Coordinate
            if n_dim == 3:
                list_coordinates = [Coordinate([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
            else:
                list_coordinates = [Coordinate([X[i], Y[i], self.data[X[i], Y[i]]]) for i in range(0, len(X))]
        if sorting is not None:
            if reverse_coord not in [True, False]:
                raise ValueError('reverse_coord parameter must be a boolean')

            if sorting == 'x':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.x, reverse=reverse_coord)
            elif sorting == 'y':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.y, reverse=reverse_coord)
            elif sorting == 'z':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.z, reverse=reverse_coord)
            elif sorting == 'value':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.value, reverse=reverse_coord)
            else:
                raise ValueError("sorting parameter must be either 'x', 'y', 'z' or 'value'")

        return list_coordinates

    def getCoordinatesAveragedByValue(self):
        """
        This function computes the mean coordinate of group of labels in the image. This is especially useful for label's images.
        :return: list of coordinates that represent the center of mass of each group of value.
        """
        # 1. Extraction of coordinates from all non-null voxels in the image. Coordinates are sorted by value.
        coordinates = self.getNonZeroCoordinates(sorting='value')

        # 2. Separate all coordinates into groups by value
        groups = dict()
        for coord in coordinates:
            if coord.value in groups:
                groups[coord.value].append(coord)
            else:
                groups[coord.value] = [coord]

        # 3. Compute the center of mass of each group of voxels and write them into the output image
        averaged_coordinates = []
        for value, list_coord in groups.iteritems():
            averaged_coordinates.append(sum(list_coord) / float(len(list_coord)))

        averaged_coordinates = sorted(averaged_coordinates, key=lambda obj: obj.value, reverse=False)
        return averaged_coordinates

    # crop the image in order to keep only voxels in the mask, therefore the mask's slices must be squares or rectangles of the same size
    # orientation must be IRP to be able to go trough slices as first dimension
    def crop_and_stack(self, mask, suffix='_resized', save=True):
        """
        Cropping function to be used with a mask centered on the spinal cord. The crop slices are stack in the z direction.
        The result will be a kind of straighten image centered on the center of the mask (aka the center of the spinal cord)
        :param mask: mask image
        :param suffix: suffix to add to the file name (usefull only with the save option)
        :param save: save the image if True
        :return: no return, the image data is set to the new (crop) data
        """
        from numpy import asarray, zeros

        original_orientation = self.orientation
        mask_original_orientation = mask.orientation
        self.change_orientation('IRP')
        mask.change_orientation('IRP')
        data_array = self.data
        data_mask = mask.data

        # if the image to crop is smaller than the mask in total, we assume the image was centered and add a padding to fit the mask's shape
        if data_array.shape != data_mask.shape:
            old_data_array = data_array
            pad_1 = int((data_mask.shape[1] - old_data_array.shape[1])/2 + 1)
            pad_2 = int((data_mask.shape[2] - old_data_array.shape[2])/2 + 1)

            data_array = zeros(data_mask.shape)
            for n_slice, data_slice in enumerate(data_array):
                data_slice[pad_1:pad_1+old_data_array.shape[1], pad_2:pad_2+old_data_array.shape[2]] = old_data_array[n_slice]

            for n_slice, data_slice in enumerate(data_array):
                n_row_old_data_array = 0
                for row in data_slice[pad_2:-pad_2-1]:
                    row[pad_1:pad_1 + old_data_array.shape[1]] = old_data_array[n_slice, n_row_old_data_array]
                    n_row_old_data_array += 1

            self.data = data_array
            '''
            if save:
                self.file_name += suffix
                self.save()
            '''

        data_array = asarray(data_array)
        data_mask = asarray(data_mask)
        new_data = []
        buffer = []
        buffer_mask = []

        if len(data_array.shape) == 3:
            empty_slices = []
            for n_slice, mask_slice in enumerate(data_mask):
                for n_row, row in enumerate(mask_slice):
                    if sum(row) > 0:  # and n_row<=data_array.shape[1] and n_slice<=data_array.shape[0]:
                        buffer_mask.append(row)
                        buffer.append(data_array[n_slice][n_row])
                if buffer_mask == [] and buffer == []:
                    empty_slices.append(n_slice)
                    new_slice = []
                else:
                    new_slice_mask = asarray(buffer_mask).T
                    new_slice = asarray(buffer).T
                    buffer = []
                    for n_row, row in enumerate(new_slice_mask):
                        if sum(row) != 0:
                            buffer.append(new_slice[n_row])
                    new_slice = asarray(buffer).T
                    shape_mask = new_slice.shape
                    buffer_mask = []
                    buffer = []
                new_data.append(new_slice)
            if empty_slices is not []:
                for iz in empty_slices:
                    new_data[iz] = np.zeros(shape_mask)

        elif len(data_array.shape) == 2:
            for n_row, row in enumerate(data_mask):
                if sum(row) > 0:  # and n_row<=data_array.shape[1] and n_slice<=data_array.shape[0]:
                    buffer_mask.append(row)
                    buffer.append(data_array[n_row])

            new_slice_mask = asarray(buffer_mask).T
            new_slice = asarray(buffer).T
            buffer = []
            for n_row, row in enumerate(new_slice_mask):
                if sum(row) != 0:
                    buffer.append(new_slice[n_row])
            new_data = asarray(buffer).T
            buffer_mask = []
            buffer = []

        new_data = asarray(new_data)
        # print data_mask
        self.data = new_data
        #self.dim = self.data.shape

        self.change_orientation(original_orientation)
        mask.change_orientation(mask_original_orientation)
        if save:
            from sct_utils import add_suffix
            self.file_name += suffix
            add_suffix(self.absolutepath, suffix)
            self.save()

    def invert(self):
        self.data = self.data.max() - self.data
        return self

    @staticmethod
    def get_permutation_from_orientations(orientation_in, orientation_out):
        """
        This function return the permutation necessary to convert a coordinate/image from orientation_in
        to orientation_out
        :param orientation_in: string (ex: AIL)
        :param orientation_out: string (ex: RPI)
        :return: two lists: permutation list (int) and inversion list (-1 if need to inverse)
        """
        opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

        # change the orientation of the image
        perm = [0, 1, 2]
        inversion = [1, 1, 1]
        for i, character in enumerate(orientation_in):
            try:
                perm[i] = orientation_out.index(character)
            except ValueError:
                perm[i] = orientation_out.index(opposite_character[character])
                inversion[i] = -1

        return perm, inversion


    def change_orientation(self, orientation='RPI', inversion_orient=False):
        """
        This function changes the orientation of the data by swapping the image axis.
        Warning: the nifti image header is not changed!!!
        :param orientation: string of three character representing the new orientation (ex: AIL, default: RPI)
               inversion_orient: boolean. If True, the data change to match the orientation in the header, based on the orientation provided as the argument orientation.
        :return:
        """
        opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

        if self.orientation is None:
            from sct_image import get_orientation_3d
            self.orientation = get_orientation_3d(self)
        # get orientation to return at the end of function
        raw_orientation = self.orientation

        if inversion_orient:
            temp_orientation = self.orientation
            self.orientation = orientation
            orientation = temp_orientation

        perm, inversion = self.get_permutation_from_orientations(self.orientation, orientation)

        # axes inversion
        self.data = self.data[::inversion[0], ::inversion[1], ::inversion[2]]

        # axes manipulations
        from numpy import swapaxes

        if perm == [1, 0, 2]:
            self.data = swapaxes(self.data, 0, 1)
        elif perm == [2, 1, 0]:
            self.data = swapaxes(self.data, 0, 2)
        elif perm == [0, 2, 1]:
            self.data = swapaxes(self.data, 1, 2)
        elif perm == [2, 0, 1]:
            self.data = swapaxes(self.data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
            self.data = swapaxes(self.data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
        elif perm == [1, 2, 0]:
            self.data = swapaxes(self.data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
            self.data = swapaxes(self.data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
        elif perm == [0, 1, 2]:
            # do nothing
            pass
        else:
            print 'Error: wrong orientation'
        # update dim
        # http://math.stackexchange.com/questions/122916/what-is-the-inverse-cycle-of-permutation
        # TODO: change permutations
        # a = np.array([0,1,2,3,4])
        # perm = [4,1,2,0,3]
        # a[perm].tolist() --> [4, 1, 2, 0, 3]
        dim_temp = list(self.dim)
        dim_temp[0] = self.dim[[i for i, x in enumerate(perm) if x == 0][0]]  # nx
        dim_temp[1] = self.dim[[i for i, x in enumerate(perm) if x == 1][0]]  # ny
        dim_temp[2] = self.dim[[i for i, x in enumerate(perm) if x == 2][0]]  # nz
        dim_temp[4] = self.dim[[i for i, x in enumerate(perm) if x == 0][0]+4]  # px
        dim_temp[5] = self.dim[[i for i, x in enumerate(perm) if x == 1][0]+4]  # py
        dim_temp[6] = self.dim[[i for i, x in enumerate(perm) if x == 2][0]+4]  # pz
        self.dim = tuple(dim_temp)
        # update orientation
        self.orientation = orientation
        return raw_orientation

    def show(self):
        from matplotlib.pyplot import imshow, show
        imgplot = imshow(self.data)
        imgplot.set_cmap('gray')
        imgplot.set_interpolation('nearest')
        show()

    def transfo_pix2phys(self, coordi=None):
        """
        This function returns the physical coordinates of all points of 'coordi'. 'coordi' is a list of list of size
        (nb_points * 3) containing the pixel coordinate of points. The function will return a list with the physical
        coordinates of the points in the space of the image.

        Example:
        img = Image('file.nii.gz')
        coordi_pix = [[1,1,1]]   # for points: (1,1,1). N.B. Important to write [[x,y,z]] instead of [x,y,z]
        coordi_pix = [[1,1,1],[2,2,2],[4,4,4]]   # for points: (1,1,1), (2,2,2) and (4,4,4)
        coordi_phys = img.transfo_pix2phys(coordi=coordi_pix)

        :return:
        """
        from numpy import zeros, array, transpose, dot, asarray

        m_p2f = self.hdr.get_sform()
        m_p2f_transfo = m_p2f[0:3, 0:3]
        coord_origin = array([[m_p2f[0, 3]], [m_p2f[1, 3]], [m_p2f[2, 3]]])

        if not coordi is None:
            coordi_pix = transpose(asarray(coordi))
            coordi_phys = transpose(coord_origin + dot(m_p2f_transfo, coordi_pix))
            coordi_phys_list = coordi_phys.tolist()

            return coordi_phys_list

    def transfo_phys2pix(self, coordi=None):
        """
        This function returns the pixels coordinates of all points of 'coordi'
        'coordi' is a list of list of size (nb_points * 3) containing the pixel coordinate of points. The function will return a list with the physical coordinates of the points in the space of the image.


        :return:
        """
        from numpy import array, transpose, dot, asarray
        from numpy.linalg import inv

        m_p2f = self.hdr.get_sform()
        m_p2f_transfo = m_p2f[0:3,0:3]
        m_f2p_transfo = inv(m_p2f_transfo)

        coord_origin = array([[m_p2f[0, 3]],[m_p2f[1, 3]], [m_p2f[2, 3]]])

        if coordi != None:
            coordi_phys = transpose(asarray(coordi))
            coordi_pix =  transpose(dot(m_f2p_transfo, (coordi_phys-coord_origin)))
            coordi_pix_tmp = coordi_pix.tolist()
            coordi_pix_list = [[int(round(coordi_pix_tmp[j][i])) for i in range(len(coordi_pix_tmp[j]))] for j in range(len(coordi_pix_tmp))]

            return coordi_pix_list

    def transfo_phys2continuouspix(self, coordi=None, data_phys=None):
        """
        This function returns the pixels coordinates of all points of data_pix in the space of the image. The output is a matrix of size: size(data_phys) but containing a 3D vector.
        This vector is the pixel position of the point in the space of the image.
        data_phys must be an array of 3 dimensions for which each point contains a vector (physical position of the point).

        If coordi is different from none:
        coordi is a list of list of size (nb_points * 3) containing the pixel coordinate of points. The function will return a list with the physical coordinates of the points in the space of the image.


        :return:
        """
        from numpy import array, transpose, dot, asarray
        from numpy.linalg import inv
        from copy import copy

        m_p2f = self.hdr.get_sform()
        m_p2f_transfo = m_p2f[0:3, 0:3]
        m_f2p_transfo = inv(m_p2f_transfo)
        # e = dot(m_p2f_transfo, m_f2p_transfo)

        coord_origin = array([[m_p2f[0, 3]], [m_p2f[1, 3]], [m_p2f[2, 3]]])

        if coordi != None:
            coordi_phys = transpose(asarray(coordi))
            coordi_pix = transpose(dot(m_f2p_transfo, (coordi_phys - coord_origin)))
            coordi_pix_tmp = coordi_pix.tolist()
            coordi_pix_list = [[coordi_pix_tmp[j][i] for i in range(len(coordi_pix_tmp[j]))] for j in
                               range(len(coordi_pix_tmp))]

            return coordi_pix_list

    def get_values(self, coordi=None, interpolation_mode=0):
        """
        This function returns the intensity value of the image at the position coordi (can be a list of coordinates).
        :param coordi: continuouspix
        :param interpolation_mode: 0=nearest neighbor, 1= linear, 2= 2nd-order spline, 3= 2nd-order spline, 4= 2nd-order spline, 5= 5th-order spline
        :return: intensity values at continuouspix with interpolation_mode
        """
        return map_coordinates(self.data, coordi, output=np.float32, order=interpolation_mode)

    def get_transform(self, im_ref, mode='affine'):
        aff_im_self = self.im_file.affine
        aff_im_ref = im_ref.im_file.affine
        transform = np.matmul(np.linalg.inv(aff_im_self), aff_im_ref)
        if mode == 'affine':
            transform = np.matmul(np.linalg.inv(aff_im_self), aff_im_ref)
        else:
            T_self, R_self, Sc_self, Sh_self = decompose_affine_transform(aff_im_self)
            T_ref, R_ref, Sc_ref, Sh_ref = decompose_affine_transform(aff_im_ref)
            if mode == 'translation':
                T_transform = T_ref - T_self
                R_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid':
                T_transform = T_ref - T_self
                R_transform = np.matmul(np.linalg.inv(R_self), R_ref)
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid_scaling':
                T_transform = T_ref - T_self
                R_transform = np.matmul(np.linalg.inv(R_self), R_ref)
                Sc_transform = Sc_ref / Sc_self
                transform = compose(T_transform, R_transform, Sc_transform)
            else:
                transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return transform

    def get_inverse_transform(self, im_ref, mode='affine'):
        aff_im_self = self.im_file.affine
        aff_im_ref = im_ref.im_file.affine
        if mode == 'affine':
            transform = np.matmul(np.linalg.inv(aff_im_ref), aff_im_self)
        else:
            T_self, R_self, Sc_self, Sh_self = decompose_affine_transform(aff_im_self)
            T_ref, R_ref, Sc_ref, Sh_ref = decompose_affine_transform(aff_im_ref)
            if mode == 'translation':
                T_transform = T_self - T_ref
                R_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid':
                T_transform = T_self - T_ref
                R_transform = np.matmul(np.linalg.inv(R_ref), R_self)
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid_scaling':
                T_transform = T_self - T_ref
                R_transform = np.matmul(np.linalg.inv(R_ref), R_self)
                Sc_transform = Sc_self / Sc_ref
                transform = compose(T_transform, R_transform, Sc_transform)
            else:
                transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        return transform

    def interpolate_from_image(self, im_ref, fname_output, interpolation_mode=1):
        """
        This function interpolates an image by following the grid of a reference image.
        Example of use:

        from msct_image import Image
        im_input = Image(fname_input)
        im_ref = Image(fname_ref)
        im_input.interpolate_from_image(im_ref, fname_output, interpolation_mode=1)
        
        :param im_ref: reference Image that contains the grid on which interpolate.
        :return: a new image that has the same dimensions/grid of the reference image but the data of self image.
        """
        nx, ny, nz, nt, px, py, pz, pt = im_ref.dim
        x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
        indexes_ref = np.array(zip(x.ravel(), y.ravel(), z.ravel()))
        physical_coordinates_ref = im_ref.transfo_pix2phys(indexes_ref)

        # TODO: add optional transformation from reference space to image space to physical coordinates of ref grid.
        # TODO: add choice to do non-full transorm: translation, (rigid), affine
        # 1. get transformation
        # 2. apply transformation on coordinates

        coord_im = np.array(self.transfo_phys2continuouspix(physical_coordinates_ref))
        interpolated_values = self.get_values(np.array([coord_im[:, 0], coord_im[:, 1], coord_im[:, 2]]), interpolation_mode=interpolation_mode)

        im_output = Image(im_ref)
        if interpolation_mode == 0:
            im_output.changeType('int32')
        else:
            im_output.changeType('float32')
        im_output.data = np.reshape(interpolated_values, (nx, ny, nz))
        im_output.setFileName(fname_output)
        im_output.save()

    def get_slice(self, plane='sagittal', index=None, seg=None):
        """

        :param plane: 'sagittal', 'coronal' or 'axial'. default = 'sagittal'
        :param index: index of the slice to save (if none, middle slice in the given direction/plan)
        :param seg: segmentation to add in transparency to the image to save. Type Image.
        :return slice, slice_seg: ndarrays of the selected slices
        """
        copy_rpi = Image(self)
        copy_rpi.change_orientation('RPI')
        if seg is not None:
            seg.change_orientation('RPI')
        nx, ny, nz, nt, px, py, pz, pt = self.dim
        slice = None
        slice_seg = None
        if plane == 'sagittal':
            if index is None:
                slice = copy_rpi.data[int(round(nx/2)), :, :]
                if seg is not None:
                    slice_seg = seg.data[int(round(nx/2)), :, :]
            else:
                assert index < nx, 'Index larger than image dimension.'
                slice = copy_rpi.data[index, :, :]
                if seg is not None:
                    slice_seg = seg.data[index, :, :]

        elif plane == 'coronal':
            if index is None:
                slice = copy_rpi.data[:, int(round(ny/2)), :]
                if seg is not None:
                    slice_seg = seg.data[:, int(round(ny/2)), :]
            else:
                assert index < ny, 'Index larger than image dimension.'
                slice = copy_rpi.data[:, index, :]
                if seg is not None:
                    slice_seg = seg.data[:, index, :]

        elif plane == 'axial' or plane == 'transverse':
            if index is None:
                slice = copy_rpi.data[:, :, int(round(nz/2))]
                if seg is not None:
                    slice_seg = seg.data[:, :, int(round(nz/2))]
            else:
                assert index < nz, 'Index larger than image dimension.'
                slice = copy_rpi.data[:, :, index]
                if seg is not None:
                    slice_seg = seg.data[:, :, index]
        else:
            from sct_utils import printv
            printv('ERROR: wrong plan input to save slice. Please choose "sagittal", "coronal" or "axial"', self.verbose, type='error')

        return (slice, slice_seg)

    #
    def save_plane(self, plane='sagittal', index=None, format='.png', suffix='', seg=None, thr=0, cmap_col='red', path_output='./'):
        """
        Save a slice of self in the specified plan.

        :param plane: 'sagittal', 'coronal' or 'axial'. default = 'sagittal'

        :param index: index of the slice to save (if none, middle slice in the given direction/plan)

        :param format: format to be saved in. default = '.png'

        :param suffix: suffix to add to the image file name.

        :param seg: segmentation to add in transparency to the image to save. Type Image.

        :param thr: threshold to apply to the segmentation

        :param col: colormap description : 'red', 'red-yellow', or 'blue-cyan'

        :return filename_png: file name of the saved image
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from math import sqrt
        from sct_utils import slash_at_the_end
        if type(index) is not list:
            index = [index]

        slice_list = [self.get_slice(plane=plane, index=i, seg=seg) for i in index]
        path_output = slash_at_the_end(path_output, 1)
        if seg is not None:
            import matplotlib.colors as col
            color_white = col.colorConverter.to_rgba('white', alpha=0.0)
            if cmap_col == 'red-yellow':
                color_red = col.colorConverter.to_rgba('red', alpha=0.7)
                color_yellow = col.colorConverter.to_rgba('yellow', alpha=0.8)
                cmap_seg = col.LinearSegmentedColormap.from_list('cmap_seg', [color_white, color_yellow, color_red], N=256)
            elif cmap_col == 'blue-cyan':
                color_blue = col.colorConverter.to_rgba('blue', alpha=0.7)
                color_cyan = col.colorConverter.to_rgba('cyan', alpha=0.8)
                cmap_seg = col.LinearSegmentedColormap.from_list('cmap_seg', [color_white, color_blue, color_cyan], N=256)
            else:
                color_red = col.colorConverter.to_rgba('red', alpha=0.7)
                cmap_seg = col.LinearSegmentedColormap.from_list('cmap_seg', [color_white, color_red], N=256)

        n_lines = int(sqrt(len(slice_list)))
        n_col = int(len(slice_list)/n_lines)
        n_lines += 1

        try:
            fig = plt.figure(figsize=(n_lines*10, n_col*20))
            for i, slices in enumerate(slice_list):
                slice_im, slice_seg = slices
                plot = fig.add_subplot(n_lines, n_col, i+1)
                plot.imshow(slice_im, cmap=cm.gray, interpolation='nearest')
                if index[i] is None:
                    title = 'mid slice'
                else:
                    title = 'slice '+str(index[i])
                plot.set_title(title)
                if seg is not None:
                    slice_seg[slice_seg < thr] = 0
                    plot.imshow(slice_seg, cmap=cmap_seg, interpolation='nearest')
                plt.axis('off')

            # plt.imshow(slice, cmap=cm.gray, interpolation='nearest')
            # if seg is not None:
            #     plt.imshow(slice_seg, cmap=cmap_seg, interpolation='nearest')
            # plt.axis('off')
            fname_png = path_output + self.file_name + suffix + format
            plt.savefig(fname_png, bbox_inches='tight')
        except RuntimeError, e:
            from sct_utils import printv
            printv('WARNING: your device does not seem to have display feature', self.verbose, type='warning')
            printv(str(e), self.verbose, type='warning')
        return fname_png

    def save_quality_control(self, plane='sagittal', n_slices=1, seg=None, thr=0, cmap_col='red', format='.png', path_output='./', verbose=1):
        from sct_utils import printv
        nx, ny, nz, nt, px, py, pz, pt = self.dim
        if plane == 'sagittal':
            max_n_slices = nx
        elif plane == 'coronal':
            max_n_slices = ny
        elif plane == 'axial' or plane == 'transverse':
            max_n_slices = nz
        else:
            max_n_slices = None
            printv('ERROR: wrong plan input to save slice. Please choose "sagittal", "coronal" or "axial"', self.verbose, type='error')

        if n_slices > max_n_slices:
            index_list = range(max_n_slices)
        elif n_slices == 1:
            index_list = [int(round(max_n_slices/2))]
        else:
            gap = max_n_slices/n_slices
            index_list = [((i+1)*gap)-1 for i in range(n_slices)]
        index_list.sort()
        try:
            filename_image_png = self.save_plane(plane=plane, suffix='_'+plane+'_plane', index=index_list, format=format, path_output=path_output)
            info_str = 'QC output image: ' + filename_image_png
            if seg is not None:
                filename_gmseg_image_png = self.save_plane(plane=plane, suffix='_'+plane+'_plane_seg', index=index_list, seg=seg, thr=thr, cmap_col=cmap_col, format=format, path_output=path_output)
                info_str += ' & ' + filename_gmseg_image_png
            printv(info_str, verbose, 'info')
        except RuntimeError, e:
            printv('WARNING: your device does not seem to have display feature', self.verbose, type='warning')
            printv(str(e), self.verbose, type='warning')



def find_zmin_zmax(fname):
    import sct_utils as sct
    # crop image
    status, output = sct.run('sct_crop_image -i '+fname+' -dim 2 -bmax -o tmp.nii')
    # parse output
    zmin, zmax = output[output.find('Dimension 2: ')+13:].split('\n')[0].split(' ')
    return int(zmin), int(zmax)


def get_dimension(im_file, verbose=1):
    """
    Get dimension from nibabel object. Manages 2D, 3D or 4D images.
    :return: nx, ny, nz, nt, px, py, pz, pt
    """
    import nibabel.nifti1
    import sct_utils as sct
    # initialization
    nx, ny, nz, nt, px, py, pz, pt = 1, 1, 1, 1, 1, 1, 1, 1
    if type(im_file) is nibabel.nifti1.Nifti1Image:
        header = im_file.header
    elif type(im_file) is Image:
        header = im_file.hdr
    else:
        header = None
        sct.printv('WARNING: the provided image file isn\'t a nibabel.nifti1.Nifti1Image instance nor a msct_image.Image instance', verbose, 'warning')
    nb_dims = len(header.get_data_shape())
    if nb_dims == 2:
        nx, ny = header.get_data_shape()
        px, py = header.get_zooms()
    if nb_dims == 3:
        nx, ny, nz = header.get_data_shape()
        px, py, pz = header.get_zooms()
    if nb_dims == 4:
        nx, ny, nz, nt = header.get_data_shape()
        px, py, pz, pt = header.get_zooms()

    return nx, ny, nz, nt, px, py, pz, pt


def change_data_orientation(data, old_orientation='RPI', orientation="RPI"):
    """
    This function changes the orientation of a data matrix from a give orientation to another.
    This function assumes that the user already knows the orientation of the data
    :param data: data of the image
    :param old_orientation: Current orientation of the data
    :param orientation: Desired orientation for the data
    :return: Data matrix representing the
    """
    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

    # change the orientation of the image
    perm = [0, 1, 2]
    inversion = [1, 1, 1]
    for i, character in enumerate(old_orientation):
        try:
            perm[i] = orientation.index(character)
        except ValueError:
            perm[i] = orientation.index(opposite_character[character])
            inversion[i] = -1

    # axes inversion
    data = data[::inversion[0], ::inversion[1], ::inversion[2]]

    # axes manipulations
    from numpy import swapaxes

    if perm == [1, 0, 2]:
        data = swapaxes(data, 0, 1)
    elif perm == [2, 1, 0]:
        data = swapaxes(data, 0, 2)
    elif perm == [0, 2, 1]:
        data = swapaxes(data, 1, 2)
    elif perm == [2, 1, 0]:
        data = swapaxes(data, 0, 2)
    elif perm == [2, 0, 1]:
        data = swapaxes(data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
        data = swapaxes(data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
    elif perm == [1, 2, 0]:
        data = swapaxes(data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
        data = swapaxes(data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
    elif perm == [0, 1, 2]:
        # do nothing
        pass
    else:
        print 'Error: wrong orientation'

    return data

# =======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    from msct_parser import Parser
    from sct_utils import add_suffix
    import sys

    parser = Parser(__file__)
    parser.usage.set_description('Image processing functions')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image input file.",
                      mandatory=True,
                      example='im.nii.gz')
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Image output name.",
                      mandatory=False,
                      example='im_out.nii.gz')


    arguments = parser.parse(sys.argv[1:])

    image = Image(arguments["-i"])
    image.changeType('minimize')
    name_out = ''
    if "-o" in arguments:
        name_out = arguments["-o"]
