#!/usr/bin/env python
#############################################################################
#
# Image class implementation
#
#
# ---------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Benjamin De Leener
# Modified: 2015-02-20
#
# About the license: see the file LICENSE.TXT
#############################################################################

import sys, io, os, math, itertools, warnings

import nibabel
import nibabel.orientations

import numpy as np
from scipy.ndimage import map_coordinates

from msct_types import Coordinate
import sct_utils as sct


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
    else:  # General case
        N = ((-1 + math.sqrt(8 * n + 1)) / 2.0) + 1  # n+1 th root
        if N != math.floor(N):
            raise ValueError('%d is a strange number of shear elements' %
                             n)
        inds = np.triu(np.ones((N, N)), 1).astype(bool)
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
    if R.shape != (n, n):
        raise ValueError('Expecting shape (%d,%d) for rotations' % (n, n))
    A = np.eye(n + 1)
    if S is not None:
        Smat = striu2mat(S)
        ZS = np.dot(np.diag(Z), Smat)
    else:
        ZS = np.diag(Z)
    A[:n, :n] = np.dot(R, ZS)
    A[:n, n] = T[:]
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
    T = A44[:-1, -1]
    RZS = A44[:-1, :-1]
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
        Rmat[:, 0] *= -1
    return T, Rmat, np.array([sx, sy, sz]), np.array([sxy, sxz, syz])


class Slicer(object):
    """
    Image(s) slicer utility class.

    Can help getting ranges and slice indices.
    Can provide slices (being an *iterator*).
    """

    def __new__(cls, arg, axis="IS"):
        if isinstance(arg, (list, tuple)):
            return SlicerMany(arg, axis=axis)
        elif isinstance(arg, Image):
            return SlicerSingle(arg, axis=axis)
        else:
            raise ValueError()


class SlicerSingle(object):
    """
    Image slicer utility class.

    Can help getting ranges and slice indices.
    Can provide slices (being an *iterator*).
    """

    def __init__(self, im, axis="IS"):
        opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
        axis_labels = "LRPAIS"
        if len(axis) != 2:
            raise ValueError()
        if axis[0] not in axis_labels:
            raise ValueError()
        if axis[1] not in axis_labels:
            raise ValueError()
        if axis[0] != opposite_character[axis[1]]:
            raise ValueError()

        for idx_axis in range(2):
            dim_nr = im.orientation.find(axis[idx_axis])
            if dim_nr != -1:
                break
        if dim_nr == -1:
            raise ValueError()

        # SCT convention
        from_dir = im.orientation[dim_nr]
        self.direction = +1 if axis[0] == from_dir else -1
        self.nb_slices = im.dim[dim_nr]
        self.im = im
        self.axis = axis
        self._slice = lambda idx: tuple([(idx if x in axis else slice(None)) for x in im.orientation])

    def slice(self, idx):
        """
        :return: a multi-dimensional slice (what goes in numpy array __getitem__)
                 at the specified slice index of the slicer.
        :param idx: slice index
        """
        return self._slice(idx)

    def range(self, axis):
        """
        :return: a range providing indices for all the slices along the desired axis

        Example: Assuming image is in RPI with 3 z-slices,
        constructing a Slicer on "IS" (or "SI"),
        getting a range on "IS" would get [0,1,2],
        getting a range on "SI" would get [2,1,0].

        Notes:

        - To be used with direct image indexing, not as indexes for Slice[]
          which are "logical" according to the slicing direction..

        """
        if axis == self.axis:
            return range(0, self.nb_slices)
        if axis == self.axis[::-1]:
            return range(self.nb_slices-1, -1, -1)
        raise ValueError()

    def __len__(self):
       return self.nb_slices

    def __getitem__(self, idx):
       """
       :return: an image slice, at slicing index idx
       :param idx: slicing index (according to the slicing direction)
       """
       if isinstance(idx, slice):
           raise NotImplementedError()

       if idx >= self.nb_slices:
           raise IndexError("I just have {} slices!".format(self.nb_slices))

       if self.direction == -1:
           idx = self.nb_slices - 1 - idx

       return self.im.data[self.slice(idx)]

    def __call__(self):
        """
        Slice generator

        Example: [slice for slice in Slicer(im)()]
        """
        for idx_slice in self.range(self.axis):
            yield self.im[self.slice(idx_slice)]


class SlicerMany(object):
    """
    Image*s* slicer utility class.

    Can help getting ranges and slice indices.
    Can provide slices (being an *iterator*).

    Use with great care for now, that it's not very documented.
    """
    def __init__(self, images, axis="IS"):
        if len(images) == 0:
            raise ValueError("Don't expect me to work on 0 images!")

        self.slicers = [ Slicer(im, axis=axis) for im in images ]


        nb_slices = [ x.nb_slices for x in self.slicers ]
        if len(set(nb_slices)) != 1:
            raise ValueError("All images must have the same number of slices along the slicing axis!")
        self.nb_slices = nb_slices[0]

    def __len__(self):
        return self.nb_slices

    def __getitem__(self, idx):
        return [ x[idx] for x in self.slicers ]

    def range(self, axis):
        return self.slicer[0].range(axis)

    def slice(self, idx):
        return self.slicer[0].slice(idx)


class Image(object):
    """

    """

    def __init__(self, param=None, hdr=None, orientation=None, absolutepath=None, dim=None, verbose=1):
        from nibabel import Nifti1Header

        # initialization of all parameters
        self.im_file = None
        self.data = None
        self._path = None
        self.ext = ""

        if hdr is None:
            hdr = self.hdr = Nifti1Header()  # an empty header
        else:
            self.hdr = hdr

        if absolutepath is not None:
            self.absolutepath = absolutepath

        self.verbose = verbose

        # load an image from file
        if isinstance(param, str) or (sys.hexversion < 0x03000000 and isinstance(param, unicode)):
            self.loadFromPath(param, verbose)
            self.compute_transform_matrix()
        # copy constructor
        elif isinstance(param, type(self)):
            self.copy(param)
        # create an empty image (full of zero) of dimension [dim]. dim must be [x,y,z] or (x,y,z). No header.
        elif isinstance(param, list):
            self.data = np.zeros(param)
            self.hdr = hdr
            self.absolutepath = absolutepath
        # create a copy of im_ref
        elif isinstance(param, (np.ndarray, np.generic)):
            self.data = param
            self.hdr = hdr
            self.absolutepath = absolutepath
        else:
            raise TypeError('Image constructor takes at least one argument.')


    @property
    def dim(self):
        return get_dimension(self)

    @property
    def orientation(self):
        return get_orientation(self)

    @property
    def absolutepath(self):
        """
        Storage path (either actual or potential)

        Notes:

        - As several tools perform chdir() it's very important to have absolute paths
        - When set, if relative:

          - If it already existed, it becomes a new basename in the old dirname
          - Else, it becomes absolute (shortcut)

        Usually not directly touched (use `Image.save`), but in some cases it's
        the best way to set it.
        """
        return self._path

    @absolutepath.setter
    def absolutepath(self, value):
        if value is None:
            self._path = None
            return
        elif not os.path.isabs(value) and self._path is not None:
            value = os.path.join(os.path.dirname(self._path), value)
        elif not os.path.isabs(value):
            value = os.path.abspath(value)
        self._path = value

    @property
    def header(self):
        return self.hdr

    @header.setter
    def header(self, value):
        self.hdr = value

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return type(self)(deepcopy(self.data, memo), deepcopy(self.hdr, memo), deepcopy(self.orientation, memo), deepcopy(self.absolutepath, memo), deepcopy(self.dim, memo))

    def copy(self, image=None):
        from copy import deepcopy
        if image is not None:
            self.im_file = deepcopy(image.im_file)
            self.data = deepcopy(image.data)
            self.hdr = deepcopy(image.hdr)
            self.absolutepath = deepcopy(image.absolutepath)
        else:
            return deepcopy(self)

    def loadFromPath(self, path, verbose):
        """
        This function load an image from an absolute path using nibabel library
        :param path: path of the file from which the image will be loaded
        :return:
        """

        try:
            self.im_file = nibabel.load(path)
        except nibabel.spatialimages.ImageFileError:
            sct.printv('Error: make sure ' + path + ' is an image.', 1, 'error')
        self.data = self.im_file.get_data()
        self.hdr = self.im_file.get_header()
        self.absolutepath = path


    def change_shape(self, shape, generate_path=False):
        """
        """
        if shape is not None:
            change_shape(self, shape, self)

        if generate_path and self._path is not None:
            self._path = sct.add_suffix(self._path, "_shape-{}".format("-".join([str(x) for x in shape])))
        else:
            # safe option: remove path to avoid overwrites
            self._path = None
        return self

    def change_orientation(self, orientation, inverse=False, generate_path=False):
        """
        Change orientation on image.

        Note: the image path is voided.
        """
        if orientation is not None:
            change_orientation(self, orientation, self, inverse=inverse)
        if generate_path and self._path is not None:
            self._path = sct.add_suffix(self._path, "_{}".format(orientation.lower()))
        else:
            # safe option: remove path to avoid overwrites
            self._path = None
        return self

    def change_type(self, dtype, generate_path=False):
        """
        Change data type on image.

        Note: the image path is voided.
        """
        if dtype is not None:
            change_type(self, dtype, self)
        if generate_path and self._path is not None:
            self._path = sct.add_suffix(self._path, "_{}".format(dtype.name))
        else:
            # safe option: remove path to avoid overwrites
            self._path = None
        return self

    def save(self, path=None, dtype=None, verbose=1, mutable=False):
        """
        Write an image in a nifti file

        :param type:    if not set, the image is saved in the same type as input data
                        if 'minimize', image storage space is minimized
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

        if path is None and self.absolutepath is None:
            raise RuntimeError("Don't know where to save the image (no absolutepath or path parameter)")
        elif path is not None and os.path.isdir(path) and self.absolutepath is not None:
            # Save to destination directory with original basename
            path = os.path.join(path, os.path.basename(self.absolutepath))

        path = path or self.absolutepath

        from nibabel import Nifti1Image

        if dtype is not None:
            dst = self.copy()
            dst.change_type(dtype)
            data = dst.data
        else:
            data = self.data

        # update header
        hdr = self.hdr.copy() if self.hdr else None
        if hdr:
            hdr.set_data_shape(data.shape)

        # nb. that copy() is important because if it were a memory map, save()
        # would corrupt it
        img = Nifti1Image(data.copy(), None, hdr)
        if os.path.isfile(path):
            sct.printv('WARNING: File ' + path + ' already exists. Will overwrite it.', verbose, 'warning')

        # save file
        nibabel.save(img, path)

        if mutable:
            self.absolutepath = path

        if not os.path.isfile(path):
            raise RuntimeError("Couldn't save {}".format(path))

        return self

    def getNonZeroCoordinates(self, sorting=None, reverse_coord=False, coordValue=False):
        """
        This function return all the non-zero coordinates that the image contains.
        Coordinate list can also be sorted by x, y, z, or the value with the parameter sorting='x', sorting='y', sorting='z' or sorting='value'
        If reverse_coord is True, coordinate are sorted from larger to smaller.
        """
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
                try:
                    X, Y = (self.data > 0).nonzero()
                    list_coordinates = [Coordinate([X[i], Y[i], 0, self.data[X[i], Y[i]]]) for i in range(0, len(X))]
                except ValueError:
                    X, Y, Z = (self.data > 0).nonzero()
                    list_coordinates = [Coordinate([X[i], Y[i], 0, self.data[X[i], Y[i], 0]]) for i in range(0, len(X))]
        except Exception as e:
            sct.printv('ERROR: Exception ' + str(e) + ' caught while geting non Zeros coordinates', 1, 'error')

        if coordValue:
            from msct_types import CoordinateValue
            if n_dim == 3:
                list_coordinates = [CoordinateValue([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
            else:
                list_coordinates = [CoordinateValue([X[i], Y[i], 0, self.data[X[i], Y[i]]]) for i in range(0, len(X))]
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
        for value, list_coord in groups.items():
            averaged_coordinates.append(sum(list_coord) / float(len(list_coord)))

        averaged_coordinates = sorted(averaged_coordinates, key=lambda obj: obj.value, reverse=False)
        return averaged_coordinates


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

    def compute_transform_matrix(self):
        m_p2f = self.hdr.get_sform()
        self.m_p2f_transfo = m_p2f[0:3, 0:3]
        self.coord_origin = np.array([[m_p2f[0, 3]], [m_p2f[1, 3]], [m_p2f[2, 3]]])

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

        m_p2f = self.hdr.get_sform()
        m_p2f_transfo = m_p2f[0:3, 0:3]
        coord_origin = np.array([[m_p2f[0, 3]], [m_p2f[1, 3]], [m_p2f[2, 3]]])

        if coordi is not None:
            coordi = np.asarray(coordi)
            number_of_coordinates = coordi.shape[0]
            num_c = 100000
            result_temp = np.empty(shape=(0, 3))

            for i in range(int(number_of_coordinates / num_c)):
                coordi_temp = coordi[num_c * i:(i + 1) * num_c, :]
                coordi_pix = np.transpose(coordi_temp)
                dot_result = np.dot(m_p2f_transfo, coordi_pix)
                coordi_phys = np.transpose(coord_origin + dot_result)
                result_temp = np.concatenate((result_temp, coordi_phys), axis=0)

            if int(number_of_coordinates / num_c) == 0:
                coordi_temp = coordi
            else:
                coordi_temp = coordi[int(number_of_coordinates / num_c) * num_c:, :]
            coordi_pix = np.transpose(coordi_temp)
            coordi_phys = np.transpose(coord_origin + np.dot(m_p2f_transfo, coordi_pix))

            coordi_phys = np.concatenate((result_temp, coordi_phys), axis=0)
            coordi_phys_list = coordi_phys.tolist()
            # sct.printv(coordi_phys.shape)

            return coordi_phys_list
        """
        if coordi != None:
            coordi_phys = transpose(self.coord_origin + dot(self.m_p2f_transfo, transpose(asarray(coordi))))
            return coordi_phys.tolist()
        else:
            return None
        """
        return np.transpose(self.coord_origin + np.dot(self.m_p2f_transfo, np.transpose(np.asarray(coordi))))

    def transfo_phys2pix(self, coordi):
        """
        This function returns the pixels coordinates of all points of 'coordi'
        'coordi' is a list of list of size (nb_points * 3) containing the pixel coordinate of points. The function will return a list with the physical coordinates of the points in the space of the image.


        :return:
        """

        m_p2f = self.hdr.get_sform()
        m_p2f_transfo = m_p2f[0:3, 0:3]
        m_f2p_transfo = np.linalg.inv(m_p2f_transfo)

        coord_origin = np.array([[m_p2f[0, 3]], [m_p2f[1, 3]], [m_p2f[2, 3]]])

        coordi_phys = np.transpose(np.asarray(coordi))
        coordi_pix = np.transpose(np.dot(m_f2p_transfo, (coordi_phys - coord_origin)))
        coordi_pix_tmp = coordi_pix.tolist()
        coordi_pix_list = [[int(np.round(coordi_pix_tmp[j][i])) for i in range(len(coordi_pix_tmp[j]))] for j in range(len(coordi_pix_tmp))]

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

        m_p2f = self.hdr.get_sform()
        m_p2f_transfo = m_p2f[0:3, 0:3]
        m_f2p_transfo = np.linalg.inv(m_p2f_transfo)

        coord_origin = np.array([[m_p2f[0, 3]], [m_p2f[1, 3]], [m_p2f[2, 3]]])

        if coordi is not None:
            coordi_phys = np.transpose(np.asarray(coordi))
            coordi_pix = np.transpose(np.dot(m_f2p_transfo, (coordi_phys - coord_origin)))
            coordi_pix_tmp = coordi_pix.tolist()
            coordi_pix_list = [[coordi_pix_tmp[j][i] for i in range(len(coordi_pix_tmp[j]))] for j in
                               range(len(coordi_pix_tmp))]

            return coordi_pix_list

    def get_values(self, coordi=None, interpolation_mode=0, border='constant', cval=0.0):
        """
        This function returns the intensity value of the image at the position coordi (can be a list of coordinates).
        :param coordi: continuouspix
        :param interpolation_mode: 0=nearest neighbor, 1= linear, 2= 2nd-order spline, 3= 2nd-order spline, 4= 2nd-order spline, 5= 5th-order spline
        :return: intensity values at continuouspix with interpolation_mode
        """
        return map_coordinates(self.data, coordi, output=np.float32, order=interpolation_mode, mode=border, cval=cval)

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

    def get_directions(self):
        """
        This function return the X, Y, and Z axes of the image
        Returns:
            X, Y and Z axes of the image
        """
        direction_matrix = self.header.get_best_affine()
        T_self, R_self, Sc_self, Sh_self = decompose_affine_transform(direction_matrix)
        return R_self[0:3, 0], R_self[0:3, 1], R_self[0:3, 2]

    def interpolate_from_image(self, im_ref, fname_output=None, interpolation_mode=1, border='constant'):
        """
        This function interpolates an image by following the grid of a reference image.
        Example of use:

        from msct_image import Image
        im_input = Image(fname_input)
        im_ref = Image(fname_ref)
        im_input.interpolate_from_image(im_ref, fname_output, interpolation_mode=1)

        :param im_ref: reference Image that contains the grid on which interpolate.
        :param border: Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap')
        :return: a new image that has the same dimensions/grid of the reference image but the data of self image.
        """
        nx, ny, nz, nt, px, py, pz, pt = im_ref.dim
        x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
        indexes_ref = np.array(list(zip(x.ravel(), y.ravel(), z.ravel())))
        physical_coordinates_ref = im_ref.transfo_pix2phys(indexes_ref)

        # TODO: add optional transformation from reference space to image space to physical coordinates of ref grid.
        # TODO: add choice to do non-full transorm: translation, (rigid), affine
        # 1. get transformation
        # 2. apply transformation on coordinates

        coord_im = np.array(self.transfo_phys2continuouspix(physical_coordinates_ref))
        interpolated_values = self.get_values(np.array([coord_im[:, 0], coord_im[:, 1], coord_im[:, 2]]), interpolation_mode=interpolation_mode, border=border)

        im_output = Image(im_ref)
        if interpolation_mode == 0:
            im_output.change_type('int32')
        else:
            im_output.change_type('float32')
        im_output.data = np.reshape(interpolated_values, (nx, ny, nz))
        if fname_output is not None:
            im_output.absolutepath = fname_output
            im_output.save()
        return im_output


def compute_dice(image1, image2, mode='3d', label=1, zboundaries=False):
    """
    This function computes the Dice coefficient between two binary images.
    Args:
        image1: object Image
        image2: object Image
        mode: mode of computation of Dice.
                3d: compute Dice coefficient over the full 3D volume
                2d-slices: compute the 2D Dice coefficient for each slice of the volumes
        label: binary label for which Dice coefficient will be computed. Default=1
        zboundaries: True/False. If True, the Dice coefficient is computed over a Z-ROI where both segmentations are
                     present. Default=False.

    Returns: Dice coefficient as a float between 0 and 1. Raises ValueError exception if an error occurred.

    """
    MODES = ['3d', '2d-slices']
    if mode not in MODES:
        raise ValueError('\n\nERROR: mode must be one of these values:' + ',  '.join(MODES))

    dice = 0.0  # default value of dice is 0

    # check if images are in the same coordinate system
    assert image1.data.shape == image2.data.shape, "\n\nERROR: the data (" + image1.absolutepath + " and " + image2.absolutepath + ") don't have the same size.\nPlease use  \"sct_register_multimodal -i im1.nii.gz -d im2.nii.gz -identity 1\"  to put the input images in the same space"

    # if necessary, change orientation of images to RPI and compute segmentation boundaries
    if mode == '2d-slices' or (mode == '3d' and zboundaries):
        # changing orientation to RPI if necessary
        if image1.orientation != 'RPI':
            image1 = change_orientation(image1, "RPI")
        if image2.orientation != 'RPI':
            image2 = change_orientation(image2, "RPI")

        zmin, zmax = 0, image1.data.shape[2] - 1
        if zboundaries:
            # compute Z-ROI for which both segmentations are present.
            for z in range(zmin, zmax + 1):  # going from inferior to superior
                if np.any(image1.data[:, :, z]) and np.any(image2.data[:, :, z]):
                    zmin = z
                    break
            for z in range(zmax, zmin + 1, -1):  # going from superior to inferior
                if np.any(image1.data[:, :, z]) and np.any(image2.data[:, :, z]):
                    zmax = z
                    break

        if zmin > zmax:
            # segmentations do not overlap
            return 0.0

        if mode == '3d':
            # compute dice coefficient over Z-ROI
            data1 = image1.data[:, :, zmin:zmax]
            data2 = image2.data[:, :, zmin:zmax]

            dice = np.sum(data2[data1 == label]) * 2.0 / (np.sum(data1) + np.sum(data2))

        elif mode == '2d-slices':
            raise ValueError('2D slices Dice coefficient feature is not implemented yet')

    elif mode == '3d':
        # compute 3d dice coefficient
        dice = np.sum(image2.data[image1.data == label]) * 2.0 / (np.sum(image1.data) + np.sum(image2.data))

    return dice


def find_zmin_zmax(im, threshold=0.1):
    """
    Find the min (and max) z-slice index below which (and above which) slices only have voxels below a given threshold.
    :param im: Image object
    :param threshold: threshold to apply before looking for zmin/zmax, typically corresponding to noise level.
    :return: [zmin, zmax]
    """
    slicer = Slicer(im, axis="IS")

    # Iterate from bottom to top until we find data
    for zmin in slicer.range("IS"):
        dataz = im.data[slicer.slice(zmin)]
        if np.any(dataz > threshold):
            break

    # Conversely from top to bottom
    for zmax in slicer.range("SI"):
        dataz = im.data[slicer.slice(zmax)]
        if np.any(dataz > threshold):
            break

    return zmin, zmax


def get_dimension(im_file, verbose=1):
    """
    Get dimension from nibabel object. Manages 2D, 3D or 4D images.
    :return: nx, ny, nz, nt, px, py, pz, pt
    """
    import nibabel.nifti1
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


def all_refspace_strings():
    """
    :return: all possible orientation strings ['RAI', 'RAS', 'RPI', 'RPS', ...]
    """
    return [x for x in itertools.chain(*[ [ "".join(x) for x in itertools.product(*seq) ] for seq in itertools.permutations(("RL", "AP", "IS"), 3)])]


def get_orientation(im):
    """
    :param im: an Image
    :return: reference space string (ie. what's in Image.orientation)
    """
    res = "".join(nibabel.orientations.aff2axcodes(im.hdr.get_best_affine()))
    return orientation_string_nib2sct(res)
    return res # for later ;)


def orientation_string_nib2sct(s):
    """
    :return: SCT reference space code from nibabel one
    """
    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
    return "".join([opposite_character[x] for x in s])


orientation_string_sct2nib = orientation_string_nib2sct


def change_shape(im_src, shape, im_dst=None):
    """
    :return: an image with changed shape
    :param shape: shape to obtain (must be compatible with original one)

    Notes:

    - the resulting image has no path
    """

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    if im_src.data.flags.f_contiguous:
        im_dst.data = im_src.data.reshape(shape, order="F")
    elif im_src.data.flags.c_contiguous:
        warnings.warn("Encountered an array with C order, strange!")
        im_dst.data = im_src.data.reshape(shape, order="C")
    else:
        # image data may be a view
        im_dst_data = im_src_data.copy().reshape(shape, order="F")

    pair = nibabel.nifti1.Nifti1Pair(im_dst.data, im_dst.hdr.get_best_affine(), im_dst.hdr)
    im_dst.hdr = pair.header
    return im_dst

def change_orientation(im_src, orientation, im_dst=None, inverse=False):
    """
    :return: an image with changed orientation

    Note: the resulting image has no path
    """

    if len(im_src.data.shape) > 3:
        raise NotImplementedError("Use sct_image.orientation in that case")

    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

    im_src_orientation = orientation_string_sct2nib(im_src.orientation)
    im_dst_orientation = orientation_string_sct2nib(orientation)
    if inverse:
        im_src_orientation, im_dst_orientation = im_dst_orientation, im_src_orientation


    def get_permutations(im_src_orientation, im_dst_orientation):
        # change the orientation of the image
        perm = [0, 1, 2]
        inversion = [1, 1, 1]
        for i, character in enumerate(im_src_orientation):
            try:
                perm[i] = im_dst_orientation.index(character)
            except ValueError:
                perm[i] = im_dst_orientation.index(opposite_character[character])
                inversion[i] = -1

        return perm, inversion

    perm, inversion = get_permutations(im_src_orientation, im_dst_orientation)

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    # Update data by performing inversions and swaps

    # axes inversion
    data = im_src.data[::inversion[0], ::inversion[1], ::inversion[2]]

    # axes manipulations
    if perm == [1, 0, 2]:
        data = np.swapaxes(data, 0, 1)
    elif perm == [2, 1, 0]:
        data = np.swapaxes(data, 0, 2)
    elif perm == [0, 2, 1]:
        data = np.swapaxes(data, 1, 2)
    elif perm == [2, 0, 1]:
        data = np.swapaxes(data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
        data = np.swapaxes(data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
    elif perm == [1, 2, 0]:
        data = np.swapaxes(data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
        data = np.swapaxes(data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
    elif perm == [0, 1, 2]:
        # do nothing
        pass
    else:
        raise NotImplementedError()


    # Update header

    im_src_aff = im_src.hdr.get_best_affine()
    aff = nibabel.orientations.inv_ornt_aff(np.array((perm, inversion)).T, im_src.data.shape)
    im_dst_aff = np.matmul(im_src_aff, aff)

    im_dst.header.set_qform(im_dst_aff)
    im_dst.header.set_sform(im_dst_aff)
    im_dst.header.set_data_shape(data.shape)
    im_dst.data = data

    return im_dst

def change_type(im_src, dtype, im_dst=None):
    """
    Change the voxel type of the image
    :param dtype:    if not set, the image is saved in standard type
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

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    if dtype is None:
        return im_dst

    # get min/max from input image
    min_in = np.nanmin(im_src.data)
    max_in = np.nanmax(im_src.data)

    # find optimum type for the input image
    if dtype in ('minimize', 'minimize_int'):
        # warning: does not take intensity resolution into account, neither complex voxels

        # check if voxel values are real or integer
        isInteger = True
        if dtype == 'minimize':
            for vox in self.data.flatten():
                if int(vox) != vox:
                    isInteger = False
                    break

        if isInteger:
            if min_in >= 0:  # unsigned
                if max_in <= np.iinfo(np.uint8).max:
                    dtype = np.uint8
                elif max_in <= np.iinfo(np.uint16):
                    dtype = np.uint16
                elif max_in <= np.iinfo(np.uint32).max:
                    dtype = np.uint32
                elif max_in <= np.iinfo(np.uint64).max:
                    dtype = np.uint64
                else:
                    raise ValueError("Maximum value of the image is to big to be represented.")
            else:
                if max_in <= np.iinfo(np.int8).max and min_in >= np.iinfo(np.int8).min:
                    dtype = np.int8
                elif max_in <= np.iinfo(np.int16).max and min_in >= np.iinfo(np.int16).min:
                    dtype = np.int16
                elif max_in <= np.iinfo(np.int32).max and min_in >= np.iinfo(np.int32).min:
                    dtype = np.int32
                elif max_in <= np.iinfo(np.int64).max and min_in >= np.iinfo(np.int64).min:
                    dtype = np.int64
                else:
                    raise ValueError("Maximum value of the image is to big to be represented.")
        else:
            # if max_in <= np.finfo(np.float16).max and min_in >= np.finfo(np.float16).min:
            #    type = 'np.float16' # not supported by nibabel
            if max_in <= np.finfo(np.float32).max and min_in >= np.finfo(np.float32).min:
                dtype = np.float32
            elif max_in <= np.finfo(np.float64).max and min_in >= np.finfo(np.float64).min:
                dtype = np.float64

        dtype = to_dtype(dtype)
    else:
        dtype = to_dtype(dtype)

        # if output type is int, check if it needs intensity rescaling
        if "int" in dtype.name:
            # get min/max from output type
            min_out = np.iinfo(dtype).min
            max_out = np.iinfo(dtype).max
            # before rescaling, check if there would be an intensity overflow

            if (min_in < min_out) or (max_in > max_out):
                # This condition is important for binary images since we do not want to scale them
                sct.printv('WARNING: To avoid intensity overflow due to convertion to '+dtype.name+', intensity will be rescaled to the maximum quantization scale.', 1, 'warning')
                # rescale intensity
                data_rescaled = im_src.data * (max_out - min_out) / (max_in - min_in)
                im_dst.data = data_rescaled - ( data_rescaled.min() - min_out )

    # change type of data in both numpy array and nifti header
    im_dst.data = getattr(np, dtype.name)(im_dst.data)
    im_dst.hdr.set_data_dtype(dtype)
    return im_dst


def to_dtype(dtype):
    """
    Take a dtypeification and return an np.dtype

    :param dtype: dtypeification (string or np.dtype or None are supported for now)
    :return: dtype or None
    """
    # TODO add more or filter on things supported by nibabel

    if dtype is None:
        return None
    if isinstance(dtype, type):
        try:
            if isinstance(dtype(0).dtype, np.dtype):
                 return dtype(0).dtype
        except: # TODO
            raise
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, str):
        return np.dtype(dtype)

    raise TypeError("data type {}: {} not understood".format(dtype.__class__, dtype))

def zeros_like(img, dtype=None):
    dst = change_type(img, dtype)
    dst.data[:] = 0
    return dst

def empty_like(img, dtype=None):
    dst = change_type(img, dtype)
    return dst
