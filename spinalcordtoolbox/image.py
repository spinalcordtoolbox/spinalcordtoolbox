#!/usr/bin/env python
#########################################################################################
#
# SCT Image API
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: Sort out the use of Image.hdr and Image.header --> they seem to carry duplicated information.

from __future__ import division, absolute_import

import sys, os, itertools, warnings, logging

import nibabel
import nibabel.orientations

import numpy as np
from scipy.ndimage import map_coordinates

import transforms3d.affines as affines
from spinalcordtoolbox.types import Coordinate
from spinalcordtoolbox.utils import sct_dir_local_path, add_suffix

sys.path.append(sct_dir_local_path('scripts'))
import sct_utils as sct # [AJ] FIXME

logger = logging.getLogger(__name__)


def _get_permutations(im_src_orientation, im_dst_orientation):
    """

    :param im_src_orientation str: Orientation of source image. Example: 'RPI'
    :param im_dest_orientation str: Orientation of destination image. Example: 'SAL'
    :return: list of axes permutations and list of inversions to achieve an orientation change
    """

    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

    perm = [0, 1, 2]
    inversion = [1, 1, 1]
    for i, character in enumerate(im_src_orientation):
        try:
            perm[i] = im_dst_orientation.index(character)
        except ValueError:
            perm[i] = im_dst_orientation.index(opposite_character[character])
            inversion[i] = -1

    return perm, inversion


class Slicer(object):
    """
    Provides a sliced view onto original image data.
    Can be used as a sequence.

    Notes:

    - The original image data is directly available without copy,
      which is a nice feature, not a bug! Use .copy() if you need copies...

    Example:

    .. code:: python

       for slice2d in msct_image.SlicerFancy(im3d, "RPI"):
           print(slice)

    """
    def __init__(self, im, orientation="LPI"):
        """

        :param im: image to iterate through
        :param spec: "from" letters to indicate how to slice the image.
                     The slices are done on the last letter axis,
                     and they are defined as the first/second letter.
        """

        if not isinstance(im, Image):
            raise ValueError("Expecting an image")
        if not orientation in all_refspace_strings():
            raise ValueError("Invalid orientation spec")

        # Get a different view on data, as if we were doing a reorientation

        perm, inversion = _get_permutations(im.orientation, orientation)

        # axes inversion (flip)
        data = im.data[::inversion[0], ::inversion[1], ::inversion[2]]

        # axes manipulations (transpose)
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

        self._data = data
        self._orientation = orientation
        self._nb_slices = data.shape[2]

    def __len__(self):
       return self._nb_slices

    def __getitem__(self, idx):
       """

       :return: an image slice, at slicing index idx
       :param idx: slicing index (according to the slicing direction)
       """
       if not isinstance(idx, int):
           raise NotImplementedError()

       if idx >= self._nb_slices:
           raise IndexError("I just have {} slices!".format(self._nb_slices))

       return self._data[:,:,idx]


class SlicerOneAxis(object):
    """
    Image slicer to use when you don't care about the 2D slice orientation,
    and don't want to specify them.
    The slicer will just iterate through the right axis that corresponds to
    its specification.

    Can help getting ranges and slice indices.
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

       return self.im.data[self._slice(idx)]


class SlicerMany(object):
    """
    Image*s* slicer utility class.

    Can help getting ranges and slice indices.
    Can provide slices (being an *iterator*).

    Use with great care for now, that it's not very documented.
    """
    def __init__(self, images, slicerclass, *args, **kw):
        if len(images) == 0:
            raise ValueError("Don't expect me to work on 0 images!")

        self.slicers = [ slicerclass(im, *args, **kw) for im in images ]

        nb_slices = [ x._nb_slices for x in self.slicers ]
        if len(set(nb_slices)) != 1:
            raise ValueError("All images must have the same number of slices along the slicing axis!")
        self._nb_slices = nb_slices[0]

    def __len__(self):
        return self._nb_slices

    def __getitem__(self, idx):
        return [ x[idx] for x in self.slicers ]


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
            self._path = os.path.abspath(absolutepath)

        self.verbose = verbose

        # load an image from file
        if isinstance(param, str) or (sys.hexversion < 0x03000000 and isinstance(param, unicode)):
            self.loadFromPath(param, verbose)
        # copy constructor
        elif isinstance(param, type(self)):
            self.copy(param)
        # create an empty image (full of zero) of dimension [dim]. dim must be [x,y,z] or (x,y,z). No header.
        elif isinstance(param, list):
            self.data = np.zeros(param)
            self.hdr = hdr
        # create a copy of im_ref
        elif isinstance(param, (np.ndarray, np.generic)):
            self.data = param
            self.hdr = hdr
        else:
            raise TypeError('Image constructor takes at least one argument.')

        # TODO: In the future, we might want to check qform_code and enforce its value. Related to #2454
        # Check qform_code
        # if not self.hdr['qform_code'] in [0, 1]:
        #     # Set to 0 (unknown)
        #     self.hdr.set_qform(self.hdr.get_qform(), code=0)
        #     self.header.set_qform(self.hdr.get_qform(), code=0)

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
            self._path = deepcopy(image._path)
        else:
            return deepcopy(self)

    def copy_qform_from_ref(self, im_ref):
        """
        Copy qform and sform and associated codes from a reference Image object

        :param im_ref:
        :return:
        """
        # Copy q/sform and code
        self.hdr.set_qform(im_ref.hdr.get_qform())
        self.hdr._structarr['qform_code'] = im_ref.hdr._structarr['qform_code']
        self.hdr.set_sform(im_ref.hdr.get_sform())
        self.hdr._structarr['sform_code'] = im_ref.hdr._structarr['sform_code']

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
        self.hdr = self.im_file.header
        self.absolutepath = path
        if path != self.absolutepath:
            logger.debug("Loaded %s (%s) orientation %s shape %s", path, self.absolutepath, self.orientation, self.data.shape)
        else:
            logger.debug("Loaded %s orientation %s shape %s", path, self.orientation, self.data.shape)

    def change_shape(self, shape, generate_path=False):
        """
        Change data shape (in-place)

        :param generate_path: whether to create a derived path name from the\
                              original absolutepath (note: while it will generate\
                              a file suffix, don't expect the suffix but rather\
                              use the Image's absolutepath.\
                              If not set, the absolutepath is voided.

        This is mostly useful for adding/removing a fourth dimension,
        you probably don't want to use this function.

        """
        if shape is not None:
            change_shape(self, shape, self)

        if generate_path and self._path is not None:
            self._path = add_suffix(self._path, "_shape-{}".format("-".join([str(x) for x in shape])))
        else:
            # safe option: remove path to avoid overwrites
            self._path = None
        return self

    def change_orientation(self, orientation, inverse=False, generate_path=False):
        """
        Change orientation on image (in-place).

        :param orientation: orientation string (SCT "from" convention)

        :param inverse: if you think backwards, use this to specify that you actually\
                        want to transform *from* the specified orientation, not *to*\
                        it.
        :param generate_path: whether to create a derived path name from the\
                              original absolutepath (note: while it will generate\
                              a file suffix, don't expect the suffix but rather\
                              use the Image's absolutepath.\
                              If not set, the absolutepath is voided.

        """
        if orientation is not None:
            change_orientation(self, orientation, self, inverse=inverse)
        if generate_path and self._path is not None:
            self._path = add_suffix(self._path, "_{}".format(orientation.lower()))
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
            self._path = add_suffix(self._path, "_{}".format(dtype.name))
        else:
            # safe option: remove path to avoid overwrites
            self._path = None
        return self

    def save(self, path=None, dtype=None, verbose=1, mutable=False):
        """
        Write an image in a nifti file

        :param path: Where to save the data, if None it will be taken from the\
                     absolutepath member.\
                     If path is a directory, will save to a file under this directory\
                     with the basename from the absolutepath member.

        :param dtype: if not set, the image is saved in the same type as input data\
                      if 'minimize', image storage space is minimized\
                        (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),\
                        (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),\
                        (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),\
                        (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),\
                        (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),\
                        (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),\
                        (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),\
                        (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),\
                        (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),\
                        (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),\
                        (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),\
                        (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),\
                        (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),\
                        (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),

        :param mutable: whether to update members with newly created path or dtype
        """

        if path is None and self.absolutepath is None:
            raise RuntimeError("Don't know where to save the image (no absolutepath or path parameter)")
        elif path is not None and os.path.isdir(path) and self.absolutepath is not None:
            # Save to destination directory with original basename
            path = os.path.join(os.path.abspath(path), os.path.basename(self.absolutepath))

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
            # Update dtype if provided (but not if based on SCT-specific values: 'minimize')
            if (dtype is not None) and (dtype not in ['minimize', 'minimize_int']):
                hdr.set_data_dtype(dtype)

        # nb. that copy() is important because if it were a memory map, save()
        # would corrupt it
        img = Nifti1Image(data.copy(), None, hdr)
        if os.path.isfile(path):
            if verbose:
                logger.warning('File ' + path + ' already exists. Will overwrite it.')

        # save file
        if os.path.isabs(path):
            logger.debug("Saving image to %s orientation %s shape %s",
             path, self.orientation, data.shape)
        else:
            logger.debug("Saving image to %s (%s) orientation %s shape %s",
             path, os.path.abspath(path), self.orientation, data.shape)

        nibabel.save(img, path)

        if mutable:
            self.absolutepath = path
            self.data = data

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
            from spinalcordtoolbox.types import CoordinateValue
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


    def transfo_pix2phys(self, coordi=None):
        """
        This function returns the physical coordinates of all points of 'coordi'.

        :param coordi: sequence of (nb_points x 3) values containing the pixel coordinate of points.
        :return: sequence with the physical coordinates of the points in the space of the image.

        Example:

        .. code:: python

            img = Image('file.nii.gz')
            coordi_pix = [[1,1,1]]   # for points: (1,1,1). N.B. Important to write [[x,y,z]] instead of [x,y,z]
            coordi_pix = [[1,1,1],[2,2,2],[4,4,4]]   # for points: (1,1,1), (2,2,2) and (4,4,4)
            coordi_phys = img.transfo_pix2phys(coordi=coordi_pix)

        """

        m_p2f = self.hdr.get_best_affine()
        aug = np.hstack((np.asarray(coordi), np.ones((len(coordi), 1))))
        ret = np.empty_like(coordi, dtype=np.float64)
        for idx_coord, coord in enumerate(aug):
            phys = np.matmul(m_p2f, coord)
            ret[idx_coord] = phys[:3]
        return ret


    def transfo_phys2pix(self, coordi, real=True):
        """
        This function returns the pixels coordinates of all points of 'coordi'

        :param coordi: sequence of (nb_points x 3) values containing the pixel coordinate of points.
        :param real: whether to return real pixel coordinates
        :return: sequence with the physical coordinates of the points in the space of the image.
        """

        m_p2f = self.hdr.get_best_affine()
        m_f2p = np.linalg.inv(m_p2f)
        aug = np.hstack((np.asarray(coordi), np.ones((len(coordi), 1))))
        ret = np.empty_like(coordi, dtype=np.float64)
        for idx_coord, coord in enumerate(aug):
            phys = np.matmul(m_f2p, coord)
            ret[idx_coord] = phys[:3]
        if real:
            return np.int32(np.round(ret))
        else:
            return ret


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
            T_self, R_self, Sc_self, Sh_self = affines.decompose44(aff_im_self)
            T_ref, R_ref, Sc_ref, Sh_ref = affines.decompose44(aff_im_ref)
            if mode == 'translation':
                T_transform = T_ref - T_self
                R_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid':
                T_transform = T_ref - T_self
                R_transform = np.matmul(np.linalg.inv(R_self), R_ref)
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid_scaling':
                T_transform = T_ref - T_self
                R_transform = np.matmul(np.linalg.inv(R_self), R_ref)
                Sc_transform = Sc_ref / Sc_self
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            else:
                transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return transform

    def get_inverse_transform(self, im_ref, mode='affine'):
        aff_im_self = self.im_file.affine
        aff_im_ref = im_ref.im_file.affine
        if mode == 'affine':
            transform = np.matmul(np.linalg.inv(aff_im_ref), aff_im_self)
        else:
            T_self, R_self, Sc_self, Sh_self = affines.decompose44(aff_im_self)
            T_ref, R_ref, Sc_ref, Sh_ref = affines.decompose44(aff_im_ref)
            if mode == 'translation':
                T_transform = T_self - T_ref
                R_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid':
                T_transform = T_self - T_ref
                R_transform = np.matmul(np.linalg.inv(R_ref), R_self)
                Sc_transform = np.array([1.0, 1.0, 1.0])
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            elif mode == 'rigid_scaling':
                T_transform = T_self - T_ref
                R_transform = np.matmul(np.linalg.inv(R_ref), R_self)
                Sc_transform = Sc_self / Sc_ref
                transform = affines.compose(T_transform, R_transform, Sc_transform)
            else:
                transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        return transform

    def get_directions(self):
        """
        This function return the X, Y, and Z axes of the image

        return: X, Y and Z axes of the image
        """
        direction_matrix = self.header.get_best_affine()
        T_self, R_self, Sc_self, Sh_self = affines.decompose44(direction_matrix)
        return R_self[0:3, 0], R_self[0:3, 1], R_self[0:3, 2]

    def interpolate_from_image(self, im_ref, fname_output=None, interpolation_mode=1, border='constant'):
        """
        This function interpolates an image by following the grid of a reference image.
        Example of use:

        .. code:: python

            from spinalcordtoolbox.image import Image
            im_input = Image(fname_input)
            im_ref = Image(fname_ref)
            im_input.interpolate_from_image(im_ref, fname_output, interpolation_mode=1)

        :param im_ref: reference Image that contains the grid on which interpolate.
        :param border: Points outside the boundaries of the input are filled according\
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

        coord_im = self.transfo_phys2pix(physical_coordinates_ref, real=False)
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

    def mean(self, dim):
        """
        Average across specified dimension

        :param dim: int: axis used for averaging
        :return: Image object
        """
        im_out = empty_like(self)
        im_out.data = np.mean(self.data, dim)
        # TODO: the line below fails because .dim is immutable. We should find a solution to update dim accordingly
        #  because as of now, this field contains wrong values (in this case, the dimension should be changed)
        # im_out.dim = im_out.data.shape[:dim] + (1,) + im_out.data.shape[dim:]
        return im_out


def compute_dice(image1, image2, mode='3d', label=1, zboundaries=False):
    """
    This function computes the Dice coefficient between two binary images.
    :param image1: object Image
    :param image2: object Image
    :param mode: mode of computation of Dice.\
            3d: compute Dice coefficient over the full 3D volume\
            2d-slices: compute the 2D Dice coefficient for each slice of the volumes\
    :param: label: binary label for which Dice coefficient will be computed. Default=1
    :paaram: zboundaries: True/False. If True, the Dice coefficient is computed over a Z-ROI where both segmentations are\
                 present. Default=False.

    :return: Dice coefficient as a float between 0 and 1. Raises ValueError exception if an error occurred.
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


def concat_data(fname_in_list, dim, pixdim=None, squeeze_data=False):
    """
    Concatenate data

    :param im_in_list: list of Images or image filenames
    :param dim: dimension: 0, 1, 2, 3.
    :param pixdim: pixel resolution to join to image header
    :param squeeze_data: bool: if True, remove the last dim if it is a singleton.
    :return im_out: concatenated image
    """
    # WARNING: calling concat_data in python instead of in command line causes a non-understood issue (results are
    # different with both options) from numpy import concatenate, expand_dims

    dat_list = []
    data_concat_list = []

    for i, fname in enumerate(fname_in_list):
        # if there is more than 100 images to concatenate, then it does it iteratively to avoid memory issue.
        if i != 0 and i % 100 == 0:
            data_concat_list.append(np.concatenate(dat_list, axis=dim))
            im = Image(fname)
            dat = im.data
            # if image shape is smaller than asked dim, then expand dim
            if len(dat.shape) <= dim:
                dat = np.expand_dims(dat, dim)
            dat_list = [dat]
            del im
            del dat
        else:
            im = Image(fname)
            dat = im.data
            # if image shape is smaller than asked dim, then expand dim
            if len(dat.shape) <= dim:
                dat = np.expand_dims(dat, dim)
            dat_list.append(dat)
            del im
            del dat
    if data_concat_list:
        data_concat_list.append(np.concatenate(dat_list, axis=dim))
        data_concat = np.concatenate(data_concat_list, axis=dim)
    else:
        data_concat = np.concatenate(dat_list, axis=dim)
    # write file
    im_out = empty_like(Image(fname_in_list[0]))
    im_out.data = data_concat
    if isinstance(fname_in_list[0], str):
        im_out.absolutepath = add_suffix(fname_in_list[0], '_concat')
    else:
        if fname_in_list[0].absolutepath:
            im_out.absolutepath = add_suffix(fname_in_list[0].absolutepath, '_concat')

    if pixdim is not None:
        im_out.hdr['pixdim'] = pixdim

    if squeeze_data and data_concat.shape[dim] == 1:
        # remove the last dim if it is a singleton.
        im_out.data = data_concat.reshape(
            tuple([x for (idx_shape, x) in enumerate(data_concat.shape) if idx_shape != dim]))
    else:
        im_out.data = data_concat

    # TODO: the line below fails because .dim is immutable. We should find a solution to update dim accordingly
    #  because as of now, this field contains wrong values (in this case, the dimension should be changed). Also
    #  see mean()
    # im_out.dim = im_out.data.shape[:dim] + (1,) + im_out.data.shape[dim:]

    return im_out


def find_zmin_zmax(im, threshold=0.1):
    """
    Find the min (and max) z-slice index below which (and above which) slices only have voxels below a given threshold.

    :param im: Image object
    :param threshold: threshold to apply before looking for zmin/zmax, typically corresponding to noise level.
    :return: [zmin, zmax]
    """
    slicer = SlicerOneAxis(im, axis="IS")

    # Make sure image is not empty
    if not np.any(slicer):
        logger.error('Input image is empty')

    # Iterate from bottom to top until we find data
    for zmin in range(0, len(slicer)):
        if np.any(slicer[zmin] > threshold):
            break

    # Conversely from top to bottom
    for zmax in range(len(slicer)-1, zmin, -1):
        dataz = slicer[zmax]
        if np.any(slicer[zmax] > threshold):
            break

    return zmin, zmax


def get_dimension(im_file, verbose=1):
    """
    Get dimension from Image or nibabel object. Manages 2D, 3D or 4D images.

    :param: im_file: Image or nibabel object
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
        logger.warning("The provided image file is neither a nibabel.nifti1.Nifti1Image instance nor an Image instance")

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

    :param shape: shape to obtain (must be compatible with original one)
    :return: an image with changed shape

    .. note::
        The resulting image has no path
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
        im_dst_data = im_src.data.copy().reshape(shape, order="F")

    pair = nibabel.nifti1.Nifti1Pair(im_dst.data, im_dst.hdr.get_best_affine(), im_dst.hdr)
    im_dst.hdr = pair.header
    return im_dst


def change_orientation(im_src, orientation, im_dst=None, inverse=False, data_only=False):
    """

    :param im_src: source image
    :param orientation: orientation string (SCT "from" convention)
    :param im_dst: destination image (can be the source image for in-place
                   operation, can be unset to generate one)
    :param inverse: if you think backwards, use this to specify that you actually
                    want to transform *from* the specified orientation, not *to* it.
    :param data_only: If you want to only permute the data, not the header. Only use if you know there is a problem
                      with the native orientation of the input data.
    :return: an image with changed orientation

    .. note::
        - the resulting image has no path member set
        - if the source image is < 3D, it is reshaped to 3D and the destination is 3D
    """

    # TODO: make sure to cover all cases for setorient-data
    if len(im_src.data.shape) < 3:
        pass # Will reshape to 3D
    elif len(im_src.data.shape) == 3:
        pass # OK, standard 3D volume
    elif len(im_src.data.shape) == 4:
        pass # OK, standard 4D volume
    elif len(im_src.data.shape) == 5 and im_src.header.get_intent()[0] == "vector":
        pass # OK, physical displacement field
    else:
        raise NotImplementedError("Don't know how to change orientation for this image")

    im_src_orientation = im_src.orientation
    im_dst_orientation = orientation
    if inverse:
        im_src_orientation, im_dst_orientation = im_dst_orientation, im_src_orientation


    perm, inversion = _get_permutations(im_src_orientation, im_dst_orientation)

    if im_dst is None:
        im_dst = im_src.copy()
        im_dst._path = None

    im_src_data = im_src.data
    if len(im_src_data.shape) < 3:
        im_src_data = im_src_data.reshape(tuple(list(im_src_data.shape) + ([1]*(3-len(im_src_data.shape)))))

    # Update data by performing inversions and swaps

    # axes inversion (flip)
    data = im_src_data[::inversion[0], ::inversion[1], ::inversion[2]]

    # axes manipulations (transpose)
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
    aff = nibabel.orientations.inv_ornt_aff(
     np.array((perm, inversion)).T,
     im_src_data.shape)
    im_dst_aff = np.matmul(im_src_aff, aff)

    if not data_only:
        im_dst.header.set_qform(im_dst_aff)
        im_dst.header.set_sform(im_dst_aff)
        im_dst.header.set_data_shape(data.shape)
    im_dst.data = data

    return im_dst


def change_type(im_src, dtype, im_dst=None):
    """
    Change the voxel type of the image

    :param dtype:    if not set, the image is saved in standard type\
                    if 'minimize', image space is minimize\
                    if 'minimize_int', image space is minimize and values are approximated to integers\
                    (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),\
                    (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),\
                    (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),\
                    (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),\
                    (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),\
                    (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),\
                    (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),\
                    (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),\
                    (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),\
                    (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),\
                    (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),\
                    (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),\
                    (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),\
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
            for vox in im_src.data.flatten():
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
    """

    :param img: reference image
    :param dtype: desired data type (optional)
    :return: an Image with the same shape and header, filled with zeros

    Similar to numpy.zeros_like(), the goal of the function is to show the developer's
    intent and avoid doing a copy, which is slower than initialization with a constant.

    """
    dst = change_type(img, dtype)
    dst.data[:] = 0
    return dst


def empty_like(img, dtype=None):
    """
    :param img: reference image
    :param dtype: desired data type (optional)
    :return: an Image with the same shape and header, whose data is uninitialized

    Similar to numpy.empty_like(), the goal of the function is to show the developer's
    intent and avoid touching the allocated memory, because it will be written to
    afterwards.

    """
    dst = change_type(img, dtype)
    return dst


def spatial_crop(im_src, spec, im_dst=None):
    """
    Crop an image in {0,1,2} dimension(s),
    properly altering the header to not change the physical-logical corresondance.

    :param spec: dict of dim -> [lo,hi] bounds (integer voxel coordinates)
    """

    # Compute bounds
    bounds = [ (0, x-1) for x in im_src.data.shape ]
    for k, v in spec.items():
        bounds[k] = v

    bounds_ndslice = tuple([ slice(a,b+1) for (a,b) in bounds ])

    bounds = np.array(bounds)

    # Crop data
    new_data = im_src.data[bounds_ndslice]

    # Update header
    #
    # Ref: https://mail.python.org/pipermail/neuroimaging/2017-August/001501.html
    # Given A, we want to find A' that is identical up to the intercept, such
    # that A * [x_0, y_0, z_0, 1]' == A' * [0, 0, 0, 1].
    # Conveniently, A' * [0, 0, 0, 1]' is the fourth row in the affine matrix, so
    # we're done as soon as we calculate the LHS:

    aff = im_src.header.get_best_affine()
    new_aff = aff.copy()
    new_aff[:, [3]] = aff.dot(np.vstack((bounds[:, [0]], [1])))

    new_img = nibabel.Nifti1Image(new_data, new_aff, im_src.header)

    if im_dst is None:
        im_dst = im_src.copy()

    im_dst.header = new_img.header
    im_dst.data = new_data

    return im_dst

def convert(img: Image, squeeze_data=True, dtype=None):
    """
    """
    if squeeze_data:
        img.data = np.squeeze(img.data)
    if dtype:
        img.change_type(dtype)
    return img
