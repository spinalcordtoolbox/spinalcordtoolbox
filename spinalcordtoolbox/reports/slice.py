# -*- coding: utf-8 -*-

# TODO: Replace slice by spinalcordtoolbox.image.Slicer


from __future__ import print_function, absolute_import, division

import abc
import logging
import math

import numpy as np
from scipy import ndimage

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.resampling import resample_nib
from nibabel.nifti1 import Nifti1Image

logger = logging.getLogger(__name__)


class Slice(object):
    """Abstract class representing slicing applied to >=1 volumes for the purpose
    of generating ROI slices.

    Typically, the first volumes are images, while the last volume is a segmentation, which is used as overlay on the
    image, and/or to retrieve the center of mass to center the image on each QC mosaic square.

    For convenience, the volumes are all brought in the SAL reference frame.

    Functions with the suffix `_slice` gets a slice cut in the desired axis at the
    "i" position of the data of the 3D image. While the functions with the suffix
    `_dim` gets the size of the desired dimension of the 3D image.

    IMPORTANT: Convention for orientation is "SAL"

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, images, p_resample=0.6):
        """
        :param images: list of 3D volumes to be separated into slices.
        """
        logger.info('Resample images to {}x{} mm'.format(p_resample, p_resample))
        self._images = list()
        image_ref = None  # first pass: we don't have a reference image to resample to
        for i, image in enumerate(images):
            img = image.copy()
            img.change_orientation('SAL')
            if p_resample:
                if i == len(images) - 1:
                    # Last volume corresponds to a segmentation, therefore use linear interpolation here
                    type_img = 'seg'
                else:
                    # Otherwise it's an image: use spline interpolation
                    type_img = 'im'
                img_r = self._resample_slicewise(img, p_resample, type_img=type_img, image_ref=image_ref)
            else:
                img_r = img.copy()
            self._images.append(img_r)
            image_ref = self._images[0]  # 2nd and next passes: we resample any image to the space of the first one

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

        :param image: input Image
        :returns: numpy.ndarray
        """
        return

    def _axial_center(self, image):
        """Gets the center of mass in the axial plan

        :param image : input Image
        :returns: centers of mass in the x and y axis (tuple of numpy.ndarray of int)
        """
        logger.info('Compute center of mass at each slice')
        data = np.array(image.data)  # we cast np.array to overcome problem if inputing nii format
        nz = image.dim[0]  # SAL orientation
        centers_x = np.zeros(nz)
        centers_y = np.zeros(nz)
        for i in range(nz):
            centers_x[i], centers_y[i] = ndimage.measurements.center_of_mass(data[i, :, :])
        try:
            Slice.nan_fill(centers_x)
            Slice.nan_fill(centers_y)
        except ValueError as err:
            logger.error("Axial center of the spinal cord is not found: %s", err)
            raise
        return centers_x, centers_y

    def mosaic(self, nb_column=0, size=15, return_center=False):
        """Obtain matrices of the mosaics

        Calculates how many squares will fit in a row based on the column and the size
        Multiply by 2 because the sides are of size*2. Central point is size +/-.

        :param nb_column: number of mosaic columns
        :param size: each column size
        :return: tuple of numpy.ndarray containing the mosaics of each slice pixels
        :return: list of tuples, each tuple representing the center of each square of the mosaic. Only with param return_center is True
        """

        # Calculate number of columns to display on the report
        dim = self.get_dim(self._images[0])  # dim represents the 3rd dimension of the 3D matrix
        if nb_column == 0:
            nb_column = 600 // (size * 2)

        nb_row = math.ceil(dim // nb_column) + 1

        # Compute the matrix size of the final mosaic image
        matrix_sz = (int(size * 2 * nb_row), int(size * 2 * nb_column))

        centers_mosaic = []
        for irow in range(nb_row):
            for icol in range(nb_column):
                centers_mosaic.append((icol * size * 2 + size, irow * size * 2 + size))

        # Get center of mass for each slice of the image. If the input is the cord segmentation, these coordinates are
        # used to center the image on each panel of the mosaic.
        centers_x, centers_y = self.get_center()

        matrices = list()
        for image in self._images:
            matrix = np.zeros(matrix_sz)
            for i in range(dim):
                x = int(centers_x[i])
                y = int(centers_y[i])
                # crop slice around center of mass and add slice to the matrix layout
                # TODO: resample there after cropping based on physical dimensions
                self.add_slice(matrix, i, nb_column, size, self.crop(self.get_slice(image.data, i), x, y, size, size))

            matrices.append(matrix)
        if return_center is True:
            return matrices, centers_mosaic
        else:
            return matrices

    def single(self):
        """Obtain the matrices of the single slices. Flatten

        :returns: tuple of numpy.ndarray, matrix of the input 3D MRI
                  containing the slices and matrix of the transformed 3D MRI
                  to output containing the slices
        """
        assert len(set([x.data.shape for x in self._images])) == 1, "Volumes don't have the same size"

        image = self._images[0]
        dim = self.get_dim(image)

        matrices = list()
        index = self.get_center_spit()
        for image in self._images:
            # Fetch mid-sagittal plane
            matrix = self.get_slice(image.data, dim / 2)
            for j in range(len(index)):
                # For each slice, translate in the R-L direction to center the cord
                matrix[j] = self.get_slice(image.data, int(np.round(index[j])))[j]
            matrices.append(matrix)

        return matrices

    def aspect(self):
        return [self.get_aspect(x) for x in self._images]

    def _resample_slicewise(self, image, p_resample, type_img, image_ref=None):
        """
        Resample at a fixed resolution to make sure the cord always appears with similar scale, regardless of the native
        resolution of the image. Assumes SAL orientation.
        :param image: Image() to resample
        :param p_resample: float: Resampling resolution in mm
        :param type_img: {'im', 'seg'}: If im, interpolate using spline. If seg, interpolate using linear then binarize.
        :param image_ref: Destination Image() to resample image to.
        :return:
        """
        dict_interp = {'im': 'spline', 'seg': 'linear'}
        # Create nibabel object
        nii = Nifti1Image(image.data, image.hdr.get_best_affine())
        # If no reference image is provided, resample to specified resolution
        if image_ref is None:
            # Resample to px x p_resample x p_resample mm (orientation is SAL by convention in QC module)
            nii_r = resample_nib(nii, new_size=[image.dim[4], p_resample, p_resample], new_size_type='mm',
                                 interpolation=dict_interp[type_img])
        # Otherwise, resampling to the space of the reference image
        else:
            # Create nibabel object for reference image
            nii_ref = Nifti1Image(image_ref.data, image_ref.hdr.get_best_affine())
            nii_r = resample_nib(nii, image_dest=nii_ref, interpolation=dict_interp[type_img])
        # If resampled image is a segmentation, binarize using threshold at 0.5
        if type_img == 'seg':
            img_r_data = (nii_r.get_data() > 0.5) * 1
        else:
            img_r_data = nii_r.get_data()
        # Create Image objects
        image_r = Image(img_r_data, hdr=nii_r.header, dim=nii_r.header.get_data_shape()). \
            change_orientation(image.orientation)
        return image_r


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
        """Retrieve index of the medial plane (in the R-L direction) for each slice (in the I-S direction) in order
        to center the spinal cord in the sagittal view.
        Exception: if the input mask only has a single label (e.g., for sct_detect_pmj), then output the index that has
        the sagittal slice centered at that label."""
        image = self._images[img_idx].copy()
        # If mask is empty, raise error
        if np.argwhere(image.data).shape[0] == 0:
            logging.error('Mask is empty')
        # If mask only has one label (e.g., in sct_detect_pmj), return the repmat of the R-L index (assuming SAL orient)
        elif np.argwhere(image.data).shape[0] == 1:
            return [np.argwhere(image.data)[0][2]] * image.data.shape[2]
        # Otherwise, find the center of mass per slice and return the R-L index
        else:
            from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
            image.change_orientation('RPI')  # need to do that because get_centerline operates in RPI orientation
            # Get coordinate of centerline
            _, arr_ctl_RPI, _, _ = get_centerline(image, param=ParamCenterline())
            # Extend the centerline by copying values below zmin and above zmax to avoid discontinuities
            zmin, zmax = arr_ctl_RPI[2, :].min().astype(int), arr_ctl_RPI[2, :].max().astype(int)
            index_RL_in_RPI = np.concatenate([np.ones(zmin) * arr_ctl_RPI[0, 0],
                                              arr_ctl_RPI[0, 1:],
                                              np.ones(image.data.shape[2] - zmax) * arr_ctl_RPI[0, -1]])
            # reorient R-L index to go from RPI to SAL
            index_RL_in_SAL = image.data.shape[0] - index_RL_in_RPI
            # then reverse to go from RL to LR
            index_RL_in_SAL = index_RL_in_SAL[::-1]
            return index_RL_in_SAL

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
