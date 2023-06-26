"""
Image slice extractors for generating QC reports

Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

# TODO: Replace slice by spinalcordtoolbox.image.Slicer

import abc
import logging
import math

import numpy as np
from scipy.ndimage import center_of_mass
from nibabel.nifti1 import Nifti1Image

from spinalcordtoolbox.image import Image, split_img_data
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.cropping import ImageCropper
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline

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
        :param images: list of 3D or 4D volumes to be separated into slices.
        """
        logger.info('Resample images to {}x{} mm'.format(p_resample, p_resample))
        self._images = list()  # 3d volumes
        self._4d_images = list()  # 4d volumes
        self._image_seg = None  # for cropping
        self._absolute_paths = list()  # Used because change_orientation removes the field absolute_path
        image_ref = None  # first pass: we don't have a reference image to resample to
        for i, image in enumerate(images):
            img = image.copy()
            self._absolute_paths.append(img.absolutepath)  # change_orientation removes the field absolute_path
            img.change_orientation('SAL')
            if p_resample:
                # Check if image is a segmentation (binary or soft) by making sure:
                # - 0/1 are the two most common voxel values
                # - 0/1 account for >99% of voxels (to allow for some soft voxels)
                unique, counts = np.unique(img.data, return_counts=True)
                unique, counts = unique[np.argsort(counts)[::-1]], counts[np.argsort(counts)[::-1]]  # Sort by counts
                binary_most_common = set(unique[0:2].astype(float)) == {0.0, 1.0}
                binary_percentage = np.sum(counts[0:2]) / np.sum(counts)
                if binary_most_common and binary_percentage > 0.95:
                    # If a segmentation, use linear interpolation and apply thresholding
                    type_img = 'seg'
                else:
                    # Otherwise it's an image: use spline interpolation
                    type_img = 'im'
                img_r = self._resample_slicewise(img, p_resample, type_img=type_img, image_ref=image_ref)
            else:
                img_r = img.copy()
            if img_r.dim[3] == 1:   # If image is 3D, nt = 1
                self._images.append(img_r)
                image_ref = self._images[0]  # 2nd and next passes: we resample any image to the space of the first one
            else:
                self._4d_images.append(img_r)
                # image_ref = self._4d_images[0]  # img_dest is not covered for 4D volumes in resample_nib()

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
    def add_slice(matrix, i, column, patch):
        """Adds a slice to the canvas containing all the slices

        TODO : Move this to the Axial class

        :param matrix: input/output "big canvas"
        :param i: slice position
        :param column: number of columns in mosaic
        :param size:
        :param patch: patch to insert
        :return: matrix
        """
        start_col = (i % column) * patch.shape[1]
        end_col = start_col + patch.shape[1]

        start_row = (i // column) * patch.shape[0]
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
        x = nans.ravel().nonzero()[0]
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
            centers_x[i], centers_y[i] = center_of_mass(data[i, :, :])
        try:
            Slice.nan_fill(centers_x)
            Slice.nan_fill(centers_y)
        except ValueError as err:
            logger.error("Axial center of the spinal cord is not found: %s", err)
            raise
        return centers_x, centers_y

    @abc.abstractmethod
    def mosaic(self):
        """Obtain matrices of the mosaics"""
        return

    def single(self):
        """Obtain the matrices of the single slices. Flatten

        :returns: tuple of numpy.ndarray, matrix of the input 3D MRI
                  containing the slices and matrix of the transformed 3D MRI
                  to output containing the slices
        """
        assert len(set([x.data.shape for x in self._images])) == 1, "Volumes don't have the same size"

        matrices = list()
        # Retrieve the L-R center of the slice for each row (i.e. in the S-I direction).
        index = self.get_center_spit()
        # Loop across images and generate matrices for the image and the overlay
        for image in self._images:
            # Initialize matrix with zeros. This matrix corresponds to the 2d slice shown on the QC report.
            matrix = np.zeros(image.dim[0:2])
            for row in range(len(index)):
                # For each slice, translate in the R-L direction to center the cord
                matrix[row] = self.get_slice(image.data, int(np.round(index[row])))[row]
            matrices.append(matrix)

        return matrices

    def aspect(self):
        if len(self._4d_images) == 0:  # For 3D images
            return [self.get_aspect(x) for x in self._images]
        else:  # For 4D images
            return [self.get_aspect(x) for x in self._4d_images]

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
            # Resample each slice to p_resample x p_resample mm (orientation is SAL by convention in QC module)
            if isinstance(self, Axial):
                new_size = [image.dim[4], p_resample, p_resample]
            elif isinstance(self, Sagittal):
                new_size = [p_resample, p_resample, image.dim[6]]
            else:
                raise TypeError(f"Unexpected slice type: {type(self)}")
            nii_r = resample_nib(nii, new_size=new_size, new_size_type='mm', interpolation=dict_interp[type_img])

        # Otherwise, resampling to the space of the reference image
        else:
            # Create nibabel object for reference image
            nii_ref = Nifti1Image(image_ref.data, image_ref.hdr.get_best_affine())
            nii_r = resample_nib(nii, image_dest=nii_ref, interpolation=dict_interp[type_img])
        # If resampled image is a segmentation, binarize using threshold at 0.5 for binary segmentation
        # Apply threshold at 0.5 for non-binary segmentation
        if type_img == 'seg':
            # Check if input image is binary
            is_binary = np.isin(np.asanyarray(nii.dataobj), [0, 1]).all()
            img_r_data = np.asanyarray(nii_r.dataobj)
            if is_binary:
                img_r_data = (img_r_data > 0.5) * 1
            else:
                img_r_data[img_r_data < 0.5] = 0
        else:
            img_r_data = np.asanyarray(nii_r.dataobj)
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

    def get_center(self, img_idx=-1):
        """Get the center of mass of each slice. For 4D images, segmentation is placed in self.image_seg.
        For 3D images, by default, it assumes that self._images is a list of images, and the
        last item is the segmentation from which the center of mass is computed."""
        if self._image_seg is None:  # For 3D images
            image = self._images[img_idx]
        else:  # For 4D images
            image = self._image_seg
        return self._axial_center(image)

    def mosaic(self, return_center=False):
        """Obtain matrices of the mosaics

        Calculates how many squares will fit in a row based on the column and the size
        Multiply by 2 because the sides are of size*2. Central point is size +/-.

        :param return_center: bool, whether to return the center of each square of the mosaic.
        :return: tuple of numpy.ndarray containing the mosaics of each slice pixels
        :return: list of tuples, each tuple representing the center of each square of the mosaic.
        """

        # Calculate number of columns to display on the report
        dim = self.get_dim(self._images[0])  # dim represents the 3rd dimension of the 3D matrix
        size = 15  # (By default, size=15 -> 30x30 squares -> 20 columns)
        nb_column = 600 // (size * 2)

        nb_row = math.ceil(dim / nb_column)

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
                self.add_slice(matrix, i, nb_column, self.crop(self.get_slice(image.data, i), x, y, size, size))

            matrices.append(matrix)
        if return_center is True:
            return matrices, centers_mosaic
        else:
            return matrices

    def mosaics_through_time(self):
        """Obtain mosaics for each volume

        :return: list of tuples of numpy.ndarray containing the mosaics of each volumes
        """
        mosaics = list()
        self._image_seg = self._images[0].copy()  # segmentation used for cropping

        for i, img in enumerate(self._4d_images):
            # The absolutepath is changed to None after change_orientation see issue #3304
            img.absolutepath = self._absolute_paths[i]

            im_t_list = (split_img_data(img, dim=3, squeeze_data=True))  # Split along T dimension
            self._images.clear()
            self._images = im_t_list
            matrices, centers_mosaic = self.mosaic(return_center=True)
            mosaics.append(matrices)
        return mosaics, centers_mosaic


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
        """
        Retrieve index along in the R-L direction for each S-I slice in order to center the spinal cord in the
        medial plane, around the labels or segmentation.

        By default, it looks at the latest image in the input list of images, assuming the latest is the labels or
        segmentation.

        If only one label is found, the cord will be centered at that label.

        :return: index: [int] * n_SI
        """
        image = self._images[img_idx].copy()
        assert image.orientation == 'SAL'
        # If mask is empty, raise error
        if np.argwhere(image.data).shape[0] == 0:
            raise ValueError("Label/segmentation image is empty. Can't retrieve RL slice indices.")
        # If mask only has one label (e.g., in sct_detect_pmj), return the R-L index (repeated n_SI times)
        elif np.argwhere(image.data).shape[0] == 1:
            return [np.argwhere(image.data)[0][2]] * image.data.shape[0]  # SAL orientation, so shape[0] -> SI axis
        # Otherwise, find the center of mass of each label (per axial plane) and extrapolate linearly
        else:
            image.change_orientation('RPI')  # need to do that because get_centerline operates in RPI orientation
            # Get coordinate of centerline
            # Here we use smooth=0 because we want the centerline to pass through the labels, and minmax=True extends
            # the centerline below zmin and above zmax to avoid discontinuities
            data_ctl_RPI, _, _, _ = get_centerline(
                image, param=ParamCenterline(algo_fitting='linear', smooth=0, minmax=False))
            data_ctl_RPI.change_orientation('SAL')
            index_RL = np.argwhere(data_ctl_RPI.data)
            return [index_RL[i][2] for i in range(len(index_RL))]

    def get_center(self, img_idx=-1):
        image = self._images[img_idx]
        dim = self.get_dim(image)
        size_y = self.axial_dim(image)
        size_x = self.coronal_dim(image)
        return np.ones(dim) * size_x / 2, np.ones(dim) * size_y / 2

    def mosaic(self):
        """Obtain matrices of the mosaics

        Mosaic images are cropped based on the bounding box of the spinal cord segmentation.

        :return: tuple of numpy.ndarray containing the mosaics of each slice pixels
        """
        # 0. Use the segmentation image to initialize the image cropper
        self._image_seg = self._images[-1]
        cropper = ImageCropper()
        cropper.get_bbox_from_mask(self._image_seg)

        # 1. Compute the sizes of the patches, as well as the overall image
        # 1a. Compute width and height of mosaic squares. (This is assumed to be a square for Axial slices.)
        size_h = cropper.bbox.xmax - cropper.bbox.xmin + 1  # SAL -> SI axis provides height
        size_w = cropper.bbox.ymax - cropper.bbox.ymin + 1  # SAL -> AP axis provides width
        # 1b. Calculate number of columns to display on the report.
        nb_column = 600 // size_w
        # 1c. Calculate number of rows to display.
        nb_slices = cropper.bbox.zmax - cropper.bbox.zmin + 1  # SAL -> LR axis provides nb_slices
        nb_row = math.ceil(nb_slices / nb_column)
        # 1d. Compute the matrix size of the overall mosaic image
        matrix_sz = (int(size_h * nb_row), int(size_w * nb_column))

        # 2. Use the previously-defined segmentation-based cropper to crop the images to the necessary size
        matrices = list()
        for image in self._images:
            image_cropped = cropper.crop(img_in=image)
            matrix = np.zeros(matrix_sz)
            for i in range(nb_slices):
                # Fetch the sagittal slice (which has already been cropped)
                lrslice_cropped = self.get_slice(image_cropped.data, i)
                # Add the cropped slice to the matrix layout
                self.add_slice(matrix, i, nb_column, lrslice_cropped)
            matrices.append(matrix)

        return matrices
