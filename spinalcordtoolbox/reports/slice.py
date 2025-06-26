"""
Image slice extractors for generating QC reports

Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

# TODO: Replace slice by spinalcordtoolbox.image.Slicer

import os
import logging
import math

import numpy as np
from scipy.ndimage import center_of_mass

from spinalcordtoolbox.image import Image, split_img_data
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.cropping import ImageCropper
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline
from spinalcordtoolbox.utils.sys import LazyLoader

nib = LazyLoader("nib", globals(), "nibabel")

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

    def get_aspect(self, image):
        raise NotImplementedError

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
    def inf_nan_fill(A):
        """Interpolate inf and NaN values with neighboring values in a 1D array, in-place.
        If only inf and NaNs, fills the array with zeros.
        """
        valid = np.isfinite(A)
        invalid = ~valid
        if np.all(invalid):
            A.fill(0)
        elif np.any(invalid):
            A[invalid] = np.interp(
                np.nonzero(invalid)[0],
                np.nonzero(valid)[0],
                A[valid])

    def get_slice(self, data, i):
        """Abstract method to obtain a slice of a 3d matrix

        :param data: volume
        :param i: position to slice
        :return: 2D slice
        """
        raise NotImplementedError

    def get_dim(self, image):
        """Abstract method to obtain the depth of the 3d matrix.

        :param image: input Image
        :returns: numpy.ndarray
        """
        raise NotImplementedError

    def mosaic(self):
        """Obtain matrices of the mosaics"""
        raise NotImplementedError

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
        logger.info(f'Resampling image "{os.path.basename(image.absolutepath)}" to {p_resample}x{p_resample} mm')
        dict_interp = {'im': 'spline', 'seg': 'linear'}
        # Create nibabel object
        nii = nib.Nifti1Image(image.data, affine=image.hdr.get_best_affine(), header=image.hdr)
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
            nii_ref = nib.Nifti1Image(image_ref.data, affine=image_ref.hdr.get_best_affine(), header=image_ref.hdr)
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

    def get_aspect(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return py / pz

    def get_slice(self, data, i):
        return data[i, :, :]

    def get_dim(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nx

    def get_center(self, img_idx=-1):
        """Get the center of mass of each slice. For 4D images, segmentation is placed in self.image_seg.
        For 3D images, by default, it assumes that self._images is a list of images, and the
        last item is the segmentation from which the center of mass is computed."""
        if self._image_seg is None:  # For 3D images
            image = self._images[img_idx]
        else:  # For 4D images
            image = self._image_seg
        logger.info('Compute center of mass at each slice')
        data = np.array(image.data)  # we cast np.array to overcome problem if inputing nii format
        nz = image.dim[0]  # SAL orientation
        centers_x = np.zeros(nz)
        centers_y = np.zeros(nz)
        for i in range(nz):
            centers_x[i], centers_y[i] = center_of_mass(data[i, :, :])
        Slice.inf_nan_fill(centers_x)
        Slice.inf_nan_fill(centers_y)
        return centers_x, centers_y

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

    def get_aspect(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return px / py

    def get_slice(self, data, i):
        return data[:, :, int(i)]

    def get_dim(self, image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nz

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
        if image.orientation != 'SAL':
            raise ValueError(f"Image orientation should be SAL, but got: {image.orientation}")
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

    def mosaic(self):
        """Obtain matrices of the mosaics

        Mosaic images are cropped based on the bounding box of the spinal cord segmentation.

        :return: tuple of numpy.ndarray containing the mosaics of each slice pixels
        """
        # 0. Initialize an image cropper to trim the AP and SI axes
        # 0a. Use the segmentation image to initialize the image cropper
        self._image_seg = self._images[-1]
        cropper = ImageCropper()
        cropper.get_bbox_from_mask(self._image_seg)
        # 0b. Modify the overall width/height of the bounding box to show extra context at the extents
        cropper.bbox.xmin = max(cropper.bbox.xmin - 50, 0)
        cropper.bbox.xmax = min(cropper.bbox.xmax + 50, self._image_seg.data.shape[0] - 1)
        cropper.bbox.ymin = max(cropper.bbox.ymin - 15, 0)
        cropper.bbox.ymax = min(cropper.bbox.ymax + 15, self._image_seg.data.shape[1] - 1)

        # 1. Compute the sizes of the patches, as well as the overall image
        # 1a. Compute width and height of mosaic squares. (This is assumed to be a square for Axial slices.)
        size_h = cropper.bbox.xmax - cropper.bbox.xmin + 1  # SAL -> SI axis provides height
        size_w = cropper.bbox.ymax - cropper.bbox.ymin + 1  # SAL -> AP axis provides width
        # 1b. Calculate number of columns to display on the report.
        nb_column = round(600 / size_w)
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

    def single(self):
        """Obtain the matrices of the single slices. Flatten

        :returns: tuple of numpy.ndarray, matrix of the input 3D MRI
                  containing the slices and matrix of the transformed 3D MRI
                  to output containing the slices
        """
        if len(set([x.data.shape for x in self._images])) != 1:
            raise ValueError("Volumes don't have the same size")

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
