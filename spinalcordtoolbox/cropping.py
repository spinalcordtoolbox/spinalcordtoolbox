#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with image cropping


import logging
import numpy as np

from .image import Image, zeros_like


logger = logging.getLogger(__name__)


class BoundingBox(object):
    def __init__(self, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def get_minmax(self, img=None):
        """
        Get voxel-based bounding box from coordinates. Replaces '-1' with max dim along each axis, '-2' with max dim
        minus 1, etc.
        :param img: Image object to get dimensions
        :return:
        """
        def _get_min_value(input):
            if input is None:
                return 0
            else:
                return input

        def _get_max_value(input, dim):
            # If empty, return maximum dimension (i.e. no change)
            if input is None:
                return dim
            # If input is "-1", return maximum dimension (i.e. no change). If input is "-2", returns maximum
            # dimension minus one, etc.
            elif np.sign(input) == -1:
                return input + dim
            # If user specified a non-negative value, use that
            else:
                return input

        xyz_to_num = {'x': 0, 'y': 1, 'z': 2}
        bbox_voxel = BoundingBox()
        for attr, value in self.__dict__.items():
            if attr[-3:] == 'min':
                bbox_voxel.__setattr__(attr, _get_min_value(self.__getattribute__(attr)))
            elif attr[-3:] == 'max':
                bbox_voxel.__setattr__(attr, _get_max_value(self.__getattribute__(attr), img.dim[xyz_to_num[attr[0]]]))
            else:
                raise Exception(ValueError)
        return bbox_voxel


class ImageCropper(object):
    def __init__(self, img_in, mask=None, bbox=BoundingBox(), ref=None):
        """

        :param img_in:
        :param mask:
        :param bbox: BoundingBox object with min and max values for each dimension, used for cropping.
        :param ref:
        """
        self.img_in = img_in
        self.mask = mask
        self.bbox = bbox
        self.ref = ref

    def crop(self, background=None):
        """
        Crop image (change dimension)
        :param background: int: If set, the output image will not be cropped. Instead, voxels outside the bounding
        box will be set to the value specified by this parameter.
        :return Image: img_out
        """
        bbox = self.bbox

        logger.info("Bounding box: x=[{}, {}], y=[{}, {}], z=[{}, {}]"
                    .format(bbox.xmin, bbox.xmax+1, bbox.ymin, bbox.ymax+1, bbox.zmin, bbox.zmax+1))

        # Crop the image
        if background is None:
            logger.info("Cropping the image...")
            data_crop = self.img_in.data[bbox.xmin:bbox.xmax+1, bbox.ymin:bbox.ymax+1, bbox.zmin:bbox.zmax+1]
            img_out = Image(param=data_crop, hdr=self.img_in.hdr)

            # adapt the origin in the sform and qform matrix
            new_origin = np.dot(img_out.hdr.get_qform(), [bbox.xmin, bbox.ymin, bbox.zmin, 1])
            img_out.hdr.structarr['qoffset_x'] = new_origin[0]
            img_out.hdr.structarr['qoffset_y'] = new_origin[1]
            img_out.hdr.structarr['qoffset_z'] = new_origin[2]
            img_out.hdr.structarr['srow_x'][-1] = new_origin[0]
            img_out.hdr.structarr['srow_y'][-1] = new_origin[1]
            img_out.hdr.structarr['srow_z'][-1] = new_origin[2]

        # Set voxels outside the bbox to the value 'background'
        else:
            logger.info("Setting voxels outside the bounding box to: {}".format(background))
            img_out = self.img_in.copy()
            img_out.data[:] = background
            img_out.data[bbox.xmin:bbox.xmax+1, bbox.ymin:bbox.ymax+1, bbox.zmin:bbox.zmax+1] = \
                self.img_in.data[bbox.xmin:bbox.xmax+1, bbox.ymin:bbox.ymax+1, bbox.zmin:bbox.zmax+1]

        return img_out

    def get_bbox_from_minmax(self, bbox=None):
        """
        Get voxel bounding box from xmin, xmax, ymin, ymax, zmin, zmax user input
        """
        self.bbox = bbox.get_minmax(img=self.img_in)

    def get_bbox_from_mask(self, img_mask):
        """
        Get bounding box from input binary mask, by looking at min/max values of the binary object in each dimension.
        """
        data_nonzero = np.nonzero(img_mask.data)
        # find min and max boundaries of the mask
        dim = len(data_nonzero)
        self.bbox.xmin, self.bbox.ymin, self.bbox.zmin = [min(data_nonzero[i]) for i in range(dim)]
        self.bbox.xmax, self.bbox.ymax, self.bbox.zmax = [max(data_nonzero[i]) for i in range(dim)]

    def get_bbox_from_ref(self, img_ref):
        """
        Get bounding box from input reference image, by looking at min/max indices in each dimension.
        img_ref and self.img_in should have the same dimensions.
        """
        from spinalcordtoolbox.resampling import resample_nib
        #  Check that img_ref has the same length as img_in
        if not len(img_ref.data.shape) == len(self.img_in.data.shape):
            logger.error("Inconsistent dimensions: n_dim(img_ref)={}; n_dim(img_in)={}"
                         .format(len(img_ref.data.shape), len(self.img_in.data.shape)))
            raise Exception(ValueError)
        # Fill reference data with ones
        img_ref.data[:] = 1
        # Resample new image (in reference coordinates) into input image
        img_ref_r = resample_nib(img_ref, image_dest=self.img_in, interpolation='nn', mode='constant')
        # img_ref_r.save('test.nii')  # for debug
        # Get bbox from this resampled mask
        self.get_bbox_from_mask(img_ref_r)

    def get_bbox_from_gui(self):
        """
        Launch a GUI. The medial sagittal plane of the image is shown. User selects two points: top-left and bottom-
        right of the cropping window.
        Note: There is no cropping along the right-left direction.
        :return:
        """
        from spinalcordtoolbox.gui import base
        from spinalcordtoolbox.gui.sagittal import launch_sagittal_dialog

        # Change orientation to SAL (for displaying sagittal view in the GUI)
        native_orientation = self.img_in.orientation
        self.img_in.change_orientation('SAL')

        # Launch GUI
        params = base.AnatomicalParams()
        params.vertebraes = [1, 2]  # TODO: Have user draw a sliding rectangle instead (more intuitive)
        params.subtitle = "Click on the top-left (Label 1) and bottom-right (Label 2) of the image to select your " \
                          "cropping window."
        img_labels = zeros_like(self.img_in)
        launch_sagittal_dialog(self.img_in, img_labels, params)

        # Extract coordinates
        img_labels.change_orientation(native_orientation)
        cropping_coord = img_labels.getNonZeroCoordinates(sorting='value')
        # Since there is no cropping along the R-L direction, xmin/xmax are based on image dimension
        self.bbox.xmin, self.bbox.ymin, self.bbox.zmin = (
            0,
            min(cropping_coord[0].y, cropping_coord[1].y),
            min(cropping_coord[0].z, cropping_coord[1].z),
        )
        self.bbox.xmax, self.bbox.ymax, self.bbox.zmax = (
            img_labels.dim[0],
            max(cropping_coord[0].y, cropping_coord[1].y),
            max(cropping_coord[0].z, cropping_coord[1].z),
        )
        # Put back input image in native orientation
        self.img_in.change_orientation(native_orientation)
