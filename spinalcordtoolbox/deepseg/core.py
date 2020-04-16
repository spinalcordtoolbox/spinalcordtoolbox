# coding: utf-8
"""
Interface API for the deepseg module, which performs segmentation using deep learning with the ivadomed package.
"""


import logging
import numpy as np
import nibabel as nib
import ivadomed as imed
import ivadomed.utils
import ivadomed.postprocessing

import spinalcordtoolbox as sct
import spinalcordtoolbox.deepseg.models


logger = logging.getLogger(__name__)

# Default values if not asked during CLI call and if not present in json metadata.
DEFAULT_THRESHOLD = 0.9
DEFAULT_KEEP_LARGEST_OBJECT = True
DEFAULT_FILL_HOLES = True


class ParamDeepseg:
    """
    Parameters for deepseg module.
    """
    def __init__(self):
        self.threshold = None
        self.keep_largest_object = None
        self.fill_holes = None
        self.output_suffix = '_seg'
        self.remove_temp_files = 1
        self.verbose = 1
        # TODO: add threshold, keep_big_obj


class PostProcessing:
    """
    Deals with post-processing of the segmentation. Consider param (i.e. user's flag) with more priority than
    metadata (i.e. from model's json file).
    """
    def __init__(self, param, metadata):
        """
        :param param: class ParamDeepseg: Defined by user's parameter
        :param metadata: dict: From model's json metadata
        """
        self.param = param
        self.metadata = metadata

    def threshold(self, nii_seg):
        """
        Threshold the prediction. For no prediction, set 'threshold' to 0.
        """
        if self.param.threshold:
            thr = self.param.threshold
        else:
            if 'threshold' in self.metadata:
                thr = self.metadata.threshold
            else:
                logger.warning("'threshold' is not defined in the model json file. Using threshold of: {}".format(
                    DEFAULT_THRESHOLD))
                thr = DEFAULT_THRESHOLD
        if thr:
            nii_seg = imed.postprocessing.threshold_predictions(nii_seg, thr)
        return nii_seg

    def keep_largest_object(self, nii_seg):
        """
        Only keep largest object
        """
        # TODO: This if/elif below is ugly. Cannot think of something better for now...
        do_process = DEFAULT_KEEP_LARGEST_OBJECT
        if self.param.keep_largest_object is True:
            do_process = True
        elif self.param.keep_largest_object is None:
            if 'keep_largest_object' in self.metadata:
                do_process = self.metadata.keep_largest_object
        if do_process:
            # Make sure input is binary
            if np.array_equal(nii_seg.get_fdata(), nii_seg.get_fdata().astype(bool)):
                # Fetch axis corresponding to superior-inferior direction
                # TODO: move that code in image
                affine = nii_seg.get_header().get_best_affine()
                code = nib.orientations.aff2axcodes(affine)
                if 'I' in code:
                    axis_infsup = code.index('I')
                elif 'S' in code:
                    axis_infsup = code.index('S')
                else:
                    raise ValueError(
                        "Neither I nor S is present in code: {}, for affine matrix: {}".format(code, affine))
                nii_seg = imed.postprocessing.keep_largest_object_per_slice(nii_seg, axis=axis_infsup)
            else:
                logger.warning("Algorithm 'keep largest object' can only be run on binary segmentation. Skipping.")
        return nii_seg

    def fill_holes(self, nii_seg):
        """
        Fill holes
        """
        # TODO: This if/elif below is ugly. Cannot think of something better for now...
        do_process = DEFAULT_FILL_HOLES
        if self.param.fill_holes is True:
            do_process = True
        elif self.param.fill_holes is None:
            if 'fill_holes' in self.metadata:
                do_process = self.metadata.fill_holes
        if do_process:
            if np.array_equal(nii_seg.get_fdata(), nii_seg.get_fdata().astype(bool)):
                nii_seg = imed.postprocessing.fill_holes(nii_seg)
            else:
                logger.warning("Algorithm 'fill holes' can only be run on binary segmentation. Skipping.")
            return nii_seg


def segment_nifti(fname_image, folder_model, param):
    """
    Segment a nifti file.

    :param fname_image: str: Filename of the image to segment.
    :param folder_model: str: Folder that encloses the deep learning model.
    :param param: class ParamDeepseg: Parameter class ParamDeepseg()
    :return: fname_out: str: Output filename.
    """

    nii_seg = imed.utils.segment_volume(folder_model, fname_image)

    # Postprocessing
    metadata = sct.deepseg.models.get_metadata(folder_model)
    # TODO: is this PostProcessing class an overkill? Given that this class will only be used here... maybe a better
    #  alternative would be to create a file deepseg/postprocessing and move all postprocessing functions there,
    #  instead of having them inside a class.
    postproc = PostProcessing(param, metadata)
    nii_seg = postproc.threshold(nii_seg)
    nii_seg = postproc.keep_largest_object(nii_seg)
    nii_seg = postproc.fill_holes(nii_seg)

    # TODO: use args to get output name
    fname_out = sct.utils.add_suffix(fname_image, '_seg')
    nib.save(nii_seg, fname_out)
    return fname_out
