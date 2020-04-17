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
DEFAULTS = {
    'threshold': 0.9,
    'keep_largest_object': True,
    'fill_holes': True
    }


def postprocess(nii_seg, param, metadata):
    """
    Wrapper to apply postprocessing on the segmentation, depending on user's, metadata or default options.
    :param nii_seg: nibabel: Segmentation
    :param param: dict: Defined by user's parameter
    :param metadata: dict: From model's json metadata
    :return:
    """
    def threshold(nii_seg, thr):
        """Threshold the prediction. For no threshold, set 'threshold' to 0."""
        if thr:
            nii_seg = imed.postprocessing.threshold_predictions(nii_seg, thr)
        return nii_seg

    def keep_largest_object(nii_seg):
        """Only keep largest object."""
        # Make sure input is binary. If not, skip with verbose.
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

    def fill_holes(nii_seg):
        """Fill holes"""
        # Make sure input is binary. If not, skip with verbose.
        if np.array_equal(nii_seg.get_fdata(), nii_seg.get_fdata().astype(bool)):
            nii_seg = imed.postprocessing.fill_holes(nii_seg)
        else:
            logger.warning("Algorithm 'fill holes' can only be run on binary segmentation. Skipping.")
        return nii_seg

    options = {**DEFAULTS, **metadata, **param}
    if options['threshold']:
        nii_seg = threshold(nii_seg, options['threshold'])
    if options['keep_largest_object']:
        nii_seg = keep_largest_object(nii_seg)
    if options['fill_holes']:
        nii_seg = fill_holes(nii_seg)
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
    nii_seg = postprocess(nii_seg, param, metadata)

    # TODO: use args to get output name
    fname_out = sct.utils.add_suffix(fname_image, '_seg')
    nib.save(nii_seg, fname_out)
    return fname_out
