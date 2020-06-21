# coding: utf-8
"""
Interface API for the deepseg module, which performs segmentation using deep learning with the ivadomed package.
"""

# TODO: deal with multi-class segmentation in segment_nifti()

import logging
import os

import ivadomed as imed
import ivadomed.utils
import ivadomed.postprocessing
import nibabel as nib
import numpy as np
import spinalcordtoolbox as sct
import spinalcordtoolbox.deepseg.models

logger = logging.getLogger(__name__)

# Default values if not asked during CLI call and if not present in json metadata.
DEFAULTS = {
    'thr': 0.9,
    'largest': 1,
    'fill_holes': 1,
}


def postprocess(nii_seg, options):
    """
    Wrapper to apply postprocessing on the segmentation, depending on user's, metadata or default options.
    :param nii_seg: nibabel: Segmentation
    :param options: dict: Parameters for postprocessing, including keys such as: threshold, keep_largest_object.
    :return:
    """

    def threshold(nii_seg, thr):
        """Threshold the prediction. For no threshold, set 'thr' to 0."""
        logger.info("Threshold: {}".format(thr))
        if thr:
            nii_seg = imed.postprocessing.threshold_predictions(nii_seg, thr)
        return nii_seg

    def keep_largest_objects(nii_seg, n_objects):
        """Only keep the n largest objects."""
        logger.info("Keep largest objects: {}".format(n_objects))
        if n_objects > 1:
            # TODO: implement the thing below.
            NotImplementedError("For now, the algorithm can only remove the largest object, no more than that.")
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
        logger.info("Fill holes")
        # Make sure input is binary. If not, skip with verbose.
        if np.array_equal(nii_seg.get_fdata(), nii_seg.get_fdata().astype(bool)):
            nii_seg = imed.postprocessing.fill_holes(nii_seg)
        else:
            logger.warning("Algorithm 'fill holes' can only be run on binary segmentation. Skipping.")
        return nii_seg

    logger.info("\nProcessing segmentation\n" + "-" * 23)
    if options['thr']:
        nii_seg = threshold(nii_seg, options['thr'])
    if options['largest']:
        nii_seg = keep_largest_objects(nii_seg, options['largest'])
    if options['fill_holes']:
        nii_seg = fill_holes(nii_seg)
    return nii_seg


def segment_nifti(fname_image, folder_model, fname_prior=None, param=None):
    """
    Segment a nifti file.

    :param fname_image: str: Filename of the image to segment.
    :param folder_model: str: Folder that encloses the deep learning model.
    :param fname_prior: str: Filename of a previous segmentation that is used here as a prior.
    :param param: dict: Dictionary of user's parameter
    :return: fname_out: str: Output filename. If directory does not exist, it will be created.
    """
    if param is None:
        param = {}

    nii_seg = imed.utils.segment_volume(folder_model, fname_image, fname_prior)

    # Postprocessing
    metadata = sct.deepseg.models.get_metadata(folder_model)
    options = {**DEFAULTS, **metadata, **param}
    nii_seg = postprocess(nii_seg, options)

    # Save output seg
    if 'o' in options:
        fname_out = options['o']
    else:
        fname_out = ''.join([sct.utils.splitext(fname_image)[0], '_seg.nii.gz'])
    # If output folder does not exist, create it
    path_out = os.path.dirname(fname_out)
    if not (path_out == '' or os.path.exists(path_out)):
        os.makedirs(path_out)
    nib.save(nii_seg, fname_out)
    return fname_out
