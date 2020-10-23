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
    'remove_small': None
}


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

    options = {**DEFAULTS, **vars(param)}
    nii_seg = imed.utils.segment_volume(folder_model, fname_image, fname_prior, options=options)

    # Save output seg
    if 'o' in options:
        fname_out = options['o']
    else:
        fname_out = ''.join([sct.image.splitext(fname_image)[0], '_seg.nii.gz'])
    # If output folder does not exist, create it
    path_out = os.path.dirname(fname_out)
    if not (path_out == '' or os.path.exists(path_out)):
        os.makedirs(path_out)
    nib.save(nii_seg, fname_out)
    return fname_out
