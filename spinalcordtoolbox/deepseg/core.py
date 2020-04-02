# coding: utf-8
# Interface API for the deepseg module, which performs segmentation using deep learning with the ivadomed package.


import logging
import nibabel as nib

import ivadomed as imed
import ivadomed.utils

from spinalcordtoolbox.deepseg.models import DeepsegModel
from sct_utils import add_suffix

logger = logging.getLogger(__name__)


class ParamDeepseg:
    """
    Parameters for deepseg module.
    """
    def __init__(self):
        self.output_suffix = '_seg'
        self.remove_temp_files = 1
        self.verbose = 1


def segment_nifti(fname_image, name_model):
    """
    Segment a nifti file.

    :param fname_image: str: Filename of the image to segment.
    :param model_name: str: Name of model to use. See deepseg.model.MODELS
    :return: fname_out: str: Output filename.
    """
    model = DeepsegModel(name_model)
    if not model.is_installed():
        if not model.install():
            logger.error("Model needs to be installed.")
            exit(RuntimeError)

    nii_seg = imed.utils.segment_volume(model.folder, fname_image)

    # TODO: use args to get output name
    fname_out = add_suffix(fname_image, '_seg')
    nib.save(nii_seg, fname_out)
    return fname_out
