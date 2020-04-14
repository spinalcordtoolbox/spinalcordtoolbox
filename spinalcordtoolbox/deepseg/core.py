# coding: utf-8
"""
Interface API for the deepseg module, which performs segmentation using deep learning with the ivadomed package.
"""


import logging
import nibabel as nib
import ivadomed as imed
import ivadomed.utils
import ivadomed.postprocessing

import spinalcordtoolbox as sct
import spinalcordtoolbox.utils
import spinalcordtoolbox.deepseg.models


logger = logging.getLogger(__name__)

# Default values if not asked during CLI call and if not present in json metadata.
DEFAULT_THRESHOLD = 0.9


class ParamDeepseg:
    """
    Parameters for deepseg module.
    """
    def __init__(self):
        self.threshold = None
        self.output_suffix = '_seg'
        self.remove_temp_files = 1
        self.verbose = 1
        # TODO: add threshold, keep_big_obj


def segment_nifti(fname_image, folder_model, param):
    """
    Segment a nifti file.

    :param fname_image: str: Filename of the image to segment.
    :param folder_model: str: Folder that encloses the deep learning model.
    :param param: class ParamDeepseg: Parameter class ParamDeepseg()
    :return: fname_out: str: Output filename.
    """

    nii_seg = imed.utils.segment_volume(folder_model, fname_image)

    # TODO: postprocessing based on model (info to add in model's json), and if user asked for it (arg)
    metadata = sct.deepseg.models.get_metadata(folder_model)

    # Threshold the prediction. For no prediction, set threshold to -1.
    if param.threshold:
        threshold = param.threshold
    else:
        if 'threshold' in metadata:
            threshold = param.threshold
        else:
            logger.warning("'threshold' is not defined in the model json file. Using threshold of: {}".format(DEFAULT_THRESHOLD))
            threshold = DEFAULT_THRESHOLD
    if not threshold == -1:
        nii_seg = imed.postprocessing.threshold_predictions_nib(nii_seg, threshold)

    # TODO: use args to get output name
    fname_out = sct.utils.add_suffix(fname_image, '_seg')
    nib.save(nii_seg, fname_out)
    return fname_out
