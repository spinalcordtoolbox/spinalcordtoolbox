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
        self.keep_largest_object = True
        self.output_suffix = '_seg'
        self.remove_temp_files = 1
        self.verbose = 1
        # TODO: add threshold, keep_big_obj


class PostProcessing:
    """
    Deals with post-processing of the segmentation.
    """
    def __init__(self, param, metadata):
        """
        :param param:
        :param metadata:
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
                thr = self.param.threshold
            else:
                logger.warning("'threshold' is not defined in the model json file. Using threshold of: {}".format(
                    DEFAULT_THRESHOLD))
                thr = DEFAULT_THRESHOLD
        if thr:
            nii_seg = imed.postprocessing.threshold_predictions_nib(nii_seg, thr)
        return nii_seg

    def keep_largest_object(self, nii_seg):
        """
        Only keep largest object
        """
        if self.param.keep_largest_object:
            if self.threshold:
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
                nii_seg = imed.postprocessing.keep_largest_object_per_slice_nib(nii_seg, axis=axis_infsup)
            else:
                logger.warning("Algorithm 'keep largest object' can only be run on binary segmentation.")
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
    postproc = PostProcessing(param, metadata)
    nii_seg = postproc.threshold(nii_seg)
    nii_seg = postproc.keep_largest_object(nii_seg)

    # TODO: use args to get output name
    fname_out = sct.utils.add_suffix(fname_image, '_seg')
    nib.save(nii_seg, fname_out)
    return fname_out
