# coding: utf-8
# Interface API for the deepseg module, which performs segmentation using deep learning with the ivadomed package.


import nibabel as nib

from ivadomed.utils import segment_volume

from sct_utils import add_suffix


# TODO Add class ParamDeepseg

def segment_nifti(fname_image, folder_model):
    """
    Segment a nifti file.

    :param fname_image: str: Filename of the image to segment.
    :param param_deepseg: class ParamDeepseg: Segmentation parameters.
    :return: fname_out: str: Output filename.
    """
    nii_seg = segment_volume(folder_model, fname_image)

    # TODO: use args to get output name
    fname_out = add_suffix(fname_image, '_seg')
    nib.save(nii_seg, fname_out)
    return fname_out
