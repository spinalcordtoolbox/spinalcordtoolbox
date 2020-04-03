# coding: utf-8
"""
Deals with models for deepseg module. Available models are listed under MODELS.
"""


import os
import logging

from spinalcordtoolbox import __sct_dir__

logger = logging.getLogger(__name__)

MODELS = {
    'cord-t2star':
        {'url': 'https://osf.io/v9hs8/download?version=1',
         'description': 'Cord segmentation on T2*-weighted contrast.'},
    'uqueensland-mice-sc':
        {'url': 'https://osf.io/nu3ma/download?version=1',
         'description': 'Cord segmentation on mouse MRI. Data from University of Queensland.'},
    'uqueensland-mice-gm':
        {'url': 'https://osf.io/mfxwg/download?version=1',
         'description': 'Gray matter segmentation on mouse MRI. Data from University of Queensland.'},
    }


def folder(name):
    """
    Return absolute path of deep learning models.
    :param name: str: Name of model.
    :return:
    """
    return os.path.join(__sct_dir__, 'models', name)


def is_installed(name):
    """
    Check if model is installed under SCT directory.
    :param name: str: Name of model.
    """
    if os.path.exists(os.path.join(folder(name), name + '.pt')) and \
            os.path.exists(os.path.join(folder(name), name + '.json')):
        return True
    else:
        raise FileNotFoundError("The model is not properly installed. Both the .pt and .json files should be "
                                "present, and the basename should be the same as the folder name. Example: "
                                "my_model/my_model.pt, my_model/my_model.json")


def install(name):
    """
    Download and install model under SCT directory.
    :param name: str: Name of model.
    :return:
    """
    raise NotImplementedError
