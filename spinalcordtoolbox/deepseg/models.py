# coding: utf-8
"""
Deals with models for deepseg module. Available models are listed under MODELS.
"""


import os
import json
import logging

from spinalcordtoolbox import __sct_dir__

logger = logging.getLogger(__name__)

# List of models. The convention for model names is: (species)_(university)_(contrast)_region
# Regions could be: sc, gm, lesion, tumor
MODELS = {
    't2star_sc':
        {'url': 'https://osf.io/v9hs8/download?version=2',
         'description': 'Cord segmentation on T2*-weighted contrast.'},
    'uqueensland_mice_sc':
        {'url': 'https://osf.io/nu3ma/download?version=2',
         'description': 'Cord segmentation on mouse MRI. Data from University of Queensland.'},
    'uqueensland_mice_gm':
        {'url': 'https://osf.io/mfxwg/download?version=2',
         'description': 'Gray matter segmentation on mouse MRI. Data from University of Queensland.'},
    }


def folder(name):
    """
    Return absolute path of deep learning models.
    :param name: str: Name of model.
    :return:
    """
    return os.path.join(__sct_dir__, 'models', name)


def is_model(name):
    """
    Check if model is listed in MODELS
    :param name: str: Name of model.
    :return:
    """
    if name not in list(MODELS.keys()):
        raise ValueError("The selected model does not exist: {}".format(name))
    return True


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


def get_metadata(folder_model):
    """
    Get metadata from json file located in folder_model
    :param path_model: str: Model folder
    :return: dict
    """
    fname_metadata = os.path.join(folder_model, os.path.basename(folder_model) + '.json')
    with open(fname_metadata, "r") as fhandle:
        metadata = json.load(fhandle)
    return metadata
