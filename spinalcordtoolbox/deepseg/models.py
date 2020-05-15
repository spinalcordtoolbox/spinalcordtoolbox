# coding: utf-8
"""
Deals with models for deepseg module. Available models are listed under MODELS.
"""


import os
import json
import logging

import spinalcordtoolbox as sct
import spinalcordtoolbox.download


logger = logging.getLogger(__name__)

# List of models. The convention for model names is: (species)_(university)_(contrast)_region
# Regions could be: sc, gm, lesion, tumor
# TODO: add mirror
# TODO: add a test to make sure all fields are present in the dict below
# TODO: deal with "Folder __MACOSX"
MODELS = {
    't2star_sc':
        {'url': 'https://osf.io/v9hs8/download?version=2',
         'description': 'Cord segmentation on T2*-weighted contrast.',
         'default': True},
    'mice_uqueensland_sc':
        {'url': 'https://osf.io/nu3ma/download?version=4',
         'description': 'Cord segmentation on mouse MRI. Data from University of Queensland.',
         'default': False},
    'mice_uqueensland_gm':
        {'url': 'https://osf.io/mfxwg/download?version=4',
         'description': 'Gray matter segmentation on mouse MRI. Data from University of Queensland.',
         'default': False},
    }


def folder(name_model):
    """
    Return absolute path of deep learning models.
    :param name: str: Name of model.
    :return: str: Folder to model
    """
    return os.path.join(sct.__models_dir__, name_model)


def install_model(name_model):
    """
    Download and install specified model under SCT installation dir.
    :param name: str: Name of model.
    :return: None
    """
    logger.info("\nINSTALLING MODEL: {}".format(name_model))
    sct.download.install_data(MODELS[name_model]['url'], os.path.split(folder(name_model))[0])


def install_default_models():
    """
    Download all default models and install them under SCT installation dir.
    :return: None
    """
    for name_model, value in MODELS.items():
        if value['default']:
            install_model(name_model)


def is_installed(name_model):
    """
    Check if model is installed under SCT directory.
    :param name: str: Name of model.
    """
    if os.path.exists(os.path.join(folder(name_model), name_model + '.pt')) and \
            os.path.exists(os.path.join(folder(name_model), name_model + '.json')):
        return True
    else:
        return False


def list_models():
    """
    Display available models with description. Color is used to indicate if model is installed or not. For default
    models, a '*' is added next to the model name.
    :return:
    """
    color = {True: '\033[92m', False: '\033[91m'}
    default = {True: '[*]', False: ''}
    print("{:<25s}DESCRIPTION".format("MODEL"))
    print("-"*80)
    for name_model, value in MODELS.items():
        print("{}{:<25s}{}\033[0m".format(color[is_installed(name_model)], name_model+default[value['default']],
                                          value['description']))
    print('\nLegend: {}installed\033[0m | {}not installed\033[0m | default: [*]\n'.format(color[True], color[False]))


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
