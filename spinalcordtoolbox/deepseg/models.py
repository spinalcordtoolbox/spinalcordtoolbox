# coding: utf-8
"""
Deals with models for deepseg module. Available models are listed under MODELS.
"""


import os
import json
import logging
import colored

import spinalcordtoolbox as sct
import spinalcordtoolbox.download


logger = logging.getLogger(__name__)

# List of models. The convention for model names is: (species)_(university)_(contrast)_region
# Regions could be: sc, gm, lesion, tumor
MODELS = {
    "t2star_sc": {
        "url": [
            "https://github.com/ivadomed/t2star_sc/releases/download/r20200622/r20200622_t2star_sc.zip",
            "https://osf.io/v9hs8/download?version=5",
        ],
        "description": "Cord segmentation model on T2*-weighted contrast.",
        "default": True,
    },
    "mice_uqueensland_sc": {
        "url": [
            "https://github.com/ivadomed/mice_uqueensland_sc/releases/download/r20200622/r20200622_mice_uqueensland_sc.zip",
            "https://osf.io/nu3ma/download?version=6",
        ],
        "description": "Cord segmentation model on mouse MRI. Data from University of Queensland.",
        "default": False,
    },
    "mice_uqueensland_gm": {
        "url": [
            "https://github.com/ivadomed/mice_uqueensland_gm/releases/download/r20200622/r20200622_mice_uqueensland_gm.zip",
            "https://osf.io/mfxwg/download?version=6",
        ],
        "description": "Gray matter segmentation model on mouse MRI. Data from University of Queensland.",
        "default": False,
    },
    "t2_tumor": {
        "url": [
            "https://github.com/ivadomed/t2_tumor/releases/download/r20200621/r20200621_t2_tumor.zip",
            "https://osf.io/uwe7k/download?version=2",
        ],
        "description": "Cord tumor segmentation model, trained on T2-weighted contrast.",
        "default": False,
    },
    "findcord_tumor": {
        "url": [
            "https://github.com/ivadomed/findcord_tumor/releases/download/r20200621/r20200621_findcord_tumor.zip",
            "https://osf.io/qj6d5/download?version=1",
        ],
        "description": "Cord localisation model, trained on T2-weighted images with tumor.",
        "default": False,
    },
    "model_find_disc_t2": {
        "url": ["https://github.com/ivadomed/model_find_disc_t2/archive/r20200928.zip"],
        "description": "intervertebral disc localisation model, trained on T2-weighted images",
        "default": True,
    },
    "model_find_disc_t1": {
        "url": ["https://github.com/ivadomed/model_find_disc_t1/archive/r20201013.zip"],
        "description": "intervertebral disc localisation model, trained on T1-weighted images",
        "default": True,
    }
}

# List of task. The convention for task names is: action_(animal)_region_(contrast)
# Regions could be: sc, gm, lesion, tumor
TASKS = {
    'seg_sc_t2star':
        {'description': 'Cord segmentation on T2*-weighted contrast.',
         'models': ['t2star_sc']},
    'seg_mice_sc':
        {'description': 'Cord segmentation on mouse MRI.',
         'models': ['mice_uqueensland_sc']},
    'seg_mice_gm':
        {'description': 'Gray matter segmentation on mouse MRI.',
         'models': ['mice_uqueensland_gm']},
    'seg_tumor_t2':
        {'description': 'Cord tumor segmentation on T2-weighted contrast.',
         'models': ['findcord_tumor', 't2_tumor']},
    'find_disc_t2':
        {'description': 'locate posterior poit of each disc on T2 straighten image',
         'models': ["model_find_disc_t2"]},
    'find_disc_t1':
        {'description': 'locate posterior poit of each disc on T1 straighten image',
         'models': ["model_find_disc_t1"]}

}


def folder(name_model):
    """
    Return absolute path of deep learning models.

    :param name: str: Name of model.
    :return: str: Folder to model
    """
    return os.path.join(sct.__deepseg_dir__, name_model)


def install_model(name_model):
    """
    Download and install specified model under SCT installation dir.

    :param name: str: Name of model.
    :return: None
    """
    logger.info("\nINSTALLING MODEL: {}".format(name_model))
    sct.download.install_data(MODELS[name_model]['url'], folder(name_model))


def install_default_models():
    """
    Download all default models and install them under SCT installation dir.

    :return: None
    """
    for name_model, value in MODELS.items():
        if value['default']:
            install_model(name_model)


def is_valid(path_model):
    """
    Check if model has the necessary files and follow naming conventions:
    - Folder should have the same name as the enclosed files.

    :param path_model: str: Absolute path to folder that encloses the model files.
    """
    name_model = path_model.rstrip(os.sep).split(os.sep)[-1]
    return (os.path.exists(os.path.join(path_model, name_model + '.pt')) or
            os.path.exists(os.path.join(path_model, name_model + '.onnx'))) and os.path.exists(
        os.path.join(path_model, name_model + '.json'))


def list_tasks():
    """
    Display available tasks with description.
    :return: dict: Tasks that are installed
    """
    return {name: value for name, value in TASKS.items()}


def display_list_tasks():
    tasks = sct.deepseg.models.list_tasks()
    # Display beautiful output
    color = {True: 'green', False: 'red'}
    print("{:<20s}{:<50s}MODELS".format("TASK", "DESCRIPTION"))
    print("-" * 80)
    for name_task, value in tasks.items():
        path_models = [sct.deepseg.models.folder(name_model) for name_model in value['models']]
        are_models_valid = [sct.deepseg.models.is_valid(path_model) for path_model in path_models]
        task_status = colored.stylize(name_task.ljust(20),
                                      colored.fg(color[all(are_models_valid)]))
        description_status = colored.stylize(value['description'].ljust(50),
                                             colored.fg(color[all(are_models_valid)]))
        models_status = ', '.join([colored.stylize(model_name,
                                                   colored.fg(color[is_valid]))
                                   for model_name, is_valid in zip(value['models'], are_models_valid)])
        print("{}{}{}".format(task_status, description_status, models_status))

    print(
        '\nLegend: {} | {}\n'.format(
            colored.stylize("installed", colored.fg(color[True])),
            colored.stylize("not installed", colored.fg(color[False]))))
    exit(0)


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
