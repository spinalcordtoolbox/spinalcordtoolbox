# coding: utf-8
"""
Deals with models for deepseg module. Available models are listed under MODELS.
Important: model names (onnx or pt files) should have the same name as the enclosing folder.
"""


import os
import json
import logging
import textwrap
import shutil

import spinalcordtoolbox as sct
import spinalcordtoolbox.download
from spinalcordtoolbox.utils import stylize


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
        "contrasts": ["t2star"],
        "default": True,
    },
    "mice_uqueensland_sc": {
        "url": [
            "https://github.com/ivadomed/mice_uqueensland_sc/releases/download/r20200622/r20200622_mice_uqueensland_sc.zip",
            "https://osf.io/nu3ma/download?version=6",
        ],
        "description": "Cord segmentation model on mouse MRI. Data from University of Queensland.",
        "contrasts": ["t1"],
        "default": False,
    },
    "mice_uqueensland_gm": {
        "url": [
            "https://github.com/ivadomed/mice_uqueensland_gm/releases/download/r20200622/r20200622_mice_uqueensland_gm.zip",
            "https://osf.io/mfxwg/download?version=6",
        ],
        "description": "Gray matter segmentation model on mouse MRI. Data from University of Queensland.",
        "contrasts": ["t1"],
        "default": False,
    },
    "t2_tumor": {
        "url": [
            "https://github.com/ivadomed/t2_tumor/archive/r20201215.zip"
        ],
        "description": "Cord tumor segmentation model, trained on T2-weighted contrast.",
        "contrasts": ["t2"],
        "default": False,
    },
    "findcord_tumor": {
        "url": [
            "https://github.com/ivadomed/findcord_tumor/archive/r20201215.zip"
        ],
        "description": "Cord localisation model, trained on T2-weighted images with tumor.",
        "contrasts": ["t2"],
        "default": False,
    },
    "model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel": {
        "url": [
            "https://github.com/ivadomed/model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel/archive/r20201215.zip"
        ],
        "description": "Multiclass cord tumor segmentation model.",
        "contrasts": ["t2", "t1"],
        "default": False,
    },
    "model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg": {
        "url": [
            "https://github.com/ivadomed/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg/archive/r20210401_v2.zip"
        ],
        "description": "Grey/white matter seg on exvivo human T2w.",
        "contrasts": ["t2"],
        "default": False,
    },
    "model_seg_ms_sc_mp2rage": {
        "url": [
            "https://github.com/ivadomed/model_seg_ms_mp2rage/releases/download/r20211223/model_seg_ms_sc_mp2rage.zip"
        ],
        "description": "Segmentation of spinal cord on MP2RAGE data from MS participants.",
        "contrasts": ["mp2rage"],
        "default": False,
    },
    "model_seg_ms_lesion_mp2rage": {
        "url": [
            "https://github.com/ivadomed/model_seg_ms_mp2rage/releases/download/r20211223/model_seg_ms_lesion_mp2rage.zip"
        ],
        "description": "Segmentation of multiple sclerosis lesions on cropped MP2RAGE spinal cord data. To crop the "
                       "data you can first segment the spinal cord using the model 'model_seg_ms_sc_mp2rage' and "
                       "then crop the MP2RAGE image using 'sct_crop_image -i IMAGE -m IMAGE_seg'",
        "contrasts": ["mp2rage"],
        "default": False,
    },
    "model_7t_multiclass_gm_sc_unet2d": {
        "url": [
            "https://github.com/ivadomed/model_seg_gm-wm_t2star_7t_unet3d-multiclass/archive/refs/tags/r20211012.zip"
        ],
        "description": "SC/GM multiclass segmentation on T2*-w contrast at 7T. The model was created by N.J. Laines Medina, V. Callot and A. Le Troter at CRMBM-CEMEREM Aix-Marseille University, France",
        "contrasts": ["t2star"],
        "default": False,
    },
    "model_seg_epfl_t2w_lumbar_sc": {
        "url": [
            "https://github.com/ivadomed/lumbar_seg_EPFL/releases/download/r20220411/model_seg_epfl_t2w_lumbar_sc.zip"
        ],
        "description": "Lumbar SC segmentation on T2w contrast with 3D UNet",
        "contrasts": ["t2"],
        "default": False,
    }
}


# List of task. The convention for task names is: action_(animal)_region_(contrast)
# Regions could be: sc, gm, lesion, tumor
TASKS = {
    'seg_sc_t2star':
        {'description': 'Cord segmentation on T2*-weighted contrast',
         'long_description': 'This segmentation model for T2*w spinal cords uses the UNet architecture, and was '
                             'created with the `ivadomed` package. A subset of a private dataset (sct_testing_large) '
                             'was used, and consists of 236 subjects across 9 different sessions. A total of 388 pairs '
                             'of T2* images were used (anatomical image + manual cord segmentation). The image '
                             'properties include various orientations (superior, inferior) and crops (C1-C3, C4-C7, '
                             'etc.). The dataset was comprised of both non-pathological (healthy) and pathological (MS '
                             'lesion) adult patients.',
         'url': 'https://github.com/ivadomed/t2star_sc',
         'models': ['t2star_sc']},
    'seg_mice_sc':
        {'description': 'Cord segmentation on mouse MRI',
         'long_description': 'This segmentation model for T1w mouse spinal cord segmentation uses the UNet '
                             'architecture, and was created with the `ivadomed` package. Training data was provided by '
                             'Dr. A. Althobathi of the University of Queensland, with the files consisting of T1w '
                             'scans (and manual spinal cord segmentations) from 10 mice in total. The dataset was '
                             'comprised of both non-pathological (healthy) and pathological (diseased) mice.',
         'url': 'https://github.com/ivadomed/mice_uqueensland_sc/',
         'models': ['mice_uqueensland_sc']},
    'seg_mice_gm':
        {'description': 'Gray matter segmentation on mouse MRI',
         'long_description': 'This segmentation model for T1w mouse spinal gray matter segmentation uses the UNet '
                             'architecture, and was created with the `ivadomed` package. Training data was provided by '
                             'Dr. A. Althobathi of the University of Queensland, with the files consisting of T1w '
                             'scans (and manual spinal cord segmentations) from 10 mice in total. The manual '
                             'segmentations were then processed using the `sct_deepseg_gm` to obtain gray matter '
                             'segmentations, which were then used in training. The dataset was comprised of both '
                             'non-pathological (healthy) and pathological (diseased) mice.',
         'url': 'https://github.com/ivadomed/mice_uqueensland_gm/',
         'models': ['mice_uqueensland_gm']},
    'seg_tumor_t2':
        {'description': 'Cord tumor segmentation on T2-weighted contrast',
         'long_description': 'This segmentation model for T2w spinal tumor segmentation uses the UNet '
                             'architecture, and was created with the `ivadomed` package. Training data consisted of '
                             '380 pathological subjects in total: 120 with tumors of type Astrocytoma, 129 with '
                             'Ependymoma, and 131 with Hemangioblastoma. This model is used in tandem with another '
                             'model for specialized cord localisation of spinal cords with tumors '
                             '(https://github.com/ivadomed/findcord_tumor).',
         'url': 'https://github.com/sct-pipeline/tumor-segmentation',
         'models': ['findcord_tumor', 't2_tumor']},
    'seg_ms_sc_mp2rage':
        {'description': 'Cord segmentation on MP2RAGE in MS patients',
         'long_description': 'This segmentation model for MP2RAGE spinal cord segmentation uses a Modified3DUNet '
                             'architecture, and was created with the `ivadomed` package. Training data consisted of '
                             'scans from 30 pathological (MS) patients, and the dataset included manual segmentations '
                             'of MS lesions. This dataset was provided by the University of Basel.',
         'url': 'https://github.com/ivadomed/model_seg_ms_mp2rage',
         'models': ['model_seg_ms_sc_mp2rage']},
    'seg_ms_lesion_mp2rage':
        {'description': 'MS lesion segmentation on cropped MP2RAGE data',
         'long_description': 'This segmentation model for MP2RAGE MS lesion segmentation uses a Modified3DUNet '
                             'architecture, and was created with the `ivadomed` package. Training data consisted of'
                             '30 pathological (MS) patients that included manual segmentations of the spinal cord '
                             'and MS lesions. The dataset was preprocessed to crop around the spinal cord prior to '
                             'training. This dataset was provided by the University of Basel.',
         'url': 'https://github.com/ivadomed/model_seg_ms_mp2rage',
         'models': ['model_seg_ms_lesion_mp2rage']},
    'seg_tumor-edema-cavity_t1-t2':
        {'description': 'Multiclass cord tumor/edema/cavity segmentation',
         'long_description': 'This segmentation model for T1w and T2w spinal tumor, edema, and cavity segmentation '
                             'uses a 3D UNet architecture, and was created with the `ivadomed` package. Training data '
                             'consisted of a subset of the dataset used for the model `seg_tumor_t2`, with 243 '
                             'subjects in total: 49 with tumors of type Astrocytoma, 83 with Ependymoma, and 111 with '
                             'Hemangioblastoma. For each subject, the requisite parts of the affected region (tumor, '
                             'edema, cavity) were segmented individually for training purposes. This model is used in '
                             'tandem with another model for specialized cord localisation of spinal cords with tumors '
                             '(https://github.com/ivadomed/findcord_tumor).',
         'url': 'https://github.com/ivadomed/model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel',
         'models': ['findcord_tumor', 'model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel']},
    'seg_exvivo_gm-wm_t2':
        {'description': 'Grey/white matter seg on exvivo human T2w',
         'long_description': 'This segmentation model for T2w human spinal gray and white matter uses a 2D Unet '
                             'architecture, and was created with the `ivadomed` package. Training data consisted '
                             'of 149 2D slices taken from the scan of an ex vivo spinal cord of a single subject, with '
                             'the gray and white matter from each slice manually segmented for training purposes. The'
                             'data was provided by the University of Queenland, and through this collaboration, the '
                             'model was subsequently applied to the development of an ex vivo spinal cord template '
                             '(https://archive.ismrm.org/2020/1171.html).',
         'url': 'https://github.com/ivadomed/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg',
         'models': ['model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg']},
    'seg_gm_sc_7t_t2star':
        {'description': 'SC/GM seg on T2*-weighted contrast at 7T',
         'long_description': 'This multiclass model (SC/GM) has been developed from 72 subjects acquired at 7T '
                             '(T2*-w images of the cervical spinal cord) by N.J. Laines Medina, V. Callot and '
                             'A. Le Troter in the Center for Magnetic Resonance in Biology and Medicine '
                             '(CRMBM-CEMEREM, UMR 7339, CNRS, Aix-Marseille University, France) including various '
                             'pathologies (HC, MS, ALS). It was validated between 9-folds (CV) single-class and '
                             'multi-class models and was enriched by integrating a hybrid data augmentation (composed '
                             'of classical geometric transformations, MRI artifacts and real GM/WM contrasts distorted '
                             'with anatomically constrained deformation fields) finally tested with an external '
                             'multicentric database. For more information visit: ',
         'url': 'https://github.com/ivadomed/model_seg_gm-wm_t2star_7t_unet3d-multiclass',
         'models': ['model_7t_multiclass_gm_sc_unet2d']},
    'seg_lumbar_sc_t2w':
        {'description': 'Lumbar cord segmentation with 3D UNet',
         'long_description': 'This segmentation model for T2w spinal cord segmentation uses a 3D UNet architecture, '
                             'and was created with the ivadomed package. Training data was provided by Nawal Kinany '
                             'and Dimitry Van De Ville of EPFL, with the files consisting of lumbar T2w scans (and '
                             'manual spinal cord segmentations) of 11 healthy (non-pathological) patients.',
         'url': 'https://github.com/ivadomed/lumbar_seg_EPFL',
         'models': ['model_seg_epfl_t2w_lumbar_sc']}
}


def get_required_contrasts(task):
    """
    Get required contrasts according to models in tasks.

    :return: list: List of required contrasts
    """
    contrasts_required = set()
    for model in TASKS[task]['models']:
        for contrast in MODELS[model]['contrasts']:
            contrasts_required.add(contrast)

    return list(contrasts_required)


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
    color = {True: 'LightGreen', False: 'LightRed'}
    print("{:<30s}{:<50s}{:<15s}MODELS".format("TASK", "DESCRIPTION", "CONTRAST"))
    print("-" * 120)
    for name_task, value in tasks.items():
        path_models = [sct.deepseg.models.folder(name_model) for name_model in value['models']]
        are_models_valid = [sct.deepseg.models.is_valid(path_model) for path_model in path_models]
        task_status = stylize(name_task.ljust(30),
                              color[all(are_models_valid)])
        description_status = stylize(value['description'].ljust(50),
                                     color[all(are_models_valid)])
        models_status = ', '.join([stylize(model_name,
                                           color[is_valid])
                                   for model_name, is_valid in zip(value['models'], are_models_valid)])
        input_contrasts = stylize(str(', '.join(model_name for model_name in
                                                get_required_contrasts(name_task))).ljust(15),
                                  color[all(are_models_valid)])

        print("{}{}{}{}".format(task_status, description_status, input_contrasts, models_status))

    print(
        '\nLegend: {} | {}\n'.format(
            stylize("installed", color[True]),
            stylize("not installed", color[False])))

    print('To read an in-depth description of the training data, model architecture, etc. used for any of these tasks, '
          'type the following command:\n'
          '\n'
          '    {}'.format(stylize('sct_deepseg -task-description TASK_NAME', ['LightBlue', 'Bold'])))
    exit(0)


def display_long_description_task(name_task):
    value = sct.deepseg.models.list_tasks()[name_task]

    indent_len = len("LONG_DESCRIPTION: ")
    print("{}{}".format("TASK:".ljust(indent_len), name_task))
    print('\n'.join(textwrap.wrap(value['long_description'],
                    width=shutil.get_terminal_size()[0],
                    initial_indent="LONG_DESCRIPTION: ",
                    subsequent_indent=' '*indent_len)))
    print("{}{}".format("URL:".ljust(indent_len), value['url']))

    path_models = [sct.deepseg.models.folder(name_model)
                   for name_model in value['models']]
    if all([sct.deepseg.models.is_valid(path_model) for path_model in path_models]):
        installed = stylize("Yes", 'LightGreen')
    else:
        installed = stylize("No", 'LightRed')
    print("{}{}".format("INSTALLED:".ljust(indent_len), installed))

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
