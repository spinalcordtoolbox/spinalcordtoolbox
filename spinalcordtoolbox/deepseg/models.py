# coding: utf-8
"""
List of available models.
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


class DeepsegModel:
    """
    Main class that deals with deep learning models. When initialized, the class will inherit the subfields from
    MODELS[modelx] dict.
    """
    def __init__(self, name_model):
        self.name = name_model
        self.folder = os.path.join(__sct_dir__, 'models', name_model)
        for key in MODELS[name_model]:
            setattr(self, key, MODELS[name_model][key])

    def is_installed(self):
        """
        Check if model (.pt file) is installed under SCT directory
        """
        if os.path.exists(os.path.join(self.folder, self.name + '.pt')) and \
                os.path.exists(os.path.join(self.folder, self.name + '.json')):
            return True
        else:
            raise FileNotFoundError("The model is not properly installed. Both the .pt and .json files should be "
                                    "present, and the basename should be the same as the folder name. Example: "
                                    "my_model/my_model.pt, my_model/my_model.json")

    def install(self):
        """
        Download and install model under SCT directory
        :param name_model:
        :return:
        """
        raise NotImplementedError
