#!/usr/bin/env python
# Deals with Brain Imaging Data Structure (BIDS)

import os
import logging

import spinalcordtoolbox as sct

logger = logging.getLogger(__name__)


def get_json_file_name(fname, check_exist=False):
    """
    Get json file name by replacing '.nii' or '.nii.gz' extension by '.json'.
    Check if input file follows NIFTI extension rules.
    Optional: check if json file exists.
    :param fname: str: Input NIFTI file name.
    check_exist: Bool: Check if json file exists.
    :return: fname_json
    """
    list_ext = ['.nii', '.nii.gz']
    basename, ext = sct.utils.splitext(fname)
    if ext not in list_ext:
        raise ValueError("Problem with file: {}. Extension should be one of {}".format(fname, list_ext))
    fname_json = basename + '.json'

    if check_exist:
        if not os.path.isfile(fname_json):
            FileNotFoundError()

    return fname_json
