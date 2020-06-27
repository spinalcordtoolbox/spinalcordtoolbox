#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.deepseg


import os

import spinalcordtoolbox as sct
import spinalcordtoolbox.deepseg.core
import spinalcordtoolbox.deepseg.models


def test_install_model():
    """
    Download all models, to allow subsequent tests on the model packages.
    :return:
    """
    for name_model, value in sct.deepseg.models.MODELS.items():
        if value['default']:
            sct.deepseg.models.install_model(name_model)
            # Make sure all files are present after unpacking the model
            assert sct.deepseg.models.is_valid(sct.deepseg.models.folder(name_model))


def test_model_dict():
    """
    Make sure all fields are present in each model.
    :return:
    """
    for key, value in sct.deepseg.models.MODELS.items():
        assert('url' in value)
        assert('description' in value)
        assert('default' in value)


def test_segment_nifti():
    """
    Uses the locally-installed sct_testing_data
    :return:
    """
    output = sct.deepseg.core.segment_nifti(
        'sct_testing_data/t2s/t2s.nii.gz',
        os.path.join(sct.__deepseg_dir__, 't2star_sc'),
        {'o': 't2s_seg_deepseg.nii.gz'})
    # TODO: implement integrity test (for now, just checking if output segmentation file exists)
    assert output == 't2s_seg_deepseg.nii.gz'
    assert os.path.isfile(output)
