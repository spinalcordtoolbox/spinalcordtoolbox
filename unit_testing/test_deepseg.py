#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.deepseg


import os
import pytest
import numpy as np

import nibabel

import spinalcordtoolbox as sct
import spinalcordtoolbox.deepseg.core
import spinalcordtoolbox.deepseg.models


param_deepseg = [
    ({'fname_image': 'sct_testing_data/t2s/t2s.nii.gz',
      'model': os.path.join(sct.__deepseg_dir__, 't2star_sc'),
      'fname_seg_manual': 'sct_testing_data/t2s/t2s_seg-deepseg.nii.gz'}),
]


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


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('params', param_deepseg)
def test_segment_nifti(params):
    """
    Uses the locally-installed sct_testing_data
    """
    fname_out = 't2s_seg_deepseg.nii.gz'
    output = sct.deepseg.core.segment_nifti(
        params['fname_image'], params['model'], param={'o': fname_out})
    # TODO: implement integrity test (for now, just checking if output segmentation file exists)
    # Make sure output file is correct
    assert output == fname_out
    # Make sure output file exists
    assert os.path.isfile(output)
    # Compare with ground-truth segmentation
    assert np.all(nibabel.load(output).get_fdata() == nibabel.load(params['fname_seg_manual']).get_fdata())
