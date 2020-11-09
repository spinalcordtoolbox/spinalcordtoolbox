#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.deepseg


import os

import pytest
import numpy as np
import nibabel

import spinalcordtoolbox as sct
from spinalcordtoolbox.utils import sct_test_path
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


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('fname_image, fname_seg_manual, fname_out, model', [
    (sct_test_path('t2s', 't2s.nii.gz'),
     sct_test_path('t2s', 't2s_seg-deepseg.nii.gz'),
     't2s_seg_deepseg.nii.gz',
     os.path.join(sct.__deepseg_dir__, 't2star_sc')),
])
def test_segment_nifti(fname_image, fname_seg_manual, fname_out, model,
                       tmp_path):
    """
    Uses the locally-installed sct_testing_data
    """
    fname_out = str(tmp_path/fname_out)  # tmp_path for automatic cleanup
    output = sct.deepseg.core.segment_nifti(fname_image, model,
                                            param={'o': fname_out})
    # TODO: implement integrity test (for now, just checking if output segmentation file exists)
    # Make sure output file is correct
    assert output == fname_out
    # Make sure output file exists
    assert os.path.isfile(output)
    # Compare with ground-truth segmentation
    assert np.all(nibabel.load(output).get_fdata() ==
                  nibabel.load(fname_seg_manual).get_fdata())
