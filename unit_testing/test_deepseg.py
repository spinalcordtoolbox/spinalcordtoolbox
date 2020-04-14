#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.deepseg


import os

import spinalcordtoolbox as sct
import spinalcordtoolbox.deepseg.core
import spinalcordtoolbox.deepseg.models


def test_model_dict():
    """
    Make sure all fields are present in each model.
    :return:
    """
    for key, value in sct.deepseg.models.MODELS.items():
        assert('url' in value)
        assert ('description' in value)
        assert ('default' in value)


def test_segment_nifti():
    """
    Uses the locally-installed sct_testing_data
    :return:
    """
    sct.deepseg.core.segment_nifti(
        'sct_testing_data/t2s/t2s.nii.gz',
        os.path.join(sct.__models_dir__, 't2star_sc'),
        sct.deepseg.core.ParamDeepseg())
    # TODO: implement integrity test (for now, just checking if output segmentation file exists)
    assert os.path.isfile('sct_testing_data/t2s/t2s_seg.nii.gz')
