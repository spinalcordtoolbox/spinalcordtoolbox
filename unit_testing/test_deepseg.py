#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.deepseg


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



