#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.process_seg

# TODO: add test with known angle (i.e. not found with fitting)
# TODO: test empty slices and slices with two objects

from __future__ import absolute_import
import pytest
import math
import numpy as np
from spinalcordtoolbox import process_seg
from sct_process_segmentation import Param

from create_test_data import dummy_segmentation


# Define global variables
PARAM = Param()
VERBOSE = 0

# Generate a list of fake segmentation for testing: (dummy_segmentation(params), dict of expected results)
im_segs = [
    (dummy_segmentation(size_arr=(32, 32, 5)), {'area': 77, 'angle_RL': 0.0}, {'angle_corr': False}),
    (dummy_segmentation(size_arr=(64, 32, 5), pixdim=(0.5, 1, 5)), {'area': 77, 'angle_RL': 0.0},
     {'angle_corr': False}),
    (dummy_segmentation(size_arr=(32, 32, 5), pixdim=(1, 1, 5), angle_IS=15), {'area': 77, 'angle_RL': 0.0},
     {'angle_corr': False}),
    (dummy_segmentation(size_arr=(32, 32, 50), pixdim=(1, 1, 1), angle_RL=15), {'area': 77, 'angle_RL': 15.0},
     {'angle_corr': True}),
    (dummy_segmentation(size_arr=(128, 128, 5), pixdim=(1, 1, 1), shape='ellipse', radius_RL=50.0, radius_AP=30.0),
     {'area': 4701, 'angle_AP': 0.0, 'angle_RL': 0.0, 'diameter_AP': 60.0, 'diameter_RL': 100.0, 'eccentricity': 0.8,
      'orientation': 0.0, 'solidity': 1.0}, {'angle_corr': False}),
    (dummy_segmentation(size_arr=(32, 32, 5), zeroslice=[2]),
     {'area': np.nan}, {'angle_corr': False, 'slice': 2})
    ]


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('im_seg,expected,params', im_segs)
def test_compute_shape(im_seg, expected, params):
    metrics = process_seg.compute_shape(im_seg,
                                        algo_fitting=PARAM.algo_fitting,
                                        angle_correction=params['angle_corr'],
                                        verbose=VERBOSE)
    for key in expected.keys():
        # fetch obtained_value
        if 'slice' in params:
            obtained_value = float(metrics['area'].data[params['slice']])
        else:
            obtained_value = float(np.mean(metrics[key].data))
        # fetch expected_value
        if expected[key] is np.nan:
            assert math.isnan(obtained_value)
            break
        else:
            expected_value = pytest.approx(expected[key], rel=0.02)
        assert obtained_value == expected_value
