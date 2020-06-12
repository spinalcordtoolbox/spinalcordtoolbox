#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.process_seg

# TODO: add test with known angle (i.e. not found with fitting)
# TODO: test empty slices and slices with two objects

from __future__ import absolute_import
import sys
import os
import pytest
import math
import numpy as np

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))
from spinalcordtoolbox import process_seg
from spinalcordtoolbox.centerline.core import ParamCenterline

from spinalcordtoolbox.testing.create_test_data import dummy_segmentation


# Define global variables
VERBOSE = 0  # set to 2 to save files
DEBUG = False  # Set to True to save images


dict_test_orientation = [
    {'input': 0.0, 'expected': 0.0},
    {'input': math.pi, 'expected': 0.0},
    {'input': -math.pi, 'expected': 0.0},
    {'input': math.pi / 2, 'expected': 90.0},
    {'input': -math.pi / 2, 'expected': 90.0},
    {'input': 2 * math.pi, 'expected': 0.0},
    {'input': math.pi / 4, 'expected': 45.0},
    {'input': -math.pi / 4, 'expected': 45.0},
    {'input': 3 * math.pi / 4, 'expected': 45.0},
    {'input': -3 * math.pi / 4, 'expected': 45.0},
    {'input': math.pi / 8, 'expected': 22.5},
    {'input': -math.pi / 8, 'expected': 22.5},
    {'input': 3 * math.pi / 8, 'expected': 67.5},
    {'input': -3 * math.pi / 8, 'expected': 67.5},
    ]


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('test_orient', dict_test_orientation)
def test_fix_orientation(test_orient):
    assert process_seg.fix_orientation(test_orient['input']) == pytest.approx(test_orient['expected'], rel=0.0001)


# Generate a list of fake segmentation for testing: (dummy_segmentation(params), dict of expected results)
im_segs = [
    # test area
    (dummy_segmentation(size_arr=(32, 32, 5), debug=DEBUG),
     {'area': 77, 'angle_RL': 0.0, 'angle_AP': 0.0, 'length': 5.0},
     {'angle_corr': False}),
    # test anisotropic pixel dim
    (dummy_segmentation(size_arr=(64, 32, 5), pixdim=(0.5, 1, 5), debug=DEBUG),
     {'area': 77, 'angle_RL': 0.0, 'angle_AP': 0.0},
     {'angle_corr': False}),
    # test with angle IS
    (dummy_segmentation(size_arr=(32, 32, 5), pixdim=(1, 1, 5), angle_IS=15, debug=DEBUG),
     {'area': 77, 'angle_RL': 0.0, 'angle_AP': 0.0},
     {'angle_corr': False}),
    # test with ellipse shape
    (dummy_segmentation(size_arr=(64, 64, 5), shape='ellipse', radius_RL=13.0, radius_AP=5.0, angle_RL=0.0,
                        debug=DEBUG),
     {'area': 197.0, 'diameter_AP': 10.0, 'diameter_RL': 26.0, 'angle_RL': 0.0, 'angle_AP': 0.0},
     {'angle_corr': False}),
    # test with int16. Different bit ordering, which can cause issue when applying transform.warp()
    (dummy_segmentation(size_arr=(64, 320, 5), pixdim=(1, 1, 1), dtype=np.int16, orientation='RPI',
                        shape='rectangle', radius_RL=13.0, radius_AP=5.0, angle_RL=0.0, debug=DEBUG),
     {'area': 297.0, 'angle_RL': 0.0, 'angle_AP': 0.0},
     {'angle_corr': False}),
    # test with angled spinal cord (neg angle)
    (dummy_segmentation(size_arr=(64, 64, 20), shape='ellipse', radius_RL=13.0, radius_AP=5.0, angle_RL=-30.0,
                        debug=DEBUG),
     {'area': 197.0, 'diameter_AP': 10.0, 'diameter_RL': 26.0, 'angle_RL': -30.0, 'angle_AP': 0.0, 'length': 23.15},
     {'angle_corr': True}),
    # test with AP angled spinal cord
    (dummy_segmentation(size_arr=(64, 64, 20), shape='ellipse', radius_RL=13.0, radius_AP=5.0, angle_AP=20.0,
                        debug=DEBUG),
     {'area': 197.0, 'diameter_AP': 10.0, 'diameter_RL': 26.0, 'angle_RL': 0.0, 'angle_AP': 20.0, 'length': 21.02},
     {'angle_corr': True}),
    # test with RL and AP angled spinal cord
    (dummy_segmentation(size_arr=(64, 64, 50), shape='ellipse', radius_RL=13.0, radius_AP=5.0,
                        angle_RL=-10.0, angle_AP=15.0, debug=DEBUG),
     {'area': 197.0, 'diameter_AP': 10.0, 'diameter_RL': 26.0, 'angle_RL': -10.0, 'angle_AP': 15.0},
     {'angle_corr': True}),
    # Reproduce issue: "LinAlgError: SVD did not converge". Note: due to the cropping, the estimated angle_RL is wrong,
    # so it had to be made wrong in the expected values
    (dummy_segmentation(size_arr=(64, 64, 50), shape='ellipse', radius_RL=13.0, radius_AP=5.0,
                        angle_RL=-10.0, angle_AP=30.0, debug=DEBUG),
     {'area': 197.0, 'diameter_AP': 10.0, 'diameter_RL': 26.0, 'angle_RL': -11.5, 'angle_AP': 30.0},
     {'angle_corr': True}),
    # test uint8 input
    (dummy_segmentation(size_arr=(32, 32, 50), dtype=np.uint8, angle_RL=15, debug=DEBUG),
     {'area': 77, 'angle_RL': 15.0, 'angle_AP': 0.0},
     {'angle_corr': True}),
    # test all output params
    (dummy_segmentation(size_arr=(128, 128, 5), pixdim=(1, 1, 1), shape='ellipse', radius_RL=50.0, radius_AP=30.0,
                        debug=DEBUG),
     {'area': 4701, 'angle_AP': 0.0, 'angle_RL': 0.0, 'diameter_AP': 60.0, 'diameter_RL': 100.0, 'eccentricity': 0.8,
      'orientation': 0.0},
     {'angle_corr': False}),
    # test with one empty slice
    (dummy_segmentation(size_arr=(32, 32, 5), zeroslice=[2], debug=DEBUG),
     {'area': np.nan},
     {'angle_corr': False, 'slice': 2})
    ]

# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('im_seg,expected,params', im_segs)
def test_compute_shape(im_seg, expected, params):
    metrics, fit_results = process_seg.compute_shape(im_seg,
                                                     angle_correction=params['angle_corr'],
                                                     param_centerline=ParamCenterline(),
                                                     verbose=VERBOSE)
    for key in expected.keys():
        # fetch obtained_value
        if 'slice' in params:
            obtained_value = float(metrics['area'].data[params['slice']])
        else:
            if key == 'length':
                # when computing length, sums values across slices
                obtained_value = metrics[key].data.sum()
            else:
                # otherwise, average across slices
                obtained_value = metrics[key].data.mean()
        # fetch expected_value
        if expected[key] is np.nan:
            assert math.isnan(obtained_value)
            break
        else:
            expected_value = pytest.approx(expected[key], rel=0.05)
        assert obtained_value == expected_value
