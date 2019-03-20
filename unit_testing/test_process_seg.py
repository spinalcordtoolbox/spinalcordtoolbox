#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.process_seg

# TODO: add dummy image with different resolution to check impact of input res
# TODO: add test with known angle (i.e. not found with fitting)
# TODO: test empty slices and slices with two objects
# TODO: figure out why some metrics need float() for assertion

from __future__ import absolute_import
import pytest
import numpy as np
from spinalcordtoolbox import process_seg
from sct_process_segmentation import Param

from create_test_data import dummy_segmentation


# Define global variables
PARAM = Param()
VERBOSE = 0

# Generate a list of fake segmentation for testing: (dummy_segmentation(params), dict of expected results)
im_segs_without_angle = [
    (dummy_segmentation(size_arr=(128, 128, 5), shape='ellipse', angle=0, a=50.0, b=30.0),
     {'area': 47.01, 'angle_AP': 0.0, 'angle_RL': 0.0}),
    ]

im_segs_with_angle = [
    (dummy_segmentation(size_arr=(128, 128, 50), shape='ellipse', angle=15, a=50.0, b=30.0),
     {'area': 47.01, 'angle_AP': 0.0, 'angle_RL': 15.0}),
    ]
#
# # noinspection 801,PyShadowingNames
# def test_compute_csa_noangle(dummy_segmentation):
#     """Test computation of cross-sectional area from input segmentation"""
#     metrics = process_seg.compute_csa(dummy_segmentation(shape='rectangle', angle=0, a=50.0, b=30.0),
#                                       algo_fitting=PARAM.algo_fitting, angle_correction=True, use_phys_coord=False,
#                                       verbose=VERBOSE)
#     assert np.isnan(metrics['csa'].data[95])
#     assert np.mean(metrics['csa'].data[20:80]) == pytest.approx(61.61, rel=0.01)
#     assert np.mean(metrics['angle'].data[20:80]) == pytest.approx(0.0, rel=0.01)
#
#
# # noinspection 801,PyShadowingNames
# def test_compute_csa(dummy_segmentation):
#     """Test computation of cross-sectional area from input segmentation
#     Note: here, compared to the previous tests with no angle, we use smaller hanning window and smaller range for
#     computing the mean, because the smoothing creates spurious errors at edges."""
#     metrics = process_seg.compute_csa(dummy_segmentation(shape='rectangle', angle=15, a=50.0, b=30.0),
#                                       algo_fitting=PARAM.algo_fitting, angle_correction=True, use_phys_coord=False,
#                                       verbose=VERBOSE)
#     assert np.mean(metrics['csa'].data[30:70]) == pytest.approx(61.61, rel=0.01)  # theoretical: 61.61
#     assert np.mean(metrics['angle'].data[30:70]) == pytest.approx(15.00, rel=0.02)
#
#
# # noinspection 801,PyShadowingNames
# def test_compute_csa_ellipse(dummy_segmentation):
#     """Test computation of cross-sectional area from input segmentation"""
#     metrics = process_seg.compute_csa(dummy_segmentation(shape='ellipse', angle=0, a=50.0, b=30.0),
#                                       algo_fitting=PARAM.algo_fitting, angle_correction=True, use_phys_coord=False,
#                                       verbose=VERBOSE)
#     assert np.mean(metrics['csa'].data[30:70]) == pytest.approx(47.01, rel=0.01)
#     assert np.mean(metrics['angle'].data[30:70]) == pytest.approx(0.0, rel=0.01)


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('im_seg,expected', im_segs_without_angle)
def test_compute_shape_without_angle(im_seg, expected):
    metrics = process_seg.compute_shape(im_seg,
                                        algo_fitting=PARAM.algo_fitting,
                                        angle_correction=False,
                                        verbose=VERBOSE)
    assert np.mean(metrics['area'].data) == pytest.approx(expected['area'], rel=0.02)
    assert np.mean(metrics['angle_AP'].data) == pytest.approx(expected['angle_AP'], rel=0.01)
    assert np.mean(metrics['angle_RL'].data) == pytest.approx(expected['angle_RL'], rel=0.01)
    # assert np.mean(metrics['diameter_AP'].data[30:70]) == pytest.approx(6.0, rel=0.05)
    # assert np.mean(metrics['diameter_RL'].data[30:70]) == pytest.approx(10.0, rel=0.05)
    # assert np.mean(metrics['eccentricity'].data[30:70]) == pytest.approx(0.8, rel=0.05)
    # assert np.mean(metrics['orientation'].data[30:70]) == pytest.approx(0.0, rel=0.05)
    # assert np.mean(metrics['solidity'].data[30:70]) == pytest.approx(1.0, rel=0.05)
    # assert np.mean(metrics['angle_AP'].data[30:70]) == pytest.approx(0.0, rel=0.05)
    # assert np.mean(metrics['angle_RL'].data[30:70]) == pytest.approx(0.0, rel=0.05)


# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('im_seg,expected', im_segs_with_angle)
def test_compute_shape_with_angle_estimated(im_seg, expected):
    metrics = process_seg.compute_shape(im_seg,
                                        algo_fitting=PARAM.algo_fitting,
                                        angle_correction=True,
                                        verbose=VERBOSE)
    # Get mid slice to avoid edge effects
    zmid = int(round(len(metrics['area'].data) / 2.0))
    assert float(metrics['area'].data[zmid]) == pytest.approx(expected['area'], rel=0.01)
    assert float(metrics['angle_AP'].data[zmid]) == pytest.approx(expected['angle_AP'], rel=0.01)
    assert float(metrics['angle_RL'].data[zmid]) == pytest.approx(expected['angle_RL'], rel=0.05)
    # assert np.mean(metrics['diameter_AP'].data[30:70]) == pytest.approx(6.0, rel=0.05)
    # assert np.mean(metrics['diameter_RL'].data[30:70]) == pytest.approx(10.0, rel=0.05)
    # assert np.mean(metrics['eccentricity'].data[30:70]) == pytest.approx(0.8, rel=0.05)
    # assert np.mean(metrics['orientation'].data[30:70]) == pytest.approx(0.0, rel=0.05)
    # assert np.mean(metrics['solidity'].data[30:70]) == pytest.approx(1.0, rel=0.05)

#
#
# # noinspection 801,PyShadowingNames
# def test_compute_shape(dummy_segmentation):
#     """Test computation of cross-sectional area from input segmentation."""
#     # Using hanning because faster
#     metrics = process_seg.compute_shape(dummy_segmentation(shape='ellipse', angle=15, a=50.0, b=30.0),
#                                         algo_fitting=PARAM.algo_fitting, verbose=VERBOSE)
#     assert np.mean(metrics['area'].data[30:70]) == pytest.approx(47.01, rel=0.05)
#     assert np.mean(metrics['diameter_AP'].data[30:70]) == pytest.approx(6.0, rel=0.05)
#     assert np.mean(metrics['diameter_RL'].data[30:70]) == pytest.approx(10.0, rel=0.05)
#     assert np.mean(metrics['eccentricity'].data[30:70]) == pytest.approx(0.8, rel=0.05)
#     assert np.mean(metrics['orientation'].data[30:70]) == pytest.approx(0.0, rel=0.05)
#     assert np.mean(metrics['solidity'].data[30:70]) == pytest.approx(1.0, rel=0.05)
