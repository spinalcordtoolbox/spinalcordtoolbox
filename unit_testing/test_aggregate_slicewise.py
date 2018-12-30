#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.aggregate_slicewise

# TODO: add tests: with perslice/perlevel, if slices and/or levels is empty

from __future__ import absolute_import

import pytest

import numpy as np
import nibabel as nib

from spinalcordtoolbox import aggregate_slicewise
from spinalcordtoolbox.process_seg import Metric
from spinalcordtoolbox.image import Image


@pytest.fixture(scope="session")
def dummy_metric():
    """Create a dummy metric dictionary."""
    metrics = {'metric1': Metric(z=[3, 4, 5, 6, 7], value=[29., 31., 39., 41., 50.], label='Metric with float [a.u.]'),
               'metric2': Metric(z=[3, 4, 5, 6, 7], value=[99, 100, 101, 102, 103], label='Metric with int [a.u.]'),
               'metric3': Metric(z=[3, 4, 5, 6, 7], value=[99, np.nan, 101, 102, 103], label='Metric with nan'),
               'metric4': Metric(z=[3, 4, 5, 6, 7], value=[99, 100, 101, 102], label='Inconsistent value and z length.'),
               'metric5': Metric(z=[3, 4, 5, 6, 7], value=[99, "boo!", 101, 102, 103], label='Metric with string')}
    return metrics


@pytest.fixture(scope="session")
def dummy_vert_level():
    """
    Create a dummy Image representing vertebral labeling.
    Note: The z-size of this image can to be equal or larger than the metric's length, however the indexation needs
    to match the metric['z'] field.
    Example: data[4, 4, 5] = 2 means that at z=5, the vertebral level is C2.
    """
    nx, ny, nz = 9, 9, 9  # image dimension
    data = np.zeros((nx, ny, nz))
    # define vertebral level for each slice as a pixel at the center of the image
    data[4, 4, :] = [1, 1, 1, 2, 2, 3, 3, 4, 4]
    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(nii.get_data(), hdr=nii.header, orientation='RPI', dim=nii.header.get_data_shape())
    return img


# noinspection 801,PyShadowingNames
def test_aggregate_across_selected_slices(dummy_metric):
    """Test extraction of metrics aggregation across slices"""
    agg_metrics = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metric, slices=[3, 4], perslice=False,
                                                                   group_funcs=(('mean', np.mean), ('std', np.std)))
    assert agg_metrics[(3, 4)]['metrics']['metric1']['mean'] == 30.0
    assert agg_metrics[(3, 4)]['metrics']['metric1']['std'] == 1.0
    assert agg_metrics[(3, 4)]['metrics']['metric2']['mean'] == 99.5
    # check that even if there is an error in metric estimation, the function outputs a dict for specific slicegroup
    assert np.isnan(agg_metrics[(3, 4)]['metrics']['metric3']['mean'])
    assert agg_metrics[(3, 4)]['metrics']['metric4']['error'] == 'metric and z have do not have the same length'
    assert agg_metrics[(3, 4)]['metrics']['metric5']['error'] == 'cannot perform reduce with flexible type'


# noinspection 801,PyShadowingNames
def test_aggregate_across_all_slices(dummy_metric):
    """Test extraction of metrics aggregation across slices"""
    agg_metrics = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metric, perslice=False,
                                                                   group_funcs=(('mean', np.mean),))
    assert agg_metrics[(3, 4, 5, 6, 7)]['metrics']['metric1']['mean'] == 38.0


# noinspection 801,PyShadowingNames
def test_aggregate_per_slice(dummy_metric):
    """Test extraction of metrics aggregation per slice"""
    agg_metrics = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metric, slices=[3, 4], perslice=True,
                                                                   group_funcs=(('mean', np.mean),))
    assert agg_metrics[(3,)]['metrics']['metric1']['mean'] == 29.0
    assert agg_metrics[(4,)]['metrics']['metric1']['mean'] == 31.0


# noinspection 801,PyShadowingNames
def test_aggregate_across_levels(dummy_metric, dummy_vert_level):
    """Test extraction of metrics aggregation across vertebral levels"""
    group_funcs = (('mean', np.mean),)
    agg_metrics = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metric, levels=[2, 3], perlevel=False,
                                                                   vert_level=dummy_vert_level,
                                                                   group_funcs=group_funcs)
    assert agg_metrics[(3, 4, 5, 6)]['metrics']['metric1']['mean'] == 35.0
    assert agg_metrics[(3, 4, 5, 6)]['VertLevel'] == (2, 3)


# noinspection 801,PyShadowingNames
def test_aggregate_per_level(dummy_metric, dummy_vert_level):
    """Test extraction of metrics aggregation per vertebral level"""
    group_funcs = (('mean', np.mean),)
    agg_metrics = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metric, levels=[2, 3], perlevel=True,
                                                                   vert_level=dummy_vert_level,
                                                                   group_funcs=group_funcs)
    assert agg_metrics[(5, 6)]['metrics']['metric1']['mean'] == 40.0
    assert agg_metrics[(5, 6)]['VertLevel'] == (3,)
    assert agg_metrics[(5, 6)]['metrics']['metric5']['mean'] == 101.5
    assert agg_metrics[(3, 4)]['metrics']['metric5']['error'] == 'cannot perform reduce with flexible type'
