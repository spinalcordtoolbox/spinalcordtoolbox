#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.aggregate_slicewise


from __future__ import absolute_import

import pytest
import csv

import numpy as np
import nibabel as nib

import sct_utils as sct
from spinalcordtoolbox import aggregate_slicewise
from spinalcordtoolbox.process_seg import Metric
from spinalcordtoolbox.image import Image


@pytest.fixture(scope="session")
def dummy_metrics():
    """Create a Dict of dummy metric."""
    metrics = {'with float': Metric(data=np.array([29., 31., 39., 41., 50.])),
               'with int': Metric(data=np.array([99, 100, 101, 102, 103])),
               'with nan': Metric(data=np.array([99, np.nan, 101, 102, 103])),
               'inconsistent length': Metric(data=np.array([99, 100])),
               'with string': Metric(data=np.array([99, "boo!", 101, 102, 103]))}
    return metrics


@pytest.fixture(scope="session")
def dummy_data_and_labels():
    """Create a dummy data with partial volume effect, with associated mask, for testing extract_metric()."""
    data = Metric(data=np.array([20., 20., 30., 40., 40.]))
    labels = np.array([[0., 0., 0.5, 1., 1.],
                       [1., 1., 0.5, 0., 0.]]).T  # need to transpose because last dim are labels
    # Create label_struc{}
    label_struc = {0: aggregate_slicewise.LabelStruc(id=0, name='label_0'),
                   1: aggregate_slicewise.LabelStruc(id=1, name='label_1')}

    return data, labels, label_struc


@pytest.fixture(scope="session")
def dummy_vert_level():
    """
    Create a dummy Image representing vertebral labeling.
    Note: The z-size of this image can to be equal or larger than the metric's length, however the indexation needs
    to match the data last dim (typically "z").
    Example: data[4, 4, 5] = 2 means that at z=5, the vertebral level is C2.
    """
    nx, ny, nz = 9, 9, 9  # image dimension
    data = np.zeros((nx, ny, nz))
    # define vertebral level for each slice as a pixel at the center of the image
    data[4, 4, :] = [2, 2, 3, 3, 4, 4, 5, 5, 6]
    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(nii.get_data(), hdr=nii.header, orientation='RPI', dim=nii.header.get_data_shape())
    return img


# noinspection 801,PyShadowingNames
def test_aggregate_across_selected_slices(dummy_metrics):
    """Test extraction of metrics aggregation across slices: Selected slices"""
    agg_metrics = {}
    for metric in dummy_metrics:
        agg_metrics[metric] = \
            aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics[metric], slices=[1, 2], perslice=False,
                                                             group_funcs=(('MEAN', aggregate_slicewise.func_wa),
                                                                          ('STD', aggregate_slicewise.func_std)))
    assert agg_metrics['with float'][(1, 2)]['MEAN()'] == 35.0
    assert agg_metrics['with float'][(1, 2)]['STD()'] == 4.0
    assert agg_metrics['with int'][(1, 2)]['MEAN()'] == 100.5
    # check that even if there is an error in metric estimation, the function outputs a dict for specific slicegroup
    assert 'error' in agg_metrics['with nan'][(1, 2)]
    assert 'error' in agg_metrics['inconsistent length'][(1, 2)]
    assert 'error' in agg_metrics['with string'][(1, 2)]


# noinspection 801,PyShadowingNames
def test_aggregate_across_all_slices(dummy_metrics):
    """Test extraction of metrics aggregation across slices: All slices by default"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], perslice=False,
                                                                  group_funcs=(('MEAN', aggregate_slicewise.func_wa),))
    assert agg_metric[agg_metric.keys()[0]]['MEAN()'] == 38.0


# noinspection 801,PyShadowingNames
def test_aggregate_per_slice(dummy_metrics):
    """Test extraction of metrics aggregation per slice: Selected slices"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], slices=[3, 4],
                                                                  perslice=True,
                                                                  group_funcs=(('MEAN', aggregate_slicewise.func_wa),))
    assert agg_metric[(3,)]['MEAN()'] == 41.0
    assert agg_metric[(4,)]['MEAN()'] == 50.0


# noinspection 801,PyShadowingNames
def test_aggregate_across_levels(dummy_metrics, dummy_vert_level):
    """Test extraction of metrics aggregation across vertebral levels"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], levels=[2, 3],
                                                                  perslice=False, perlevel=False,
                                                                  vert_level=dummy_vert_level,
                                                                  group_funcs=(('MEAN', aggregate_slicewise.func_wa),))
    assert agg_metric[(0, 1, 2, 3)] == {'VertLevel': (2, 3), 'MEAN()': 35.0}


# noinspection 801,PyShadowingNames
def test_aggregate_across_levels_perslice(dummy_metrics, dummy_vert_level):
    """Test extraction of metrics aggregation within selected vertebral levels and per slice"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], levels=[2, 3],
                                                                  perslice=True, perlevel=False,
                                                                  vert_level=dummy_vert_level,
                                                                  group_funcs=(('MEAN', aggregate_slicewise.func_wa),))
    assert agg_metric[(0,)] == {'VertLevel': (2,), 'MEAN()': 29.0}
    assert agg_metric[(2,)] == {'VertLevel': (3,), 'MEAN()': 39.0}


# noinspection 801,PyShadowingNames
def test_aggregate_per_level(dummy_metrics, dummy_vert_level):
    """Test extraction of metrics aggregation per vertebral level"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], levels=[2, 3],
                                                                  perlevel=True, vert_level=dummy_vert_level,
                                                                  group_funcs=(('MEAN', aggregate_slicewise.func_wa),))
    assert agg_metric[(0, 1)] == {'VertLevel': (2,), 'MEAN()': 30.0}
    assert agg_metric[(2, 3)] == {'VertLevel': (3,), 'MEAN()': 40.0}


# noinspection 801,PyShadowingNames
def test_extract_metric(dummy_data_and_labels):
    """Test different estimation methods."""
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=0,
                                                    indiv_labels_ids=[0, 1], perslice=False, method='wa')
    assert agg_metric[agg_metric.keys()[0]]['WA()'] == 38.0

    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=0,
                                                    indiv_labels_ids=[0, 1], perslice=False, method='ml')
    assert agg_metric[agg_metric.keys()[0]]['ML()'] == 40.0


# noinspection 801,PyShadowingNames
def test_save_as_csv(dummy_metrics):
    """Test writing of output metric csv file"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], slices=[3, 4],
                                                                  perslice=False,
                                                                  group_funcs=(('MEAN', aggregate_slicewise.func_wa),
                                                                               ('STD', aggregate_slicewise.func_std)))
    # standard scenario
    aggregate_slicewise.save_as_csv(agg_metric, 'tmp_file_out.csv', fname_in='FakeFile.txt')
    with open('tmp_file_out.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        spamreader.next()  # skip header
        assert spamreader.next() == ['FakeFile.txt', '3:4', '', '45.5', '4.5', sct.__version__]
    # with appending
    aggregate_slicewise.save_as_csv(agg_metric, 'tmp_file_out.csv')
    aggregate_slicewise.save_as_csv(agg_metric, 'tmp_file_out.csv', append=True)
    with open('tmp_file_out.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        spamreader.next()  # skip header
        assert spamreader.next() == ['', '3:4', '', '45.5', '4.5', sct.__version__]
        assert spamreader.next() == ['', '3:4', '', '45.5', '4.5', sct.__version__]


# noinspection 801,PyShadowingNames
def test_save_as_csv_slices(dummy_metrics, dummy_vert_level):
    """Make sure slices are listed in reduced form"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], levels=[3, 4],
                                                                  perslice=False, perlevel=False,
                                                                  vert_level=dummy_vert_level,
                                                                  group_funcs=(('MEAN', aggregate_slicewise.func_wa),))
    aggregate_slicewise.save_as_csv(agg_metric, 'tmp_file_out.csv')
    with open('tmp_file_out.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = reader.next()
        assert row['Slice (I->S)'] == '2:5'
        assert row['Vertebral level'] == '3:4'


# noinspection 801,PyShadowingNames
def test_save_as_csv_sorting(dummy_metrics):
    """Make sure slices are sorted in output csv file"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], perslice=True,
                                                                  group_funcs=(('MEAN', aggregate_slicewise.func_wa),))
    aggregate_slicewise.save_as_csv(agg_metric, 'tmp_file_out.csv')
    with open('tmp_file_out.csv', 'r') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=',')
        assert [row['Slice (I->S)'] for row in spamreader] == ['0', '1', '2', '3', '4']


# noinspection 801,PyShadowingNames
def test_save_as_csv_extract_metric(dummy_data_and_labels):
    """Test file output with extract_metric()"""
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=0,
                                                    indiv_labels_ids=[0, 1], perslice=False, method='wa')
    aggregate_slicewise.save_as_csv(agg_metric, 'tmp_file_out.csv')
    with open('tmp_file_out.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        spamreader.next()  # skip header
        assert spamreader.next() == ['', '0:4', '', '38.0', 'label_0', '4.0', sct.__version__]



# TODO: test extract_metric with single file.