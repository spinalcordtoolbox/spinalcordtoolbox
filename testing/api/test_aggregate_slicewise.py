#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.aggregate_slicewise

import sys
import os
import pytest
import csv

import numpy as np
import nibabel as nib
import pandas as pd

from spinalcordtoolbox import __sct_dir__, __version__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))

from spinalcordtoolbox import aggregate_slicewise
from spinalcordtoolbox.process_seg import Metric
from spinalcordtoolbox.image import Image


@pytest.fixture(scope="session")
def dummy_metrics():
    """Create a Dict of dummy metric."""
    metrics = {'with float': Metric(data=np.array([29., 31., 39., 41., 50., 51., 59., 62., 70.])),
               'with int': Metric(data=np.array([99, 100, 101, 102, 103, 104, 105, 106, 107])),
               'with nan': Metric(data=np.array([99, np.nan, 101, 102, 103, 104, 105, 106, 107])),
               'inconsistent length': Metric(data=np.array([99, 100])),
               'with string': Metric(data=np.array([99, "boo!", 101, 102, 103, 104, 105, 106, 107])),
               '3D': Metric(data=np.resize(np.array([99, 100, 101, 102, 103, 104, 105, 106, 107]), [4, 5, 9]))}
    return metrics


@pytest.fixture(scope="session")
def dummy_data_and_labels():
    """Create a dummy data with partial volume effect, with associated mask, for testing extract_metric()."""
    data = Metric(data=np.array([19., 21., 30., 39., 41.]))
    # Create 3 labels. The last label has very small volume fraction to assess the efficacy of MAP estimation.
    labels = np.array([[0., 0., 0.5, 1., 1.],
                       [0.9, 1., 0.5, 0., 0.],
                       [0.1, 0., 0., 0., 0.]]).T  # need to transpose because last dim are labels
    # Create label_struc{}
    label_struc = {0: aggregate_slicewise.LabelStruc(id=0, name='label_0', map_cluster=0),
                   1: aggregate_slicewise.LabelStruc(id=1, name='label_1', map_cluster=1),
                   2: aggregate_slicewise.LabelStruc(id=2, name='label_2', map_cluster=1),
                   99: aggregate_slicewise.LabelStruc(id=[1, 2], name='label_1,2', map_cluster=None)}
    return data, labels, label_struc


@pytest.fixture(scope="session")
def dummy_data_and_labels_2d():
    """Create a dummy 2d data with associated 2d label, for testing extract_metric()."""
    data = Metric(data=np.array([[5, 5],
                                 [5, 5]]))
    labels = np.array([[1, 1],
                       [1, 1]]).T  # need to transpose because last dim are labels
    labels = np.expand_dims(labels, axis=2)  # because ndim(label) = ndim(data)+1
    # Create label_struc{}
    label_struc = {0: aggregate_slicewise.LabelStruc(id=0, name='mask')}
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


@pytest.fixture(scope="session")
def dummmy_data_predictor():
    """Create a panda dataframe of dummy coefficient and mean."""
    data = [[1, 0.5], [1e-6, 1156171]]
    df = pd.DataFrame(data, columns=['coeff', 'mean'], index=['sex', 'brain-volume'])
    return df


@pytest.fixture(scope="session")
def dummmy_data_subject():
    """Create a panda dataframe of dummy data of a subject."""
    data = [[0, 960606.0]]
    df = pd.DataFrame(data, columns=['sex', 'brain-volume'])
    return df


def test_aggregate_across_selected_slices(dummy_metrics):
    """Test extraction of metrics aggregation across slices: Selected slices"""
    agg_metrics = {}
    for metric in dummy_metrics:
        agg_metrics[metric] = \
            aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics[metric], slices=[1, 2], perslice=False,
                                                             group_funcs=(('WA', aggregate_slicewise.func_wa),
                                                                          ('STD', aggregate_slicewise.func_std)))
    assert agg_metrics['with float'][(1, 2)]['WA()'] == 35.0
    assert agg_metrics['with float'][(1, 2)]['STD()'] == 4.0
    assert agg_metrics['with int'][(1, 2)]['WA()'] == 100.5
    # check that even if there is an error in metric estimation, the function outputs a dict for specific slicegroup
    assert agg_metrics['with nan'][(1, 2)]['WA()'] == 101.0
    assert agg_metrics['inconsistent length'][(1, 2)]['WA()'] == 'index 2 is out of bounds for axis 0 with size 2'
    assert agg_metrics['with string'][(1, 2)]['WA()'] == "ufunc 'isfinite' not supported for the input types, and " \
                                                           "the inputs could not be safely coerced to any supported " \
                                                           "types according to the casting rule ''safe''"


def test_aggregate_across_all_slices(dummy_metrics):
    """Test extraction of metrics aggregation across slices: All slices by default"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], perslice=False,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    assert agg_metric[list(agg_metric)[0]]['WA()'] == 48.0


def test_aggregate_per_slice(dummy_metrics):
    """Test extraction of metrics aggregation per slice: Selected slices"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], slices=[3, 4],
                                                                  perslice=True,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    assert agg_metric[(3,)]['WA()'] == 41.0
    assert agg_metric[(4,)]['WA()'] == 50.0


def test_aggregate_across_levels(dummy_metrics, dummy_vert_level):
    """Test extraction of metrics aggregation across vertebral levels"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], levels=[2, 3],
                                                                  perslice=False, perlevel=False,
                                                                  vert_level=dummy_vert_level,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    assert agg_metric[(0, 1, 2, 3)] == {'VertLevel': (2, 3), 'DistancePMJ': None, 'WA()': 35.0}


def test_aggregate_across_levels_perslice(dummy_metrics, dummy_vert_level):
    """Test extraction of metrics aggregation within selected vertebral levels and per slice"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], levels=[2, 3],
                                                                  perslice=True, perlevel=False,
                                                                  vert_level=dummy_vert_level,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    assert agg_metric[(0,)] == {'VertLevel': (2,), 'DistancePMJ': None, 'WA()': 29.0}
    assert agg_metric[(2,)] == {'VertLevel': (3,), 'DistancePMJ': None, 'WA()': 39.0}


def test_aggregate_per_level(dummy_metrics, dummy_vert_level):
    """Test extraction of metrics aggregation per vertebral level"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], levels=[2, 3],
                                                                  perlevel=True, vert_level=dummy_vert_level,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    assert agg_metric[(0, 1)] == {'VertLevel': (2,), 'DistancePMJ': None, 'WA()': 30.0}
    assert agg_metric[(2, 3)] == {'VertLevel': (3,), 'DistancePMJ': None, 'WA()': 40.0}


def test_aggregate_across_levels_and_slices(dummy_metrics, dummy_vert_level):
    """Test extraction of metrics aggregation across vertebral levels, while imposing a slice range
    Context: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3822"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], slices=[1, 2, 3, 4, 6],
                                                                  levels=[2, 3], perlevel=False, perslice=False,
                                                                  vert_level=dummy_vert_level,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    assert (1, 2, 3) in agg_metric.keys()
    assert agg_metric[(1, 2, 3)] == {'VertLevel': (2, 3), 'DistancePMJ': None, 'WA()': 37.0}

    agg_metric_ps = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], slices=[1, 2, 3, 4, 6],
                                                                     levels=[2, 3], perlevel=False, perslice=True,
                                                                     vert_level=dummy_vert_level,
                                                                     group_funcs=(('WA', aggregate_slicewise.func_wa),))
    assert ((1,), (2,), (3,)) == tuple(agg_metric_ps.keys())
    assert agg_metric_ps[(1,)] == {'VertLevel': (2,), 'DistancePMJ': None, 'WA()': 31.0}
    assert agg_metric_ps[(2,)] == {'VertLevel': (3,), 'DistancePMJ': None, 'WA()': 39.0}
    assert agg_metric_ps[(3,)] == {'VertLevel': (3,), 'DistancePMJ': None, 'WA()': 41.0}


def test_aggregate_slices_pmj(dummy_metrics):
    """Test extraction of metrics aggregation within selected slices at a PMJ distance"""
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], slices=[2, 3, 4, 5],
                                                                  distance_pmj=64, perslice=False, perlevel=False,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    assert agg_metric[(2, 3, 4, 5)] == {'VertLevel': None, 'DistancePMJ': 64, 'WA()': 45.25}


def test_extract_metric(dummy_data_and_labels):
    """Test different estimation methods."""
    # Weighted average
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=0,
                                                    perslice=False, method='wa')
    assert agg_metric[list(agg_metric)[0]]['WA()'] == 38.0

    # Binarized mask
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=0,
                                                    perslice=False, method='bin')
    assert agg_metric[list(agg_metric)[0]]['BIN()'] == pytest.approx(36.66, rel=0.01)

    # Maximum Likelihood
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=0,
                                                    indiv_labels_ids=[0, 1, 2], perslice=False, method='ml')
    assert agg_metric[list(agg_metric)[0]]['ML()'] == pytest.approx(39.9, rel=0.01)

    # Maximum A Posteriori
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=0,
                                                    indiv_labels_ids=[0, 1, 2], perslice=False, method='map')
    assert agg_metric[list(agg_metric)[0]]['MAP()'] == pytest.approx(40.0, rel=0.01)

    # Maximum
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=0,
                                                    indiv_labels_ids=[0, 1], perslice=False, method='max')
    assert agg_metric[list(agg_metric)[0]]['MAX()'] == 41.0

    # Weighted Median
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=0,
                                                    indiv_labels_ids=[0, 1], perslice=False, method='median')
    assert agg_metric[list(agg_metric)[0]]['MEDIAN()'] == 39.5

    # Combined labels
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=99,
                                                    perslice=False, method='wa')
    assert agg_metric[list(agg_metric)[0]]['WA()'] == 22.0
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=99,
                                                    indiv_labels_ids=[0, 1, 2], perslice=False, method='map')
    assert agg_metric[list(agg_metric)[0]]['MAP()'] == pytest.approx(20.0, rel=0.01)


def test_extract_metric_2d(dummy_data_and_labels_2d):
    """Test different estimation methods with 2D input array"""
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels_2d[0], labels=dummy_data_and_labels_2d[1],
                                                    label_struc=dummy_data_and_labels_2d[2], id_label=0,
                                                    indiv_labels_ids=0, perslice=False, method='wa')
    assert agg_metric[list(agg_metric)[0]]['WA()'] == 5.0

    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels_2d[0], labels=dummy_data_and_labels_2d[1],
                                                    label_struc=dummy_data_and_labels_2d[2], id_label=0,
                                                    indiv_labels_ids=0, perslice=False, method='median')
    assert agg_metric[list(agg_metric)[0]]['MEDIAN()'] == 5.0


def test_save_as_csv(tmp_path, dummy_metrics):
    """Test writing of output metric csv file"""
    path_out = str(tmp_path / 'tmp_file_out.csv')
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], slices=[3, 4],
                                                                  perslice=False,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),
                                                                               ('STD', aggregate_slicewise.func_std)))
    # standard scenario
    aggregate_slicewise.save_as_csv(agg_metric, path_out, fname_in='FakeFile.txt')
    with open(path_out, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader)  # skip header
        assert next(spamreader)[1:] == [__version__, 'FakeFile.txt', '3:4', '', '', '45.5', '4.5']
    # with appending
    aggregate_slicewise.save_as_csv(agg_metric, path_out)
    aggregate_slicewise.save_as_csv(agg_metric, path_out, append=True)
    with open(path_out, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader)  # skip header
        assert next(spamreader)[1:] == [__version__, '', '3:4', '', '', '45.5', '4.5']
        assert next(spamreader)[1:] == [__version__, '', '3:4', '', '', '45.5', '4.5']


def test_save_as_csv_slices(tmp_path, dummy_metrics, dummy_vert_level):
    """Make sure slices are listed in reduced form"""
    path_out = str(tmp_path / 'tmp_file_out.csv')
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], levels=[3, 4],
                                                                  perslice=False, perlevel=False,
                                                                  vert_level=dummy_vert_level,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    aggregate_slicewise.save_as_csv(agg_metric, path_out)
    with open(path_out, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert row['Slice (I->S)'] == '2:5'
        assert row['VertLevel'] == '3:4'


def test_save_as_csv_per_level(tmp_path, dummy_metrics, dummy_vert_level):
    """Make sure slices are listed in reduced form"""
    path_out = str(tmp_path / 'tmp_file_out.csv')
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], levels=[3, 4],
                                                                  perlevel=True,
                                                                  vert_level=dummy_vert_level,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    aggregate_slicewise.save_as_csv(agg_metric, path_out)
    with open(path_out, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert row['Slice (I->S)'] == '2:3'
        assert row['VertLevel'] == '3'


def test_save_as_csv_per_slice_then_per_level(dummy_metrics, dummy_vert_level, tmp_path):
    """Test with and without specifying perlevel. See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/2141"""
    path_out = str(tmp_path / 'tmp_file_out.csv')
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], levels=[3, 4],
                                                                  perlevel=True,
                                                                  vert_level=dummy_vert_level,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    aggregate_slicewise.save_as_csv(agg_metric, path_out)
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], slices=[0],
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),),)
    aggregate_slicewise.save_as_csv(agg_metric, path_out, append=True)
    with open(path_out, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert row['Slice (I->S)'] == '2:3'
        assert row['VertLevel'] == '3'
        next(reader)
        row = next(reader)
        assert row['Slice (I->S)'] == '0'
        assert row['VertLevel'] == ''


def test_save_as_csv_sorting(tmp_path, dummy_metrics):
    """Make sure slices are sorted in output csv file"""
    path_out = str(tmp_path / 'tmp_file_out.csv')
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], perslice=True,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    aggregate_slicewise.save_as_csv(agg_metric, path_out)
    with open(path_out, 'r') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=',')
        assert [row['Slice (I->S)'] for row in spamreader] == ['0', '1', '2', '3', '4', '5', '6', '7', '8']


def test_save_as_csv_pmj(tmp_path, dummy_metrics):
    """Test writing of output metric csv file with distance from PMJ method"""
    path_out = str(tmp_path / 'tmp_file_out.csv')
    agg_metric = aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics['with float'], slices=[2, 3, 4, 5],
                                                                  distance_pmj=64.0, perslice=False, perlevel=False,
                                                                  group_funcs=(('WA', aggregate_slicewise.func_wa),))
    aggregate_slicewise.save_as_csv(agg_metric, path_out)
    with open(path_out, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        row = next(reader)
        assert row['Slice (I->S)'] == '2:5'
        assert row['DistancePMJ'] == '64.0'
        assert row['VertLevel'] == ''


def test_save_as_csv_extract_metric(tmp_path, dummy_data_and_labels):
    """Test file output with extract_metric()"""
    path_out = str(tmp_path / 'tmp_file_out.csv')
    # With input label file
    agg_metric = aggregate_slicewise.extract_metric(dummy_data_and_labels[0], labels=dummy_data_and_labels[1],
                                                    label_struc=dummy_data_and_labels[2], id_label=0,
                                                    indiv_labels_ids=[0, 1], perslice=False, method='wa')
    aggregate_slicewise.save_as_csv(agg_metric, path_out)
    with open(path_out, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader)  # skip header
        assert next(spamreader)[1:-1] == [__version__, '', '0:4', '', '', 'label_0', '2.5', '38.0']


def test_dimension_mismatch_between_metric_and_vertfile(dummy_metrics, dummy_vert_level):
    """Test that an exception is raised only for mismatched metric and -vertfile images."""
    for metric in dummy_metrics:
        if metric == 'inconsistent length':
            with pytest.raises(ValueError) as err:
                aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics[metric], vert_level=dummy_vert_level)
            assert "mismatch" in str(err.value)
        else:
            # Verify that no error is thrown for all other metrics
            aggregate_slicewise.aggregate_per_slice_or_level(dummy_metrics[metric], vert_level=dummy_vert_level)


def test_normalize_csa(dummmy_data_predictor, dummmy_data_subject):
    """"Test normalizing CSA values (MEAN(area))"""
    csa_norm = aggregate_slicewise.normalize_csa(64, dummmy_data_predictor, dummmy_data_subject)
    assert csa_norm == 64.695565
