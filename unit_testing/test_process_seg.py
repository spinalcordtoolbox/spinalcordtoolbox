#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.process_seg

# TODO: directly pass image (not fname)
# TODO: add dummy image with different resolution to check impact of input res

from __future__ import absolute_import
import pytest
import numpy as np
import csv
import nibabel as nib
from skimage.transform import rotate
from spinalcordtoolbox import process_seg


@pytest.fixture(scope="session")
def dummy_segmentation():
    """Create a dummy image with a ellipse or ones running from top to bottom in the 3rd dimension, and rotate the image
    to make sure that compute_csa and compute_shape properly estimate the centerline angle.
    :return: fname_seg: filename of 3D binary image
    """
    nx, ny, nz = 200, 200, 200  # image dimension
    fname_seg = 'dummy_segmentation.nii.gz'  # output seg
    data = np.random.random((nx, ny, nz))
    xx, yy = np.mgrid[:nx, :ny]
    # loop across slices and add an ellipse of axis length a and b
    a, b = 50.0, 30.0  # diameter of ellipse (in pix size). Theoretical CSA: 4712.4
    for iz in range(nz):
        data[:, :, iz] = (((xx - nx / 2) / a) ** 2 + ((yy - ny / 2) / b) ** 2 <= 1) * 1
    # swap x-z axes (to make a rotation within y-z plane)
    data_swap = data.swapaxes(0, 2)
    # rotate by 15 deg, and re-grid using nearest neighbour interpolation (compute_shape only takes binary iputs)
    data_swap_rot = rotate(data_swap, 15, resize=False, center=None, order=0, mode='constant', cval=0, clip=True,
                           preserve_range=False)
    # swap back
    data_rot = data_swap_rot.swapaxes(0, 2)
    xform = np.eye(4)
    for i in range(3):
        xform[i][i] = 0.1  # adjust voxel dimension to get realistic spinal cord size (important for some functions)
    img = nib.nifti1.Nifti1Image(data_rot.astype('int8'), xform)
    nib.save(img, fname_seg)
    return fname_seg


# noinspection 801,PyShadowingNames
def test_extract_centerline(dummy_segmentation):
    """Test extraction of centerline from input segmentation"""
    process_seg.extract_centerline(dummy_segmentation, 0, file_out='centerline')
    # open created csv file
    centerline_out = []
    with open('centerline.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()  # skip header
        for row in reader:
            centerline_out.append([int(i) for i in row])
    # build ground-truth centerline
    centerline_true_20to180 = [[20, 99, 79], [60, 99, 89], [100, 99, 100], [140, 99, 111], [180, 99, 122]]
    assert centerline_out[20:200:40] == centerline_true_20to180


# noinspection 801,PyShadowingNames
def test_compute_csa(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation"""
    metrics = process_seg.compute_csa(dummy_segmentation, algo_fitting='hanning', type_window='hanning',
                                      window_length=50, angle_correction=True, use_phys_coord=True)
    assert np.mean(metrics['csa'].value[20:180]) == pytest.approx(46.94, rel=0.01)
    assert np.mean(metrics['angle'].value[20:180]) == pytest.approx(14.96, rel=0.01)


# noinspection 801,PyShadowingNames
def test_compute_shape(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation."""
    # Using hanning because faster
    metrics = process_seg.compute_shape(dummy_segmentation, algo_fitting='hanning', window_length=50)
    assert np.mean(metrics['area'].value[20:180]) == pytest.approx(46.30, rel=0.01)
    assert np.mean(metrics['AP_diameter'].value[20:180]) == pytest.approx(6, rel=1)
    assert np.mean(metrics['symmetry'].value[20:180]) == pytest.approx(1.0, rel=1e-8)
    assert np.mean(metrics['ratio_minor_major'].value[20:180]) == pytest.approx(0.59, rel=0.01)
    assert np.mean(metrics['solidity'].value[20:180]) == pytest.approx(0.94, rel=0.01)
    assert np.mean(metrics['RL_diameter'].value[20:180]) == pytest.approx(10, rel=1)
    assert np.mean(metrics['equivalent_diameter'].value[20:180]) == pytest.approx(7.7, rel=0.1)
    assert np.mean(metrics['eccentricity'].value[20:180]) == pytest.approx(0.8, rel=0.1)

    # TODO: add other metrics
    # TODO: check why area different from compute_csa
