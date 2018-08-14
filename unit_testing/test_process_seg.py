#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.process_seg

from __future__ import absolute_import

import csv

import pytest

import numpy as np
import nibabel as nib
from skimage.transform import rotate

from spinalcordtoolbox import process_seg


@pytest.fixture(scope="session")
def dummy_segmentation():
    """Create a dummy image with a ellipse or ones running from top to bottom in the 3rd dimension, and rotate the image
    to make sure that compute_csa and compute_shape properly estimate the centerline angle."""
    nx, ny, nz = 20, 20, 20  # image dimension
    fname_seg = 'dummy_segmentation.nii.gz'  # output seg
    data = np.random.random((nx, ny, nz))
    xx, yy = np.mgrid[:nx, :ny]
    # loop across slices and add an ellipse of axis length a and b
    a, b = 5.0, 3.0  # diameter of ellipse
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
    img = nib.nifti1.Nifti1Image(data_rot, xform)
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
    k = [7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 13]
    centerline_true = [[i, 9, k[i]] for i in range(20)]
    assert centerline_out == centerline_true


# noinspection 801,PyShadowingNames
def test_compute_csa(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation"""
    metrics = process_seg.compute_csa(dummy_segmentation, 1, 1, 1, '5:15', '', fname_vert_levels='', perslice=0,
                                      perlevel=0, algo_fitting='hanning', type_window='hanning', window_length=10,
                                      angle_correction=True, use_phys_coord=True, file_out='csa')
    assert np.mean(metrics['CSA [mm^2]'][5:15]) == pytest.approx(45, abs=1e-3)
    assert np.mean(metrics['Angle between cord axis and z [deg]'][5:15]) == pytest.approx(15, abs=1e-3)


# noinspection 801,PyShadowingNames
def test_compute_shape(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation."""
    # here we only quantify between 5:15 because we want to avoid edge effects due to the rotation.
    process_seg.compute_shape(dummy_segmentation, slices='5:15', vert_levels='', fname_vert_levels='', perslice=0,
                              perlevel=0, file_out='shape', overwrite=0, remove_temp_files=1, verbose=1)
    # open created csv file
    with open('shape.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()  # skip header
        area, equivalent_diameter, AP_diameter, RL_diameter, ratio_minor_major, eccentricity, solidity, orientation, \
        symmetry = [float(i) for i in reader.next()[2:]]
    assert area == pytest.approx(44.863, abs=1e-3)
    assert equivalent_diameter == pytest.approx(7.554, abs=1e-3)
    assert AP_diameter == pytest.approx(5.807, abs=1e-3)
    assert RL_diameter == pytest.approx(10.170, abs=1e-3)
    assert ratio_minor_major == pytest.approx(0.571, abs=1e-3)
    assert eccentricity == pytest.approx(0.818, abs=1e-3)
    assert solidity == pytest.approx(0.854, abs=1e-3)
    assert orientation == pytest.approx(-0.010, abs=1e-3)
    assert symmetry == pytest.approx(0.998, abs=1e-3)
