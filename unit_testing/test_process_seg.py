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
    def _dummy_seg(shape='rectangle', angle=15, a=50.0, b=30.0):
        """
        Nested function to allow input parameters
        :param shape: {'rectangle', 'ellipse'}
        :param angle: int: in deg
        :param a: float: 1st radius
        :param b: float: 2nd radius
        :return:
        """
        nx, ny, nz = 200, 200, 100  # image dimension
        fname_seg = 'dummy_segmentation.nii.gz'  # output seg
        data = np.random.random((nx, ny, nz))
        xx, yy = np.mgrid[:nx, :ny]
        # loop across slices and add an ellipse of axis length a and b
        # a, b = 50.0, 30.0  # radius of the ellipse (in pix size). Theoretical CSA: 4712.4
        for iz in range(nz):
            if shape == 'rectangle':  # theoretical CSA: (a*2+1)(b*2+1)
                data[:, :, iz] = ((abs(xx - nx / 2) <= a) & (abs(yy - ny / 2) <= b)) * 1
            if shape == 'ellipse':
                data[:, :, iz] = (((xx - nx / 2) / a) ** 2 + ((yy - ny / 2) / b) ** 2 <= 1) * 1
        # swap x-z axes (to make a rotation within y-z plane)
        data_swap = data.swapaxes(0, 2)
        # rotate by 15 deg, and re-grid using nearest neighbour interpolation (compute_shape only takes binary iputs)
        data_swap_rot = rotate(data_swap, angle, resize=False, center=None, order=1, mode='constant', cval=0,
                               clip=False, preserve_range=False)
        # swap back
        data_rot = data_swap_rot.swapaxes(0, 2)
        xform = np.eye(4)
        for i in range(3):
            xform[i][i] = 0.1  # adjust voxel dimension to get realistic spinal cord size (important for some functions)
        img = nib.nifti1.Nifti1Image(data_rot.astype('float32'), xform)
        nib.save(img, fname_seg)
        return fname_seg
        # return _dummy_seg
    return _dummy_seg


# noinspection 801,PyShadowingNames
def test_extract_centerline(dummy_segmentation):
    """Test extraction of centerline from input segmentation"""
    process_seg.extract_centerline(dummy_segmentation(shape='rectangle', angle=0, a=3.0, b=1.0), 0, file_out='centerline')
    # open created csv file
    centerline_out = []
    with open('centerline.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()  # skip header
        for row in reader:
            centerline_out.append([int(i) for i in row])
    # build ground-truth centerline
    centerline_true_20to80 = [[20, 99, 100], [40, 99, 100], [60, 99, 100], [80, 99, 100]]
    assert centerline_out[20:200:20] == centerline_true_20to80


# noinspection 801,PyShadowingNames
def test_compute_csa_noangle(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation"""
    metrics = process_seg.compute_csa(dummy_segmentation(shape='rectangle', angle=0, a=50.0, b=30.0),
                                      algo_fitting='hanning', window_length=5, angle_correction=False,
                                      use_phys_coord=True, verbose=0)
    assert np.mean(metrics['csa'].value[20:80]) == pytest.approx(61.61, rel=0.01)
    assert np.mean(metrics['angle'].value[20:80]) == pytest.approx(0.0, rel=0.01)


# noinspection 801,PyShadowingNames
def test_compute_csa(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation
    Note: here, compared to the previous tests with no angle, we use smaller hanning window and smaller range for
    computing the mean, because the smoothing creates spurious errors at edges."""
    metrics = process_seg.compute_csa(dummy_segmentation(shape='rectangle', angle=15, a=50.0, b=30.0),
                                      algo_fitting='hanning', type_window='hanning',
                                      window_length=3, angle_correction=True, use_phys_coord=True, verbose=0)
    assert np.mean(metrics['csa'].value[30:70]) == pytest.approx(61.61, rel=0.01)
    assert np.mean(metrics['angle'].value[30:70]) == pytest.approx(15.00, rel=0.01)


# noinspection 801,PyShadowingNames
def test_compute_csa_ellipse(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation"""
    metrics = process_seg.compute_csa(dummy_segmentation(shape='ellipse', angle=0, a=50.0, b=30.0),
                                      algo_fitting='hanning', type_window='hanning',
                                      window_length=5, angle_correction=True, use_phys_coord=True, verbose=0)
    assert np.mean(metrics['csa'].value[30:70]) == pytest.approx(47.01, rel=0.01)
    assert np.mean(metrics['angle'].value[30:70]) == pytest.approx(0.0, rel=0.01)


# noinspection 801,PyShadowingNames
def test_compute_shape_noangle(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation."""
    # Using hanning because faster
    metrics = process_seg.compute_shape(dummy_segmentation(shape='ellipse', angle=0, a=50.0, b=30.0),
                                        algo_fitting='hanning', window_length=3, verbose=0)
    assert np.mean(metrics['area'].value[30:70]) == pytest.approx(47.01, rel=0.05)
    assert np.mean(metrics['AP_diameter'].value[30:70]) == pytest.approx(6.0, rel=0.05)
    assert np.mean(metrics['RL_diameter'].value[30:70]) == pytest.approx(10.0, rel=0.05)
    assert np.mean(metrics['ratio_minor_major'].value[30:70]) == pytest.approx(0.6, rel=0.05)
    assert np.mean(metrics['eccentricity'].value[30:70]) == pytest.approx(0.8, rel=0.05)
    assert np.mean(metrics['orientation'].value[30:70]) == pytest.approx(0.0, rel=0.05)
    assert np.mean(metrics['solidity'].value[30:70]) == pytest.approx(1.0, rel=0.05)


# noinspection 801,PyShadowingNames
def test_compute_shape(dummy_segmentation):
    """Test computation of cross-sectional area from input segmentation."""
    # Using hanning because faster
    metrics = process_seg.compute_shape(dummy_segmentation(shape='ellipse', angle=15, a=50.0, b=30.0),
                                        algo_fitting='hanning', window_length=3, verbose=0)
    assert np.mean(metrics['area'].value[30:70]) == pytest.approx(47.01, rel=0.05)
    assert np.mean(metrics['AP_diameter'].value[30:70]) == pytest.approx(6.0, rel=0.05)
    assert np.mean(metrics['RL_diameter'].value[30:70]) == pytest.approx(10.0, rel=0.05)
    assert np.mean(metrics['ratio_minor_major'].value[30:70]) == pytest.approx(0.6, rel=0.05)
    assert np.mean(metrics['eccentricity'].value[30:70]) == pytest.approx(0.8, rel=0.05)
    assert np.mean(metrics['orientation'].value[30:70]) == pytest.approx(0.0, rel=0.05)
    assert np.mean(metrics['solidity'].value[30:70]) == pytest.approx(1.0, rel=0.05)
