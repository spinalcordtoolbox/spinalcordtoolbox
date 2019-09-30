#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.resampling

# TODO: add test for 2d image

from __future__ import absolute_import

import sys, os
import pytest

import numpy as np
import nibabel as nib

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))
from spinalcordtoolbox import resampling


@pytest.fixture(scope="session")
def fake_3dimage_nib():
    """
    :return: an empty 3-d nibabel Image
    """
    nx, ny, nz = 9, 9, 9  # image dimension
    data = np.zeros((nx, ny, nz), dtype=np.int8)
    data[4, 4, 4] = 1.
    affine = np.eye(4)
    # Create nibabel object
    nii = nib.nifti1.Nifti1Image(data, affine)
    return nii


@pytest.fixture(scope="session")
def fake_3dimage_nib_big():
    """
    :return: an empty 3-d nibabel Image
    """
    nx, ny, nz = 29, 39, 19  # image dimension
    data = np.zeros((nx, ny, nz), dtype=np.int8)
    data[14, 19, 9] = 1.
    affine = np.eye(4)
    # Create nibabel object
    nii = nib.nifti1.Nifti1Image(data, affine)
    return nii


@pytest.fixture(scope="session")
def fake_4dimage_nib():
    """
    :return: an empty 4-d nibabel Image
    """
    nx, ny, nz, nt = 9, 9, 9, 3  # image dimension
    data = np.zeros((nx, ny, nz, nt), dtype=np.int8)
    data[4, 4, 4, 0] = 1.
    affine = np.eye(4)
    # Create nibabel object
    nii = nib.nifti1.Nifti1Image(data, affine)
    return nii


# noinspection 801,PyShadowingNames
def test_nib_resample_image_3d(fake_3dimage_nib):
    """Test resampling with 3D nibabel image"""
    img_r = resampling.resample_nib(fake_3dimage_nib, new_size=[2, 2, 1], new_size_type='factor', interpolation='nn')
    assert img_r.get_data().shape == (18, 18, 9)
    assert img_r.get_data()[8, 8, 4] == 1.0  # make sure there is no displacement in world coordinate system
    assert img_r.header.get_zooms() == (0.5, 0.5, 1.0)
    # debug
    # nib.save(img_r, 'test_4.nii.gz')


# noinspection 801,PyShadowingNames
def test_nib_resample_image_3d_to_dest(fake_3dimage_nib, fake_3dimage_nib_big):
    """Test resampling with 3D nibabel image"""
    img_r = resampling.resample_nib(fake_3dimage_nib, image_dest=fake_3dimage_nib_big, interpolation='linear')
    assert img_r.get_data().shape == (29, 39, 19)
    assert img_r.get_data()[4, 4, 4] == 1.0


# noinspection 801,PyShadowingNames
def test_nib_resample_image_4d(fake_4dimage_nib):
    """Test resampling with 4D nibabel image"""
    img_r = resampling.resample_nib(fake_4dimage_nib, new_size=[2, 2, 1, 1], new_size_type='factor', interpolation='nn')
    assert img_r.get_data().shape == (18, 18, 9, 3)
    assert img_r.get_data()[8, 8, 4, 0] == 1.0  # make sure there is no displacement in world coordinate system
    assert img_r.get_data()[8, 8, 4, 1] == 0.0
    assert img_r.header.get_zooms() == (0.5, 0.5, 1.0, 1.0)
