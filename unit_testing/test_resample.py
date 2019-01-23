#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.resample


from __future__ import absolute_import

import pytest

import numpy as np
import nibabel as nib
from nipy.io.nifti_ref import nifti2nipy, nipy2nifti

from spinalcordtoolbox.resample import nipy_resample


@pytest.fixture(scope="session")
def fake_3dimage_nipy():
    """
    :return: an empty 3-d nipy Image
    """
    nx, ny, nz = 9, 9, 9  # image dimension
    data = np.zeros((nx, ny, nz), dtype=np.int8)
    data[4, 4, 4] = 1.
    affine = np.eye(4)
    # Create nibabel object
    nii = nib.nifti1.Nifti1Image(data, affine)
    # return nipy object
    return nifti2nipy(nii)


@pytest.fixture(scope="session")
def fake_4dimage_nipy():
    """
    :return: an empty 4-d nipy Image
    """
    nx, ny, nz, nt = 9, 9, 9, 3  # image dimension
    data = np.zeros((nx, ny, nz, nt), dtype=np.int8)
    data[4, 4, 4, 0] = 1.
    affine = np.eye(4)
    # Create nibabel object
    nii = nib.nifti1.Nifti1Image(data, affine)
    # return nipy object
    return nifti2nipy(nii)


# noinspection 801,PyShadowingNames
def test_nipy_resample_image_3d(fake_3dimage_nipy):
    """Test resampling with 3D nipy image"""
    img_r = nipy_resample.resample_nipy(fake_3dimage_nipy, new_size='2x2x1', new_size_type='factor', interpolation='nn')
    assert img_r.get_data().shape == (18, 18, 9)
    assert img_r.get_data()[8, 8, 4] == 1.0  # make sure there is no displacement in world coordinate system
    assert nipy2nifti(img_r).header.get_zooms() == (0.5, 0.5, 1.0)
    # debug
    # nib.save(nipy2nifti(img_r), 'test_4.nii.gz')


# noinspection 801,PyShadowingNames
def test_nipy_resample_image_4d(fake_4dimage_nipy):
    """Test resampling with 4D nipy image"""
    img_r = nipy_resample.resample_nipy(fake_4dimage_nipy, new_size='2x2x1x1', new_size_type='factor', interpolation='nn')
    assert img_r.get_data().shape == (18, 18, 9, 3)
    assert img_r.get_data()[8, 8, 4, 0] == 1.0  # make sure there is no displacement in world coordinate system
    assert img_r.get_data()[8, 8, 4, 1] == 0.0
    assert nipy2nifti(img_r).header.get_zooms() == (0.5, 0.5, 1.0, 1.0)
