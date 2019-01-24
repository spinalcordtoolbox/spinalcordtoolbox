#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.centerline


from __future__ import absolute_import

import pytest

import numpy as np
import nibabel as nib

from spinalcordtoolbox.centerline.core import get_centerline
from spinalcordtoolbox.image import Image

# TODO: test various orientations.
# TODO: create synthetic centerline using polynomial functions.


@pytest.fixture(scope="session")
def dummy_centerline_small():
    """
    Create a dummy Image centerline of small size. Return the full and sub-sampled version along z.
    """
    nx, ny, nz = 9, 9, 9
    data = np.zeros((nx, ny, nz))
    # define curved centerline with value=1 voxels.
    data[4, 4, 0] = 1
    data[4, 4, 1] = 1
    data[4, 4, 2] = 1
    data[4, 5, 3] = 1
    data[5, 5, 4] = 1
    data[5, 6, 5] = 1
    data[5, 7, 6] = 1
    data[5, 8, 7] = 1
    data[5, 8, 8] = 1
    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, orientation='RPI', dim=nii.header.get_data_shape())
    # define sub data
    data_sub = np.zeros((nx, ny, nz))
    for iz in range(0, nz, 3):
        data_sub[..., iz] = data[..., iz]
    img_sub = Image(data_sub, hdr=nii.header, orientation='RPI', dim=nii.header.get_data_shape())
    return img, img_sub


# noinspection 801,PyShadowingNames
def test_get_centerline(dummy_centerline_small):
    """Test extraction of metrics aggregation across slices: All slices by default"""
    img, img_sub = dummy_centerline_small
    img_out, arr_out = get_centerline(img, algo_fitting='polyfit')
    assert np.linalg.norm(np.where(img.data) - arr_out) < 5  # TODO: adjust this threshold
