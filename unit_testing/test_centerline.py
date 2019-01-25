#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.centerline

# TODO: test various orientations.
# TODO: create synthetic centerline using polynomial functions.
# TODO: adjust this threshold

from __future__ import absolute_import

import os
import pytest
import tempfile
import numpy as np
import nibabel as nib

from spinalcordtoolbox.centerline import optic
from spinalcordtoolbox.centerline.core import get_centerline
from spinalcordtoolbox.image import Image


def temp_file_nii():
    return os.path.join(tempfile.tempdir, 'img.nii')


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
def test_get_centerline_polyfit(dummy_centerline_small):
    """Test extraction of metrics aggregation across slices: All slices by default"""
    img, img_sub = dummy_centerline_small
    img_out, arr_out = get_centerline(img, algo_fitting='polyfit')
    assert np.linalg.norm(np.where(img.data) - arr_out) < 5


# noinspection 801,PyShadowingNames
def test_get_centerline_optic(dummy_centerline_small):
    """Test extraction of metrics aggregation across slices: All slices by default"""
    img, img_sub = dummy_centerline_small
    path_script = os.path.dirname(__file__)
    path_sct = os.path.dirname(path_script)
    optic_models_path = os.path.join(path_sct, 'data', 'optic_models', '{}_model'.format('t1'))
    file_nii = temp_file_nii()
    _ = img.save(file_nii)
    img_ctr = optic.detect_centerline(image_fname=file_nii, optic_models_path=optic_models_path,
                                      file_output='img_centerline.nii')
    assert np.linalg.norm(np.where(img.data) - np.array(np.where(img_ctr.data))) < 10
