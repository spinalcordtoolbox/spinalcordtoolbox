#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.resample


from __future__ import absolute_import

import pytest

import numpy as np
import nibabel as nib
from nipy.io.nifti_ref import nifti2nipy

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


# noinspection 801,PyShadowingNames
def test_nipy_resample_image(fake_3dimage_nipy):
    """Test resampling with 3D nipy image"""

    img_r = nipy_resample.resample_image(fake_3dimage_nipy, new_size='2x2x1', new_size_type='factor', interpolation='nn')
    assert img_r.get_data().shape == (18, 18, 9)
    assert img_r.get_data()[8, 8, 4] == 1.0  # make sure there is no displacement in world coordinate system
