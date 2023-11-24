# pytest unit tests for spinalcordtoolbox.resampling

# TODO: add test for 2d image

import pytest

import numpy as np
import nibabel as nib

from spinalcordtoolbox import resampling


@pytest.fixture(scope="session")
def fake_3dimage_nib():
    """
    :return: an almost empty 3-d nibabel Image
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
    :return: an almost empty 3-d nibabel Image
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
    :return: an almost empty 4-d nibabel Image
    """
    nx, ny, nz, nt = 9, 9, 9, 3  # image dimension
    data = np.zeros((nx, ny, nz, nt), dtype=np.int8)
    data[4, 4, 4, 0] = 1.
    affine = np.eye(4)
    # Create nibabel object
    nii = nib.nifti1.Nifti1Image(data, affine)
    return nii


def test_nib_resample_image_3d(fake_3dimage_nib):
    """Test resampling with 3D nibabel image"""
    img_r = resampling.resample_nib(fake_3dimage_nib, new_size=[2, 2, 1], new_size_type='factor', interpolation='nn')
    assert np.asanyarray(img_r.dataobj).shape == (18, 18, 9)
    assert np.asanyarray(img_r.dataobj)[8, 8, 4] == 1.0  # make sure there is no displacement in world coordinate system
    assert img_r.header.get_zooms() == (0.5, 0.5, 1.0)
    # debug
    # nib.save(img_r, 'test_4.nii.gz')


def test_nib_resample_image_3d_to_dest(fake_3dimage_nib, fake_3dimage_nib_big):
    """Test resampling with 3D nibabel image"""
    img_r = resampling.resample_nib(fake_3dimage_nib, image_dest=fake_3dimage_nib_big, interpolation='linear')
    assert np.asanyarray(img_r.dataobj).shape == (29, 39, 19)
    assert np.asanyarray(img_r.dataobj)[4, 4, 4] == 1.0


def test_nib_resample_image_4d(fake_4dimage_nib):
    """Test resampling with 4D nibabel image"""
    img_r = resampling.resample_nib(fake_4dimage_nib, new_size=[2, 2, 1, 1], new_size_type='factor', interpolation='nn')
    assert np.asanyarray(img_r.dataobj).shape == (18, 18, 9, 3)
    assert np.asanyarray(img_r.dataobj)[8, 8, 4, 0] == 1.0  # make sure there is no displacement in world coordinate system
    assert np.asanyarray(img_r.dataobj)[8, 8, 4, 1] == 0.0
    assert img_r.header.get_zooms() == (0.5, 0.5, 1.0, 1.0)


def test_nib_resample_image_convert_to_float(fake_3dimage_nib, caplog):
    """Test that arithmetic resampling of an integer image gets converted to floating point first."""
    assert fake_3dimage_nib.dataobj.dtype.kind == 'i', "test input should have an integer dtype"
    img_r = resampling.resample_nib(fake_3dimage_nib, new_size=[2, 1, 1], new_size_type='factor', interpolation='linear')
    assert img_r.dataobj.dtype.kind == 'f', "test output should have a floating point dtype"
    assert "Converting image" in caplog.text, "there should be a type conversion warning"


def test_nib_resample_image_no_convert_to_float(fake_3dimage_nib):
    """Test that a nearest neighbour resampling of an integer image stays integer."""
    assert fake_3dimage_nib.dataobj.dtype.kind == 'i', "test input should have an integer dtype"
    img_r = resampling.resample_nib(fake_3dimage_nib, new_size=[2, 1, 1], new_size_type='factor', interpolation='nn')
    assert img_r.dataobj.dtype == fake_3dimage_nib.dataobj.dtype, "nearest neighbour resampling should not have changed the dtype"
