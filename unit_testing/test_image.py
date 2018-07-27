#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for Image stuff

import sys, io, os, time, itertools

import pytest

import numpy as np
import nibabel
import nibabel.orientations

import sct_utils as sct
import msct_image
import sct_image

@pytest.fixture(scope="session")
def image_paths():
    ret = []
    sct_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(sct_dir, "data")
    for cwd, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".nii.gz", ".nii")):
                path = os.path.join(cwd, file)
                ret.append(path)
    return ret



@pytest.fixture(scope="session")
def fake_3dimage():
    """
    :return: a Nifti1Image (3D) in RAS+ space

    Following characteristics:

    - shape[LR] = 7
    - shape[PA] = 8
    - shape[IS] = 9
    """
    shape = (7,8,9)
    data = np.zeros(shape, dtype=np.float32, order="F")

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                data[x,y,z] = (1+x)*1 + (1+y)*10 + (1+z)*100

    if 0:
        for z in range(shape[2]):
            for y in range(shape[1]):
                for x in range(shape[0]):
                    sys.stdout.write(" % 3d" % data[x,y,z])
                sys.stdout.write("\n")
            sys.stdout.write("\n")

    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)

@pytest.fixture(scope="session")
def fake_3dimage2():
    """
    :return: a Nifti1Image (3D) in RAS+ space

    Following characteristics:

    - shape[LR] = 1
    - shape[PA] = 2
    - shape[IS] = 3
    """
    shape = (1,2,3)
    data = np.zeros(shape, dtype=np.float32, order="F")

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                data[x,y,z] = (1+x)*1 + (1+y)*10 + (1+z)*100

    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)


@pytest.fixture(scope="session")
def fake_3dimage_sct():
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    """
    i = fake_3dimage()
    img = msct_image.Image(i.get_data(), hdr=i.header,
     orientation="LPI",
     dim=i.header.get_data_shape(),
    )
    return img

@pytest.fixture(scope="session")
def fake_3dimage_sct2():
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    """
    i = fake_3dimage2()
    img = msct_image.Image(i.get_data(), hdr=i.header,
     orientation="LPI",
     dim=i.header.get_data_shape(),
    )
    return img


def test_slicer(fake_3dimage_sct, fake_3dimage_sct2):
    im3d = fake_3dimage_sct
    slicer = msct_image.Slicer(im3d, "IS")
    assert slicer.direction == +1
    assert slicer.nb_slices == 9
    if 0:
        for im2d in slicer:
            print(im2d)

    assert 100 < np.mean(slicer[0]) < 200

    slicer = msct_image.Slicer(im3d, "SI")
    assert slicer.direction == -1
    assert slicer.nb_slices == 9

    if 0:
        for im2d in slicer:
            print(im2d)

    assert 900 < np.mean(slicer[0]) < 1000

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "LI")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "L")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "Lx")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "LA")

    im3d2 = fake_3dimage_sct.copy()
    im3d2.data += 1000
    slicer = msct_image.Slicer((im3d, im3d2), "IS")
    for idx_slice, (im2d_a, im2d_b) in enumerate(slicer):
        assert np.all(im2d_b == im2d_a + 1000)

    im3d2 = fake_3dimage_sct2
    with pytest.raises(ValueError):
        slicer = msct_image.Slicer((im3d, im3d2), "IS")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(1)
def test_change_shape(fake_3dimage_sct):

    # Add dimension
    im_src = fake_3dimage_sct
    shape = tuple(list(im_src.data.shape) + [1])
    im_dst = msct_image.change_shape(im_src, shape)
    path_tmp = sct.tmp_create(basename="test_reshape")
    src_path = os.path.join(path_tmp, "src.nii")
    dst_path = os.path.join(path_tmp, "dst.nii")
    im_src.save(src_path)
    im_dst.save(dst_path)
    im_src = msct_image.Image(src_path)
    im_dst = msct_image.Image(dst_path)
    assert im_dst.data.shape == shape

    data_src = im_src.data
    data_dst = im_dst.data

    assert (data_dst.reshape(data_src.shape) == data_src).all()

    # Remove dimension
    im_dst = im_dst.change_shape(im_src.data.shape)
    assert im_dst.data.shape == im_src.data.shape
