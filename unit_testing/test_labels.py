#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.labels

import logging
from time import time

import numpy as np
import pytest

import spinalcordtoolbox.labels as sct_labels
from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.utils import sct_test_path
from spinalcordtoolbox.types import Coordinate
from test_image import fake_3dimage, fake_3dimage2

logger = logging.getLogger(__name__)

seg_img = Image(sct_test_path('t2', 't2_seg-manual.nii.gz'))
t2_img = Image(sct_test_path('t2', 't2.nii.gz'))
labels_img = Image(sct_test_path('t2', 'labels.nii.gz'))


# TODO [AJ] investigate how to parametrize fixtures from test_image.py
# without redefining the function here
def fake_3dimage_sct2():
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    shape = (1,2,3)
    """
    i = fake_3dimage2()
    img = Image(i.get_data(), hdr=i.header,
                orientation="RPI",
                dim=i.header.get_data_shape(),
                )
    return img


def fake_3dimage_sct():
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    shape = (7,8,9)
    """
    i = fake_3dimage()
    img = Image(i.get_data(), hdr=i.header,
                orientation="LPI",
                dim=i.header.get_data_shape(),
                )
    return img


test_images = [fake_3dimage_sct(), fake_3dimage_sct2(), t2_img]


@pytest.mark.parametrize("test_image", test_images)
def test_create_labels_empty(test_image):
    a = test_image.copy()
    expected = zeros_like(a)

    labels = [Coordinate(l) for l in [[0, 0, 0, 7], [0, 1, 2, 5]]]
    expected.data[0, 0, 0] = 7
    expected.data[0, 1, 2] = 5

    b = sct_labels.create_labels_empty(a, labels)

    diff = b.data == expected.data
    assert diff.all()


@pytest.mark.parametrize("test_image", test_images)
def test_create_labels(test_image):
    a = test_image.copy()
    labels = [Coordinate(l) for l in [[0, 1, 0, 99], [0, 1, 2, 5]]]

    b = sct_labels.create_labels(a, labels)

    assert b.data[0, 1, 0] == 99
    assert b.data[0, 1, 2] == 5


@pytest.mark.parametrize("test_seg", [seg_img])
def test_create_labels_along_segmentation(test_seg):
    a = test_seg.copy()
    labels = [(5, 1), (14, 2), (23, 3)]

    b = sct_labels.create_labels_along_segmentation(a, labels)

    assert b.orientation == a.orientation
    # TODO [AJ] implement test


@pytest.mark.parametrize("test_image", test_images)
def test_cubic_to_point(test_image):
    a = test_image.copy()
    sct_labels.cubic_to_point(a)
    # TODO [AJ] implement test

@pytest.mark.parametrize("test_image", test_images)
def test_increment_z_inverse(test_image):
    a = zeros_like(test_image.copy())
    a.data[0, 1, 0] = 1
    a.data[0, 1, 1] = 5
    a.data[0, 1, 2] = 2

    expected = a.copy()
    expected.data[0, 1, 0] = 3
    expected.data[0, 1, 1] = 2
    expected.data[0, 1, 2] = 1

    b = sct_labels.increment_z_inverse(a)

    diff = b.data == expected.data
    assert diff.all()


@pytest.mark.parametrize("test_seg,test_labels", [(seg_img, labels_img)])
def test_labelize_from_discs(test_seg, test_labels):
    seg = test_seg.copy()
    ref = test_labels.copy()

    sct_labels.labelize_from_discs(seg, ref)
    # TODO [AJ] implement test


def test_label_vertebrae():
    a = fake_3dimage_sct2()
    expected = zeros_like(a)
    expected.data[0, 0, 0] = 111
    b = sct_labels.label_vertebrae(a, [111])

    diff = b.data == expected.data
    assert diff.all()


def test_compute_mean_squared_error():
    src = fake_3dimage_sct()
    ref = src.copy()

    for x, y, z, _ in src.getNonZeroCoordinates():
        if z < 5:
            ref.data[x, y, z+2] = src.data[x, y, z]

    mse = sct_labels.compute_mean_squared_error(src, ref)
    assert mse == 1.1547005383792515


@pytest.mark.skip(reason="Too long to run on large image!")
@pytest.mark.parametrize("test_image", test_images)
def test_remove_missing_labels(test_image):
    src = test_image.copy()
    ref = test_image.copy()

    expected = test_image.copy()

    # introduce 2 discrepancies
    src.data[0, 0, 0] = 99
    src.data[0, 1, 2] = 99

    ref.data[0, 0, 0] = 1
    ref.data[0, 1, 2] = 1

    # manually set expected
    expected.data[0, 0, 0] = 0
    expected.data[0, 1, 2] = 0

    res = sct_labels.remove_missing_labels(src, ref)
    diff = res.data == expected.data

    assert diff.all()


@pytest.mark.parametrize("test_image", test_images)
def test_continuous_vertebral_levels(test_image):
    a = test_image.copy()
    b = sct_labels.continuous_vertebral_levels(a)

    # check that orientation is maintained
    assert b.orientation == a.orientation
    # TODO [AJ] implement test


@pytest.mark.parametrize("test_image", test_images)
def test_remove_labels_from_image(test_image):
    img = test_image.copy()
    expected = test_image.copy()

    labels = [1, 2]

    img.data[0, 0, 0] = 1
    img.data[0, 1, 0] = 2

    res = sct_labels.remove_labels_from_image(img, labels)

    expected.data[0, 0, 0] = 0
    expected.data[0, 1, 0] = 0

    diff = res.data == expected.data
    assert diff.all()


@pytest.mark.parametrize("test_image", test_images)
def test_remove_other_labels_from_image(test_image):
    img = test_image.copy()
    expected = zeros_like(test_image)

    labels = [5, 6]

    img.data[0, 0, 0] = 5
    img.data[0, 1, 0] = 6

    res = sct_labels.remove_other_labels_from_image(img, labels)

    expected.data[0, 0, 0] = 5
    expected.data[0, 1, 0] = 6

    diff = res.data == expected.data

    assert diff.all()


@pytest.mark.parametrize("test_image", test_images)
def test_find_missing_label(test_image):
    img = test_image.copy()
    false_positive = test_image.copy()

    # modifying the data to create one false negative and one false positive
    # label 1500 is not in test_image originally, and label 111 is.
    false_positive.data = np.where(false_positive.data == 111, 1500, false_positive.data)
    FP, FN = sct_labels.find_missing_label(false_positive, img)

    assert int(FP[0]) == 1500
    assert int(FN[0]) == 111
