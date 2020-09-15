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
from test_image import fake_3dimage

logger = logging.getLogger(__name__)

src_seg = sct_test_path('t2', 't2_seg-manual.nii.gz')


def fake_image():
    i = fake_3dimage()
    img = Image(i.get_data(), hdr=i.header,
                orientation="RPI",
                dim=i.header.get_data_shape(),
                )
    return img


# AJ remove test if we keep add_faster + refactor caller
def test_add():
    a = fake_image()
    val = 4

    t1 = time()
    sct_added = sct_labels.add(a, val)
    t2 = time()
    np_added = sct_labels.add_faster(a, val)
    t3 = time()

    c = sct_added.data == np_added.data
    assert c.all()

    l1 = t2 - t1
    l2 = t3 - t2
    logger.debug(f"time to run sct_labels.add() -> {l1}")
    logger.debug(f"time to run np add -> {l2}")
    logger.debug(f"speed x improvement -> {l1/l2}")


def test_create_labels_empty():
    a = fake_image()
    ref = zeros_like(a)

    labels = [Coordinate(l) for l in [[1, 2, 3, 7], [4, 5, 6, 5]]]
    ref.data[1][2][3] = 7
    ref.data[4][5][6] = 5

    b = sct_labels.create_labels_empty(a, labels)

    c = b.data == ref.data
    assert c.all()


def test_create_labels():
    a = fake_image()
    labels = [Coordinate(l) for l in [[1, 2, 3, 99], [4, 5, 6, 5]]]

    b = sct_labels.create_labels(a, labels)

    assert b.data[1][2][3] == 99
    assert b.data[4][5][6] == 5


def test_create_labels_along_segmentation():
    a = Image(src_seg)
    labels = [(5, 1), (14, 2), (23, 3)]

    og_orientation = a.orientation
    b = sct_labels.create_labels_along_segmentation(a, labels)

    assert b.orientation == og_orientation

    # TODO [AJ] how to validate labels?


@pytest.mark.skip(reason="To be implemented")
def test_cubic_to_point():
    raise NotImplementedError()


@pytest.mark.skip(reason="To be implemented")
def test_increment_z_inverse():
    raise NotImplementedError()


@pytest.mark.skip(reason="To be implemented")
def test_labelize_from_disks():
    raise NotImplementedError()


@pytest.mark.skip(reason="To be implemented")
def test_label_vertebrae():
    raise NotImplementedError()


@pytest.mark.skip(reason="To be implemented")
def test_compute_mean_squared_error():
    raise NotImplementedError()


def test_remove_missing_labels():
    src = fake_image()
    ref = fake_image()

    # change 2 labels in ref
    ref.data[1][2][3] = 99
    ref.data[3][2][1] = 99

    res = sct_labels.remove_missing_labels(src, ref)

    # check that missing labels have been removed
    assert res.data[1][2][3] == 0
    assert res.data[3][2][1] == 0

    # random spot check
    assert src.data[1][1][1] == res.data[1][1][1]
    assert src.data[2][2][2] == res.data[2][2][2]


@pytest.mark.skip(reason="To be implemented")
def test_get_coordinates_in_destination():
    raise NotImplementedError()


def test_labels_diff():
    src = fake_image()
    ref = fake_image()

    # change some labels
    ref.data[1][2][3] = 99
    ref.data[3][2][1] = 99

    src.data[4][5][6] = 99

    # check that changes appear in diff
    missing_from_ref, missing_from_src = sct_labels.labels_diff(src, ref)
    assert missing_from_ref[0] == Coordinate([1, 2, 3, src.data[1][2][3]])
    assert missing_from_ref[1] == Coordinate([3, 2, 1, src.data[3][2][1]])
    assert missing_from_src[0] == Coordinate([4, 5, 6, ref.data[4][5][6]])