#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.labels

import logging
from time import time

import numpy as np

import spinalcordtoolbox.labels as sct_labels
from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.utils import sct_test_path
from spinalcordtoolbox.types import Coordinate

logger = logging.getLogger(__name__)

src_img = sct_test_path('t2', 't2.nii.gz')


# AJ remove test if we keep add_faster + refactor caller
def test_add():
    a = Image(src_img)
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
    a = Image(src_img)
    ref = zeros_like(a)

    labels = [Coordinate(l) for l in [[12, 24, 32, 7], [15, 35, 33, 5]]]
    ref.data[12][24][32] = 7
    ref.data[15][35][33] = 5

    b = sct_labels.create_labels_empty(a, labels)

    c = b.data == ref.data
    assert c.all()


def test_create_labels():
    a = Image(src_img)
    labels = [Coordinate(l) for l in [[1, 2, 3, 99], [14, 28, 33, 5]]]

    b = sct_labels.create_labels(a, labels)

    assert b.data[1][2][3] == 99
    assert b.data[14][28][33] == 5
