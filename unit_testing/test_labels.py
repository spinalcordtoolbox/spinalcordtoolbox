#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.labels

import logging
from time import time

import numpy as np

import spinalcordtoolbox.labels as sct_labels
from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.utils import sct_test_path

logger = logging.getLogger(__name__)

src_img = sct_test_path('t2', 't2.nii.gz')


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
