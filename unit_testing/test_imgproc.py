#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.imgproc

import sys, io, os

import pytest

import msct_image
import spinalcordtoolbox.imgproc

@pytest.fixture(scope="session")
def an_image_path():
    sct_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(sct_dir, "data")
    for cwd, dirs, files in os.walk(data_dir):
        for file in files:
            if file == "t2.nii.gz":
                path = os.path.join(cwd, file)
                return path


def test_binarize(an_image_path):

    binarize = spinalcordtoolbox.imgproc.binarize

    for src in (an_image_path, msct_image.Image(an_image_path)):
        dst = "img_bin.nii.gz"
        dst_ = binarize(src, dst, threshold="otsu")

        dst = msct_image.Image(dst)
        dst_ = binarize(src, dst, threshold=(50, "%"))

        dst = None
        dst_ = binarize(src, dst, threshold=0.3)
        assert isinstance(dst_, type(src))
