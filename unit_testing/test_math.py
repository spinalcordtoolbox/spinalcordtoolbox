#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.math


from __future__ import absolute_import
import sys
import os
import pytest
import math
import numpy as np

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))
from spinalcordtoolbox import math

from create_test_data import dummy_blob


# Define global variables
VERBOSE = 0  # set to 2 to save files
DEBUG = False  # Set to True to save images

# Generate a list of dummy images with single pixel in the middle
list_im = [
    # test area
    (dummy_blob(size_arr=(9, 9, 9), debug=DEBUG)),
    ]

# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('im', list_im)
def test_dilate(im):
    data_dil = math.dilate(im.data, radius=3, shape='disk', dim=0)
    a=1
