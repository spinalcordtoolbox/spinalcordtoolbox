#!/usr/bin/env python
#########################################################################################
#
# Test function sct_deepseg_sc
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener & Charley Gros
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

import sys, io, os

from pandas import DataFrame

import sct_utils as sct
from spinalcordtoolbox.image import Image, compute_dice


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2.nii.gz -c t2 -qc testing-qc']  # default parameters

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Integrity test has moved to unit_testing/ and is run with pytest
    """
    return param_test
