#!/usr/bin/env python

from __future__ import print_function, absolute_import

import os
import sys
import pytest

import spinalcordtoolbox as sct
from spinalcordtoolbox.utils import sct_test_path
from spinalcordtoolbox import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))

import sct_compute_mtsat


def test_with_json_sidecar():
    sct_compute_mtsat.main(['-mt', sct_test_path('mt', 'mt1.nii.gz'),
                            '-pd', sct_test_path('mt', 'mt0.nii.gz'),
                            '-t1', sct_test_path('mt', 't1w.nii.gz')])
    # Check if output file exists
    assert os.path.isfile(sct_test_path('mt', 'mtsat.nii.gz'))
