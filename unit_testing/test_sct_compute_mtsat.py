#!/usr/bin/env python

from __future__ import print_function, absolute_import

import os
import pytest

from spinalcordtoolbox.utils import sct_test_path

import sct_compute_mtsat


INPUT_PARAMS = [
    ['-mt', sct_test_path('mt', 'mt1.nii.gz'),
     '-pd', sct_test_path('mt', 'mt0.nii.gz'),
     '-t1', sct_test_path('mt', 't1w.nii.gz')],
    ['-mt', sct_test_path('mt', 'mt1.nii.gz'),
     '-pd', sct_test_path('mt', 'mt0.nii.gz'),
     '-t1', sct_test_path('mt', 't1w.nii.gz'),
     '-trmt', '51', '-trpd', '52', '-trt1', '10', '-famt', '4', '-fapd', '5', '-fat1', '14'],
    ]


@pytest.mark.parametrize('input_params', INPUT_PARAMS)
def test_with_json_sidecar(input_params):
    sct_compute_mtsat.main(input_params)
    # Check if output file exists
    assert os.path.isfile(sct_test_path('mt', 'mtsat.nii.gz'))
