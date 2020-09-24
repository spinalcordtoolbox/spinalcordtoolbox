#!/usr/bin/env python

from __future__ import print_function, absolute_import

import os
import pytest

from spinalcordtoolbox.utils import sct_test_path

import sct_compute_mtsat

out_mstat = "out_mtsat.nii.gz"
out_t1map = "out_t1map.nii.gz"

INPUT_PARAMS = [
    ['-mt', sct_test_path('mt', 'mt1.nii.gz'),
     '-pd', sct_test_path('mt', 'mt0.nii.gz'),
     '-t1', sct_test_path('mt', 't1w.nii.gz'),
     '-omtsat', out_mstat,
     '-ot1map', out_t1map],
    ['-mt', sct_test_path('mt', 'mt1.nii.gz'),
     '-pd', sct_test_path('mt', 'mt0.nii.gz'),
     '-t1', sct_test_path('mt', 't1w.nii.gz'),
     '-omtsat', out_mstat,
     '-ot1map', out_t1map,
     '-trmt', '51', '-trpd', '52', '-trt1', '10', '-famt', '4', '-fapd', '5', '-fat1', '14'],
    ]


@pytest.mark.parametrize('input_params', INPUT_PARAMS)
def test_with_json_sidecar(input_params):
    sct_compute_mtsat.main(input_params)
    # Check if output files exist
    for f in [out_mstat, out_t1map]:
        assert os.path.isfile(f)
        os.remove(f)
