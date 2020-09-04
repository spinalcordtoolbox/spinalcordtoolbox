#!/usr/bin/env python

from __future__ import print_function, absolute_import

import os

from spinalcordtoolbox.utils import sct_test_path

import sct_compute_mtsat


class InputParam1:
    mt = sct_test_path('mt', 'mt1.nii.gz')
    pd = sct_test_path('mt', 'mt0.nii.gz')
    t1 = sct_test_path('mt', 't1w.nii.gz')
    trmt = None
    trpd = None
    trt1 = None
    famt = None
    fapd = None
    fat1 = None


class InputParam2:
    mt = sct_test_path('mt', 'mt1.nii.gz')
    pd = sct_test_path('mt', 'mt0.nii.gz')
    t1 = sct_test_path('mt', 't1w.nii.gz')
    trmt = 1
    trpd = 2
    trt1 = 3
    famt = 4
    fapd = 5
    fat1 = 6


def test_get_tr_and_flipangle():
    assert sct_compute_mtsat.get_tr_and_flipangle(InputParam1()) == (0.03, 0.03, 0.015, 9, 9, 15)
    assert sct_compute_mtsat.get_tr_and_flipangle(InputParam2()) == (1, 2, 3, 4, 5, 6)


def test_with_json_sidecar():
    sct_compute_mtsat.main(['-mt', sct_test_path('mt', 'mt1.nii.gz'),
                            '-pd', sct_test_path('mt', 'mt0.nii.gz'),
                            '-t1', sct_test_path('mt', 't1w.nii.gz')])
    # Check if output file exists
    assert os.path.isfile(sct_test_path('mt', 'mtsat.nii.gz'))
