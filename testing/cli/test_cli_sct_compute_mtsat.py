#!/usr/bin/env python

import os
import pytest
import nibabel
import numpy as np

from spinalcordtoolbox.utils import sct_test_path

from spinalcordtoolbox.scripts import sct_compute_mtsat

out_mstat = "out_mtsat.nii.gz"
out_t1map = "out_t1map.nii.gz"

expected_mean_mtsat = 1.95260
expected_mean_t1 = 1.19374

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
     '-trmt', '0.030', '-trpd', '0.030', '-trt1', '0.015', '-famt', '9', '-fapd', '9', '-fat1', '15'],
    ]


@pytest.mark.parametrize('input_params', INPUT_PARAMS)
def test_files_are_created(input_params):
    sct_compute_mtsat.main(input_params)

    # Check if output files exist
    for f in [out_mstat, out_t1map]:
        assert os.path.isfile(f)
        os.remove(f)


@pytest.mark.parametrize('input_params', INPUT_PARAMS)
def test_expected_values(input_params):
    sct_compute_mtsat.main(input_params)

    mtsat = nibabel.load(out_mstat)
    mtsat_data = mtsat.get_fdata()
    np.testing.assert_almost_equal(mtsat_data.mean(), expected_mean_mtsat, decimal=3)

    t1map = nibabel.load(out_t1map)
    t1map_data = t1map.get_fdata()
    np.testing.assert_almost_equal(t1map_data.mean(), expected_mean_t1, decimal=3)

    # Remove files
    for f in [out_mstat, out_t1map]:
        os.remove(f)
