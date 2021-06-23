#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.csa_pmj

import sys
import os
import pytest
import numpy as np

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))
from spinalcordtoolbox.image import Image
from spinalcordtoolbox import csa_pmj
from spinalcordtoolbox.utils import sct_test_path


@pytest.fixture(scope="session")
def centerline():
    """ Get centerline ndarray from sct_testing_data/t2/t2_centerline-manual.nii.gz """
    fname_ctl = sct_test_path('t2', 't2_centerline-manual.nii.gz')
    im_ctl = Image(fname_ctl).change_orientation('RPI')
    _, _, _, _, px, py, pz, _ = im_ctl.dim
    # Create an array like output of get_centerline()
    arr_ctl = np.array(np.nonzero(im_ctl.data))
    arr_ctl = arr_ctl[::, arr_ctl[2, ].argsort()]
    return arr_ctl, px, py, pz


def test_distance_from_pmj(centerline):
    """ Test computing distance from PMJ """
    arr_ctl, px, py, pz = centerline
    z_index = arr_ctl[2, :].max()  # Get max z index
    arr_lenght = csa_pmj.get_distance_from_pmj(arr_ctl, z_index, px, py, pz)
    # Check total lenght of the centerline (at z index 0)
    assert arr_lenght[0][0] == pytest.approx(54.82842712)
    assert arr_lenght[0][-1] == 0  # Check if first value of lenght is 0.


def test_get_min_distance(centerline):
    """ Test getting index of minimum distance from PMJ and centerline """
    pmj = np.array([38, 30, 49])  # RPI
    arr_ctl, px, py, pz = centerline
    min_index = csa_pmj.get_min_distance(pmj, arr_ctl, px, py, pz)
    assert min_index == 49
