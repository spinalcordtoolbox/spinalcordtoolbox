#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.qmri

import numpy as np
import nibabel
import pytest

from spinalcordtoolbox.qmri import mt
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import init_sct, set_loglevel


# Set logger to "DEBUG"
init_sct()
set_loglevel(verbose=2)


def make_sct_image(data):
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    data: scalar
    """
    affine = np.eye(4)
    nii = nibabel.nifti1.Nifti1Image(np.array([data, data]), affine)
    img = Image(nii.get_data(), hdr=nii.header, orientation="LPI", dim=nii.header.get_data_shape())
    return img


def test_compute_mtr():
    img_mtr = mt.compute_mtr(nii_mt1=make_sct_image(10),
                             nii_mt0=make_sct_image(20)
                             )
    assert img_mtr.data[0] == 0.5 * 100  # output is in percent


def test_compute_mtsat():
    img_mtsat, img_t1map = mt.compute_mtsat(nii_mt=make_sct_image(1500),
                                            nii_pd=make_sct_image(2000),
                                            nii_t1=make_sct_image(1500),
                                            tr_mt=0.030,
                                            tr_pd=0.030,
                                            tr_t1=0.015,
                                            fa_mt=9,
                                            fa_pd=9,
                                            fa_t1=15
                                            )
    assert img_mtsat.data[0] == pytest.approx(1.5327, 0.0001)
    assert img_t1map.data[0] == pytest.approx(0.8916, 0.0001)
