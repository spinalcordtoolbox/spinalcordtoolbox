from __future__ import absolute_import

import os
import sys

import numpy as np
import nibabel as nib

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.deepseg_lesion import core as deepseg_lesion

import sct_utils as sct


def test_model_file_exists():
    for model_name in deepseg_lesion.MODEL_LST:
        model_path = os.path.join(sct.__sct_dir__, 'data', 'deepseg_lesion_models', '{}_lesion.h5'.format(model_name))
        assert os.path.isfile(model_path)


def test_segment():
    contrast_test = 't2'
    model_path = os.path.join(sct.__sct_dir__, 'data', 'deepseg_lesion_models', '{}_lesion.h5'.format(contrast_test))

    # create fake data
    data = np.zeros((48,48,96))
    xx, yy = np.mgrid[:48, :48]
    circle = (xx - 24) ** 2 + (yy - 24) ** 2
    for zz in range(data.shape[2]):
        data[:,:,zz] += np.logical_and(circle < 400, circle >= 200) * 2400 # CSF
        data[:,:,zz] += (circle < 200) * 500 # SC
    data[16:22, 16:22, 64:90] = 1000 # fake lesion

    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())

    seg = deepseg_lesion.segment_3d(model_path, contrast_test, img.copy())

    assert np.any(seg.data[16:22, 16:22, 64:90]) == True  # check if lesion detected
    assert np.any(seg.data[img.data != 1000]) == False  # check if no FP


def test_intensity_normalization():
    data_in = np.random.rand(10, 10)
    min_out, max_out = 0.0, 2611.0
    landmarks_lst = sorted(list(np.random.uniform(low=500.0, high=2000.0, size=(11,), )))

    data_out = deepseg_lesion.apply_intensity_normalization_model(data_in, landmarks_lst)
    data_out = np.nan_to_num(data_out)  # replace NaN with zero

    assert data_in.shape == data_out.shape
    assert data_out.dtype == np.float32
    assert np.min(data_out) >= min_out
    assert np.max(data_out) <= max_out
