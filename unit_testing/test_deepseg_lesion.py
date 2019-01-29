from __future__ import absolute_import

import os
import numpy as np
import nibabel as nib
import sct_utils as sct
from spinalcordtoolbox.image import Image

from spinalcordtoolbox.deepseg_lesion import core as deepseg_lesion

def test_model_file_exists(self):
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
