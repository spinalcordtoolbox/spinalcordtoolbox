#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.deepseg_sc


from __future__ import absolute_import

import pytest
import numpy as np
import nibabel as nib
from keras import backend as K

import spinalcordtoolbox as sct
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.deepseg_sc.core
from spinalcordtoolbox.testing.create_test_data import dummy_centerline


param_deepseg = [
    ({'fname_seg_manual': 'sct_testing_data/t2/t2_seg-deepseg_sc-2d.nii.gz', 'contrast': 't2', 'kernel': '2d'}),
    ({'fname_seg_manual': 'sct_testing_data/t2/t2_seg-deepseg_sc-3d.nii.gz', 'contrast': 't2', 'kernel': '3d'}),
]

# noinspection 801,PyShadowingNames
@pytest.mark.parametrize('params', param_deepseg)
def test_deep_segmentation_spinalcord(params):
    """High level segmentation API"""
    fname_im = 'sct_testing_data/t2/t2.nii.gz'
    fname_centerline_manual = 'sct_testing_data/t2/t2_centerline-manual.nii.gz'
    # Set at channels_first in test_deepseg_lesion.test_segment()
    K.set_image_data_format("channels_last")
    # Call segmentation function
    im_seg, _, _ = sct.deepseg_sc.core.deep_segmentation_spinalcord(
        Image(fname_im), params['contrast'], ctr_algo='file', ctr_file=fname_centerline_manual, brain_bool=False,
        kernel_size=params['kernel'], threshold_seg=0.5)
    assert im_seg.data.dtype == np.dtype('uint8')
    # Compare with ground-truth segmentation
    assert np.all(im_seg.data == Image(params['fname_seg_manual']).data)


def test_intensity_normalization():
    data_in = np.random.rand(10, 10)
    min_out, max_out = 0, 255

    data_out = sct.deepseg_sc.core.scale_intensity(data_in, out_min=0, out_max=255)

    assert data_in.shape == data_out.shape
    assert np.min(data_out) >= min_out
    assert np.max(data_out) <= max_out


def test_crop_image_around_centerline():
    input_shape = (100, 100, 100)
    crop_size = 20

    data = np.random.rand(input_shape[0], input_shape[1], input_shape[2])
    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())

    ctr, _, _ = dummy_centerline(size_arr=input_shape)

    _, _, _, img_out = sct.deepseg_sc.core.crop_image_around_centerline(
        im_in=img.copy(), ctr_in=ctr.copy(), crop_size=crop_size)

    img_in_z0 = img.data[:,:,0]
    x_ctr_z0, y_ctr_z0 = np.where(ctr.data[:,:,0])[0][0], np.where(ctr.data[:,:,0])[1][0]
    x_start, x_end = sct.deepseg_sc.core._find_crop_start_end(x_ctr_z0, crop_size, img.dim[0])
    y_start, y_end = sct.deepseg_sc.core._find_crop_start_end(y_ctr_z0, crop_size, img.dim[1])
    img_in_z0_crop = img_in_z0[x_start:x_end, y_start:y_end]

    assert img_out.data.shape == (crop_size, crop_size, input_shape[2])
    assert np.allclose(img_in_z0_crop, img_out.data[:,:,0])


def test_uncrop_image():
    input_shape = (100, 100, 100)
    crop_size = 20
    data_crop = np.random.randint(0, 2, size=(crop_size, crop_size, input_shape[2]))
    data_in = np.random.randint(0, 1000, size=input_shape)

    x_crop_lst = list(np.random.randint(0, input_shape[0]-crop_size, input_shape[2]))
    y_crop_lst = list(np.random.randint(0,input_shape[1]-crop_size, input_shape[2]))
    z_crop_lst = range(input_shape[2])

    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data_in, affine)
    img_in = Image(data_in, hdr=nii.header, dim=nii.header.get_data_shape())

    img_uncrop = sct.deepseg_sc.core.uncrop_image(
        ref_in=img_in, data_crop=data_crop, x_crop_lst=x_crop_lst, y_crop_lst=y_crop_lst, z_crop_lst=z_crop_lst)

    assert img_uncrop.data.shape == input_shape
    z_rand = np.random.randint(0, input_shape[2])
    assert np.allclose(img_uncrop.data[x_crop_lst[z_rand]:x_crop_lst[z_rand]+crop_size,
                       y_crop_lst[z_rand]:y_crop_lst[z_rand]+crop_size,
                       z_rand],
                       data_crop[:, :, z_rand])
