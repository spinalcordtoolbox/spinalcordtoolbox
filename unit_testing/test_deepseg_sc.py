from __future__ import absolute_import

import os
import sys

import numpy as np
import nibabel as nib

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))
import sct_utils as sct
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.deepseg_sc import core as deepseg_sc
from spinalcordtoolbox import resampling
from create_test_data import dummy_centerline


def _preprocess_segment(fname_t2, fname_t2_seg, contrast_test, dim_3=False):
    tmp_folder = sct.TempFolder()
    tmp_folder_path = tmp_folder.get_path()
    tmp_folder.chdir()

    img = Image(fname_t2)
    gt = Image(fname_t2_seg)

    fname_t2_RPI, fname_t2_seg_RPI = 'img_RPI.nii.gz', 'seg_RPI.nii.gz'

    img.change_orientation('RPI')
    gt.change_orientation('RPI')
    new_resolution = 'x'.join(['0.5', '0.5', str(img.dim[6])])
    
    img_res = \
        resampling.resample_nib(img, new_size=[0.5, 0.5, img.dim[6]], new_size_type='mm', interpolation='linear')
    gt_res = \
        resampling.resample_nib(gt, new_size=[0.5, 0.5, img.dim[6]], new_size_type='mm', interpolation='linear')

    img_res.save(fname_t2_RPI)

    _, ctr_im, _ = deepseg_sc.find_centerline(algo='svm',
                                                image_fname=fname_t2_RPI,
                                                contrast_type=contrast_test,
                                                brain_bool=False,
                                                folder_output=tmp_folder_path,
                                                remove_temp_files=1,
                                                centerline_fname=None)

    _, _, _, img = deepseg_sc.crop_image_around_centerline(im_in=img_res,
                                                        ctr_in=ctr_im,
                                                        crop_size=64)
    _, _, _, gt = deepseg_sc.crop_image_around_centerline(im_in=gt_res,
                                                        ctr_in=ctr_im,
                                                        crop_size=64)
    del ctr_im

    img = deepseg_sc.apply_intensity_normalization(im_in=img)

    if dim_3:  # If 3D kernels
        fname_t2_RPI_res_crop, fname_t2_seg_RPI_res_crop = 'img_RPI_res_crop.nii.gz', 'seg_RPI_res_crop.nii.gz'
        img.save(fname_t2_RPI_res_crop)
        gt.save(fname_t2_seg_RPI_res_crop)
        del img, gt

        fname_t2_RPI_res_crop_res = 'img_RPI_res_crop_res.nii.gz'
        fname_t2_seg_RPI_res_crop_res = 'seg_RPI_res_crop_res.nii.gz'
        resampling.resample_file(fname_t2_RPI_res_crop, fname_t2_RPI_res_crop_res, new_resolution, 'mm', 'linear', verbose=0)
        resampling.resample_file(fname_t2_seg_RPI_res_crop, fname_t2_seg_RPI_res_crop_res, new_resolution, 'mm', 'linear', verbose=0)
        img, gt = Image(fname_t2_RPI_res_crop_res), Image(fname_t2_seg_RPI_res_crop_res)

    tmp_folder.chdir_undo()
    tmp_folder.cleanup()

    return img, gt


def test_segment_2d():
    from keras import backend as K
    K.set_image_data_format("channels_last")  # Set at channels_first in test_deepseg_lesion.test_segment()

    contrast_test = 't2'
    model_path = os.path.join(sct.__sct_dir__, 'data', 'deepseg_sc_models', '{}_sc.h5'.format(contrast_test))   

    fname_t2 = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2.nii.gz')  # install: sct_download_data -d sct_testing_data
    fname_t2_seg = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2_seg.nii.gz')  # install: sct_download_data -d sct_testing_data

    img, gt = _preprocess_segment(fname_t2, fname_t2_seg, contrast_test)

    seg = deepseg_sc.segment_2d(model_fname=model_path, contrast_type=contrast_test, input_size=(64,64), im_in=img)
    seg_im = img.copy()
    seg_im.data = (seg > 0.5).astype(np.uint8)

    assert msct_image.compute_dice(seg_im, gt) > 0.80


def test_segment_3d():
    from keras import backend as K
    K.set_image_data_format("channels_last")  # Set at channels_first in test_deepseg_lesion.test_segment()

    contrast_test = 't2'
    model_path = os.path.join(sct.__sct_dir__, 'data', 'deepseg_sc_models', '{}_sc_3D.h5'.format(contrast_test))   

    fname_t2 = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2.nii.gz')  # install: sct_download_data -d sct_testing_data
    fname_t2_seg = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2_seg.nii.gz')  # install: sct_download_data -d sct_testing_data

    img, gt = _preprocess_segment(fname_t2, fname_t2_seg, contrast_test, dim_3=True)

    seg = deepseg_sc.segment_3d(model_fname=model_path, contrast_type=contrast_test, im_in=img)
    seg_im = img.copy()
    seg_im.data = (seg > 0.5).astype(np.uint8)

    assert msct_image.compute_dice(seg_im, gt) > 0.80


def test_intensity_normalization():
    data_in = np.random.rand(10, 10)
    min_out, max_out = 0, 255

    data_out = deepseg_sc.scale_intensity(data_in, out_min=0, out_max=255)

    assert data_in.shape == data_out.shape
    assert np.min(data_out) >= min_out
    assert np.max(data_out) <= max_out


def test_crop_image_around_centerline():
    input_shape = (100, 100, 100)
    crop_size = 20
    crop_size_half = crop_size // 2

    data = np.random.rand(input_shape[0], input_shape[1], input_shape[2])
    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())

    ctr, _, _ = dummy_centerline(size_arr=input_shape)

    _, _, _, img_out = deepseg_sc.crop_image_around_centerline(im_in=img.copy(),
                                                        ctr_in=ctr.copy(),
                                                        crop_size=crop_size)

    img_in_z0 = img.data[:,:,0]
    x_ctr_z0, y_ctr_z0 = np.where(ctr.data[:,:,0])[0][0], np.where(ctr.data[:,:,0])[1][0]
    x_start, x_end = deepseg_sc._find_crop_start_end(x_ctr_z0, crop_size, img.dim[0])
    y_start, y_end = deepseg_sc._find_crop_start_end(y_ctr_z0, crop_size, img.dim[1])
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

    img_uncrop = deepseg_sc.uncrop_image(ref_in=img_in,
                                        data_crop=data_crop,
                                        x_crop_lst=x_crop_lst,
                                        y_crop_lst=y_crop_lst,
                                        z_crop_lst=z_crop_lst)

    assert img_uncrop.data.shape == input_shape
    z_rand = np.random.randint(0, input_shape[2])
    assert np.allclose(img_uncrop.data[x_crop_lst[z_rand]:x_crop_lst[z_rand]+crop_size,
                                        y_crop_lst[z_rand]:y_crop_lst[z_rand]+crop_size,
                                        z_rand],
                        data_crop[:, :, z_rand])

