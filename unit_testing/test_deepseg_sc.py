from __future__ import absolute_import

import os
import numpy as np
import sct_utils as sct
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.deepseg_sc import core as deepseg_sc
from spinalcordtoolbox.resample.nipy_resample import resample_file


def _preprocess_segment(fname_t2, fname_t2_seg, contrast_test, dim_3=False):
    tmp_folder = sct.TempFolder()
    tmp_folder_path = tmp_folder.get_path()
    tmp_folder.chdir()

    img = Image(fname_t2)
    gt = Image(fname_t2_seg)

    fname_t2_RPI, fname_t2_seg_RPI = 'img_RPI.nii.gz', 'seg_RPI.nii.gz'
    img.change_orientation('RPI').save(fname_t2_RPI)
    gt.change_orientation('RPI').save(fname_t2_seg_RPI)
    input_resolution = gt.dim[4:7]
    del img, gt

    fname_res, fname_ctr = deepseg_sc.find_centerline(algo='svm',
                                                        image_fname=fname_t2_RPI,
                                                        contrast_type=contrast_test,
                                                        brain_bool=False,
                                                        folder_output=tmp_folder_path,
                                                        remove_temp_files=1,
                                                        centerline_fname=None)

    fname_t2_seg_RPI_res = 'seg_RPI_res.nii.gz'
    new_resolution = 'x'.join(['0.5', '0.5', str(input_resolution[2])])
    resample_file(fname_t2_seg_RPI, fname_t2_seg_RPI_res, new_resolution, 'mm', 'linear', verbose=0)

    img, ctr, gt = Image(fname_res), Image(fname_ctr), Image(fname_t2_seg_RPI_res)
    _, _, img = deepseg_sc.crop_image_around_centerline(im_in=img,
                                                        ctr_in=ctr,
                                                        crop_size=64)
    _, _, gt = deepseg_sc.crop_image_around_centerline(im_in=gt,
                                                        ctr_in=ctr,
                                                        crop_size=64)
    del ctr

    img = deepseg_sc.apply_intensity_normalization(im_in=img)

    if dim_3:  # If 3D kernels
        fname_t2_RPI_res_crop, fname_t2_seg_RPI_res_crop = 'img_RPI_res_crop.nii.gz', 'seg_RPI_res_crop.nii.gz'
        img.save(fname_t2_RPI_res_crop)
        gt.save(fname_t2_seg_RPI_res_crop)
        del img, gt

        fname_t2_RPI_res_crop_res = 'img_RPI_res_crop_res.nii.gz'
        fname_t2_seg_RPI_res_crop_res = 'seg_RPI_res_crop_res.nii.gz'
        resample_file(fname_t2_RPI_res_crop, fname_t2_RPI_res_crop_res, new_resolution, 'mm', 'linear', verbose=0)
        resample_file(fname_t2_seg_RPI_res_crop, fname_t2_seg_RPI_res_crop_res, new_resolution, 'mm', 'linear', verbose=0)
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
    seg_im = msct_image.zeros_like(img)
    seg_im.data = seg

    assert msct_image.compute_dice(seg_im, gt) > 0.80


def test_segment_3d():
    from keras import backend as K
    K.set_image_data_format("channels_last")  # Set at channels_first in test_deepseg_lesion.test_segment()

    contrast_test = 't2'
    model_path = os.path.join(sct.__sct_dir__, 'data', 'deepseg_sc_models', '{}_sc_3D.h5'.format(contrast_test))   

    fname_t2 = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2.nii.gz')  # install: sct_download_data -d sct_testing_data
    fname_t2_seg = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2_seg.nii.gz')  # install: sct_download_data -d sct_testing_data

    img, gt = _preprocess_segment(fname_t2, fname_t2_seg, contrast_test, dim_3=True)

    seg_im = deepseg_sc.segment_3d(model_fname=model_path, contrast_type=contrast_test, im_in=img)

    assert msct_image.compute_dice(seg_im, gt) > 0.80


def test_intensity_normalization():
    data_in = np.random.rand(10, 10)
    min_out, max_out = 0, 255

    data_out = deepseg_sc.scale_intensity(data_in, out_min=0, out_max=255)

    assert data_in.shape == data_out.shape
    assert np.min(data_out) >= min_out
    assert np.max(data_out) <= max_out


def test_crop_image_around_centerline():
    pass # todo

def test_fill_z_holes():
    pass # todo

def test_remove_blobs():
    pass # todo

def test_uncrop_image():
    pass # todo
