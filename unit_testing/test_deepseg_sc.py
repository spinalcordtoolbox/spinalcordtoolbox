from __future__ import absolute_import

import os
import numpy as np
import nibabel as nib
import sct_utils as sct
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.deepseg_sc import core as deepseg_sc
from spinalcordtoolbox.resample.nipy_resample import resample_file

def _create_fake_t2_sc(input_size):
    data = np.zeros(input_size)
    xx, yy = np.mgrid[:input_size[0], :input_size[1]]
    circle = (xx - input_size[0] // 2) ** 2 + (yy - input_size[1] // 2) ** 2

    for zz in range(data.shape[2]):
        data[:,:,zz] += np.logical_and(circle < 400, circle >= 200) * 250 # CSF
        data[:,:,zz] += (circle < 200) * 100 # SC
        # Note: values of the fake SC and CSF have been chosen by looking at the normalized intensities of im_norm_in (cf deepseg_sc/core)

    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    img = Image(data, hdr=nii.header, dim=nii.header.get_data_shape())

    return img


def test_segment_2d():
    from keras import backend as K
    K.set_image_data_format("channels_last")  # Set at channels_first in test_deepseg_lesion.test_segment()

    contrast_test = 't2'
    model_path = os.path.join(sct.__sct_dir__, 'data', 'deepseg_sc_models', '{}_sc.h5'.format(contrast_test))   

    fname_t2 = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2.nii.gz')  # install: sct_download_data -d sct_testing_data
    fname_t2_seg = os.path.join(sct.__sct_dir__, 'sct_testing_data/t2/t2_seg.nii.gz')  # install: sct_download_data -d sct_testing_data

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

    seg = deepseg_sc.segment_2d(model_fname=model_path, contrast_type=contrast_test, input_size=(64,64), im_in=img)
    seg_im = msct_image.zeros_like(img)
    seg_im.data = seg
    print msct_image.compute_dice(seg_im, gt)
    
    tmp_folder.chdir_undo()
    tmp_folder.cleanup()
    # seg_im = msct_image.zeros_like(img)
    # seg_gt_im = seg_im.copy()
    # seg_gt_im.data = (img.data == 100)
    # seg_im.data = seg

    # assert np.any(seg[img.data == 100]) == True  # check if SC detected
    # assert np.any(seg[img.data != 100]) == False  # check if no FP
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

def test_segment_3d():
    pass # todo
