#!/usr/bin/env python
# -*- coding: utf-8
# Functions dealing with deepseg_sc

import os, sys, logging

import numpy as np
from scipy.ndimage.measurements import center_of_mass, label
from skimage.exposure import rescale_intensity
from scipy.ndimage import distance_transform_edt
import nibabel as nib

from spinalcordtoolbox import resampling
from .cnn_models import nn_architecture_seg, nn_architecture_ctr
from .postprocessing import post_processing_volume_wise, keep_largest_object, fill_holes_2d
from spinalcordtoolbox.image import Image, empty_like, change_type, zeros_like
from spinalcordtoolbox.centerline.core import ParamCenterline, get_centerline, _call_viewer_centerline

import sct_utils as sct
from sct_image import concat_data, split_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BATCH_SIZE = 4
# Thresholds to apply to binarize segmentations from the output of the 2D CNN. These thresholds were obtained by
# minimizing the standard deviation of cross-sectional area across contrasts. For more details, see:
# https://github.com/sct-pipeline/deepseg-threshold
THR_DEEPSEG = {'t1': 0.15, 't2': 0.7, 't2s': 0.89, 'dwi': 0.01}

logger = logging.getLogger(__name__)


def find_centerline(algo, image_fname, contrast_type, brain_bool, folder_output, remove_temp_files, centerline_fname):
    """
    Assumes RPI orientation
    :param algo:
    :param image_fname:
    :param contrast_type:
    :param brain_bool:
    :param folder_output:
    :param remove_temp_files:
    :param centerline_fname:
    :return:
    """

    im = Image(image_fname)
    ctl_absolute_path = sct.add_suffix(im.absolutepath, "_ctr")

    # isct_spine_detect requires nz > 1
    if im.dim[2] == 1:
        im = concat_data([im, im], dim=2)
        im.hdr['dim'][3] = 2  # Needs to be change manually since dim not updated during concat_data
        bool_2d = True
    else:
        bool_2d = False

    # TODO: maybe change 'svm' for 'optic', because this is how we call it in sct_get_centerline
    if algo == 'svm':
        # run optic on a heatmap computed by a trained SVM+HoG algorithm
        # optic_models_fname = os.path.join(path_sct, 'data', 'optic_models', '{}_model'.format(contrast_type))
        # # TODO: replace with get_centerline(method=optic)
        im_ctl, _, _, _ = get_centerline(im,
                                        ParamCenterline(algo_fitting='optic', contrast=contrast_type))

    elif algo == 'cnn':
        # CNN parameters
        dct_patch_ctr = {'t2': {'size': (80, 80), 'mean': 51.1417, 'std': 57.4408},
                         't2s': {'size': (80, 80), 'mean': 68.8591, 'std': 71.4659},
                         't1': {'size': (80, 80), 'mean': 55.7359, 'std': 64.3149},
                         'dwi': {'size': (80, 80), 'mean': 55.744, 'std': 45.003}}
        dct_params_ctr = {'t2': {'features': 16, 'dilation_layers': 2},
                          't2s': {'features': 8, 'dilation_layers': 3},
                          't1': {'features': 24, 'dilation_layers': 3},
                          'dwi': {'features': 8, 'dilation_layers': 2}}

        # load model
        ctr_model_fname = os.path.join(sct.__sct_dir__, 'data', 'deepseg_sc_models', '{}_ctr.h5'.format(contrast_type))
        ctr_model = nn_architecture_ctr(height=dct_patch_ctr[contrast_type]['size'][0],
                                        width=dct_patch_ctr[contrast_type]['size'][1],
                                        channels=1,
                                        classes=1,
                                        features=dct_params_ctr[contrast_type]['features'],
                                        depth=2,
                                        temperature=1.0,
                                        padding='same',
                                        batchnorm=True,
                                        dropout=0.0,
                                        dilation_layers=dct_params_ctr[contrast_type]['dilation_layers'])
        ctr_model.load_weights(ctr_model_fname)

        # compute the heatmap
        im_heatmap, z_max = heatmap(im=im,
                                    model=ctr_model,
                                    patch_shape=dct_patch_ctr[contrast_type]['size'],
                                    mean_train=dct_patch_ctr[contrast_type]['mean'],
                                    std_train=dct_patch_ctr[contrast_type]['std'],
                                    brain_bool=brain_bool)
        im_ctl, _, _, _ = get_centerline(im_heatmap,
                                        ParamCenterline(algo_fitting='optic', contrast=contrast_type))

        if z_max is not None:
            sct.printv('Cropping brain section.')
            im_ctl.data[:, :, z_max:] = 0

    elif algo == 'viewer':
        im_labels = _call_viewer_centerline(im)
        im_ctl, _, _, _ = get_centerline(im_labels, param=ParamCenterline())

    elif algo == 'file':
        im_ctl = Image(centerline_fname)
        im_ctl.change_orientation('RPI')

    else:
        logger.error('The parameter "-centerline" is incorrect. Please try again.')
        sys.exit(1)

    # TODO: for some reason, when algo == 'file', the absolutepath is changed to None out of the method find_centerline
    im_ctl.absolutepath = ctl_absolute_path

    if bool_2d:
        im_ctl = split_data(im_ctl, dim=2)[0]

    if algo != 'viewer':
        im_labels = None

    # TODO: remove unecessary return params
    return "dummy_file_name", im_ctl, im_labels


def scale_intensity(data, out_min=0, out_max=255):
    """Scale intensity of data in a range defined by [out_min, out_max], based on the 2nd and 98th percentiles."""
    p2, p98 = np.percentile(data, (2, 98))
    return rescale_intensity(data, in_range=(p2, p98), out_range=(out_min, out_max))


def apply_intensity_normalization(im_in, params=None):
    """Standardize the intensity range."""
    img_normalized = im_in.change_type(np.float32)
    img_normalized.data = scale_intensity(img_normalized.data)
    return img_normalized


def _find_crop_start_end(coord_ctr, crop_size, im_dim):
    """Util function to find the coordinates to crop the image around the centerline (coord_ctr)."""
    half_size = crop_size // 2
    coord_start, coord_end = int(coord_ctr) - half_size + 1, int(coord_ctr) + half_size + 1

    if coord_end > im_dim:
        coord_end = im_dim
        coord_start = im_dim - crop_size if im_dim >= crop_size else 0
    if coord_start < 0:
        coord_start = 0
        coord_end = crop_size if im_dim >= crop_size else im_dim

    return coord_start, coord_end


def crop_image_around_centerline(im_in, ctr_in, crop_size):
    """Crop the input image around the input centerline file."""
    data_ctr = ctr_in.data
    data_ctr = data_ctr if len(data_ctr.shape) >= 3 else np.expand_dims(data_ctr, 2)
    data_in = im_in.data.astype(np.float32)
    im_new = empty_like(im_in)  # but in fact we're going to crop it

    x_lst, y_lst, z_lst = [], [], []
    data_im_new = np.zeros((crop_size, crop_size, im_in.dim[2]))
    for zz in range(im_in.dim[2]):
        if np.any(np.array(data_ctr[:, :, zz])):
            x_ctr, y_ctr = center_of_mass(np.array(data_ctr[:, :, zz]))

            x_start, x_end = _find_crop_start_end(x_ctr, crop_size, im_in.dim[0])
            y_start, y_end = _find_crop_start_end(y_ctr, crop_size, im_in.dim[1])

            crop_im = np.zeros((crop_size, crop_size))
            x_shape, y_shape = data_in[x_start:x_end, y_start:y_end, zz].shape
            crop_im[:x_shape, :y_shape] = data_in[x_start:x_end, y_start:y_end, zz]

            data_im_new[:, :, zz] = crop_im

            x_lst.append(str(x_start))
            y_lst.append(str(y_start))
            z_lst.append(zz)

    im_new.data = data_im_new
    return x_lst, y_lst, z_lst, im_new


def scan_slice(z_slice, model, mean_train, std_train, coord_lst, patch_shape, z_out_dim):
    """Scan the entire axial slice to detect the centerline."""
    z_slice_out = np.zeros(z_out_dim)
    sum_lst = []
    # loop across all the non-overlapping blocks of a cross-sectional slice
    for idx, coord in enumerate(coord_lst):
        block = z_slice[coord[0]:coord[2], coord[1]:coord[3]]
        block_nn = np.expand_dims(np.expand_dims(block, 0), -1)
        block_nn_norm = _normalize_data(block_nn, mean_train, std_train)
        block_pred = model.predict(block_nn_norm, batch_size=BATCH_SIZE)

        if coord[2] > z_out_dim[0]:
            x_end = patch_shape[0] - (coord[2] - z_out_dim[0])
        else:
            x_end = patch_shape[0]
        if coord[3] > z_out_dim[1]:
            y_end = patch_shape[1] - (coord[3] - z_out_dim[1])
        else:
            y_end = patch_shape[1]

        z_slice_out[coord[0]:coord[2], coord[1]:coord[3]] = block_pred[0, :x_end, :y_end, 0]
        sum_lst.append(np.sum(block_pred[0, :x_end, :y_end, 0]))

    # Put first the coord of the patch were the centerline is likely located so that the search could be faster for the
    # next axial slices
    coord_lst.insert(0, coord_lst.pop(sum_lst.index(max(sum_lst))))

    # computation of the new center of mass
    if np.max(z_slice_out) > 0.5:
        z_slice_out_bin = z_slice_out > 0.5
        labeled_mask, numpatches = label(z_slice_out_bin)
        largest_cc_mask = (labeled_mask == (np.bincount(labeled_mask.flat)[1:].argmax() + 1))
        x_CoM, y_CoM = center_of_mass(largest_cc_mask)
        x_CoM, y_CoM = int(x_CoM), int(y_CoM)
    else:
        x_CoM, y_CoM = None, None

    return z_slice_out, x_CoM, y_CoM, coord_lst


def heatmap(im, model, patch_shape, mean_train, std_train, brain_bool=True):
    """Compute the heatmap with CNN_1 representing the SC localization."""
    data_im = im.data.astype(np.float32)
    im_out = change_type(im, "uint8")
    del im
    data = np.zeros(im_out.data.shape)

    x_shape, y_shape = data_im.shape[:2]
    x_shape_block, y_shape_block = np.ceil(x_shape * 1.0 / patch_shape[0]).astype(np.int), np.int(
        y_shape * 1.0 / patch_shape[1])
    x_pad = int(x_shape_block * patch_shape[0] - x_shape)
    if y_shape > patch_shape[1]:
        y_crop = y_shape - y_shape_block * patch_shape[1]
        # slightly crop the input data in the P-A direction so that data_im.shape[1] % patch_shape[1] == 0
        data_im = data_im[:, :y_shape - y_crop, :]
        # coordinates of the blocks to scan during the detection, in the cross-sectional plane
        coord_lst = [[x_dim * patch_shape[0], y_dim * patch_shape[1],
                      (x_dim + 1) * patch_shape[0], (y_dim + 1) * patch_shape[1]]
                     for y_dim in range(y_shape_block) for x_dim in range(x_shape_block)]
    else:
        data_im = np.pad(data_im, ((0, 0), (0, patch_shape[1] - y_shape), (0, 0)), 'constant')
        coord_lst = [[x_dim * patch_shape[0], 0, (x_dim + 1) * patch_shape[0], patch_shape[1]] for x_dim in
                     range(x_shape_block)]
    # pad the input data in the R-L direction
    data_im = np.pad(data_im, ((0, x_pad), (0, 0), (0, 0)), 'constant')
    # scale intensities between 0 and 255
    data_im = scale_intensity(data_im)

    x_CoM, y_CoM = None, None
    z_sc_notDetected_cmpt = 0
    for zz in range(data_im.shape[2]):
        # if SC was detected at zz-1, we will start doing the detection on the block centered around the previously
        # computed center of mass (CoM)
        if x_CoM is not None:
            z_sc_notDetected_cmpt = 0  # SC detected, cmpt set to zero
            x_0, x_1 = _find_crop_start_end(x_CoM, patch_shape[0], data_im.shape[0])
            y_0, y_1 = _find_crop_start_end(y_CoM, patch_shape[1], data_im.shape[1])
            block = data_im[x_0:x_1, y_0:y_1, zz]
            block_nn = np.expand_dims(np.expand_dims(block, 0), -1)
            block_nn_norm = _normalize_data(block_nn, mean_train, std_train)
            block_pred = model.predict(block_nn_norm, batch_size=BATCH_SIZE)

            # coordinates manipulation due to the above padding and cropping
            if x_1 > data.shape[0]:
                x_end = data.shape[0]
                x_1 = data.shape[0]
                x_0 = data.shape[0] - patch_shape[0] if data.shape[0] > patch_shape[0] else 0
            else:
                x_end = patch_shape[0]
            if y_1 > data.shape[1]:
                y_end = data.shape[1]
                y_1 = data.shape[1]
                y_0 = data.shape[1] - patch_shape[1] if data.shape[1] > patch_shape[1] else 0
            else:
                y_end = patch_shape[1]

            data[x_0:x_1, y_0:y_1, zz] = block_pred[0, :x_end, :y_end, 0]

            # computation of the new center of mass
            if np.max(data[:, :, zz]) > 0.5:
                z_slice_out_bin = data[:, :, zz] > 0.5  # if the SC was detection
                x_CoM, y_CoM = center_of_mass(z_slice_out_bin)
                x_CoM, y_CoM = int(x_CoM), int(y_CoM)
            else:
                x_CoM, y_CoM = None, None

        # if the SC was not detected at zz-1 or on the patch centered around CoM in slice zz, the entire cross-sectional
        # slice is scanned
        if x_CoM is None:
            z_slice, x_CoM, y_CoM, coord_lst = scan_slice(data_im[:, :, zz], model,
                                                          mean_train, std_train,
                                                          coord_lst, patch_shape, data.shape[:2])
            data[:, :, zz] = z_slice

            z_sc_notDetected_cmpt += 1
            # if the SC has not been detected on 10 consecutive z_slices, we stop the SC investigation
            if z_sc_notDetected_cmpt > 10 and brain_bool:
                sct.printv('Brain section detected.')
                break

        # distance transform to deal with the harsh edges of the prediction boundaries (Dice)
        data[:, :, zz][np.where(data[:, :, zz] < 0.5)] = 0
        data[:, :, zz] = distance_transform_edt(data[:, :, zz])

    if not np.any(data):
        logger.error(
            '\nSpinal cord was not detected using "-centerline cnn". Please try another "-centerline" method.\n')
        sys.exit(1)

    im_out.data = data

    # z_max is used to reject brain sections
    z_max = np.max(list(set(np.where(data)[2])))
    if z_max == data.shape[2] - 1:
        return im_out, None
    else:
        return im_out, z_max


def _normalize_data(data, mean, std):
    """Util function to normalized data based on learned mean and std."""
    data -= mean
    data /= std
    return data


def segment_2d(model_fname, contrast_type, input_size, im_in):
    """
    Segment data using 2D convolutions.
    :return: seg_crop.data: ndarray float32: Output prediction
    """
    seg_model = nn_architecture_seg(height=input_size[0],
                                    width=input_size[1],
                                    depth=2 if contrast_type != 't2' else 3,
                                    features=32,
                                    batchnorm=False,
                                    dropout=0.0)
    seg_model.load_weights(model_fname)

    seg_crop = zeros_like(im_in, dtype=np.float32)

    data_norm = im_in.data
    # TODO: use sct_progress_bar
    for zz in range(im_in.dim[2]):
        # 2D CNN prediction
        pred_seg = seg_model.predict(np.expand_dims(np.expand_dims(data_norm[:, :, zz], -1), 0),
                                     batch_size=BATCH_SIZE)[0, :, :, 0]
        seg_crop.data[:, :, zz] = pred_seg

    return seg_crop.data


def segment_3d(model_fname, contrast_type, im_in):
    """
    Perform segmentation with 3D convolutions.
    :return: seg_crop.data: ndarray float32: Output prediction
    """
    from spinalcordtoolbox.deepseg_sc.cnn_models_3d import load_trained_model
    dct_patch_sc_3d = {'t2': {'size': (64, 64, 48), 'mean': 65.8562, 'std': 59.7999},
                       't2s': {'size': (96, 96, 48), 'mean': 87.0212, 'std': 64.425},
                       't1': {'size': (64, 64, 48), 'mean': 88.5001, 'std': 66.275}}
    # load 3d model
    seg_model = load_trained_model(model_fname)

    out = zeros_like(im_in, dtype=np.float32)

    # segment the spinal cord
    z_patch_size = dct_patch_sc_3d[contrast_type]['size'][2]
    z_step_keep = list(range(0, im_in.data.shape[2], z_patch_size))
    # TODO: use sct_progress_bar
    for zz in z_step_keep:
        if zz == z_step_keep[-1]:  # deal with instances where the im.data.shape[2] % patch_size_z != 0
            patch_im = np.zeros(dct_patch_sc_3d[contrast_type]['size'])
            z_patch_extracted = im_in.data.shape[2] - zz
            patch_im[:, :, :z_patch_extracted] = im_in.data[:, :, zz:]
        else:
            z_patch_extracted = z_patch_size
            patch_im = im_in.data[:, :, zz:z_patch_size + zz]

        if np.any(patch_im):  # Check if the patch is (not) empty, which could occur after a brain detection.
            patch_norm = \
                _normalize_data(patch_im, dct_patch_sc_3d[contrast_type]['mean'], dct_patch_sc_3d[contrast_type]['std'])
            patch_pred_proba = \
                seg_model.predict(np.expand_dims(np.expand_dims(patch_norm, 0), 0), batch_size=BATCH_SIZE)
            # pred_seg_th = (patch_pred_proba > 0.5).astype(int)[0, 0, :, :, :]
            pred_seg_th = patch_pred_proba[0, 0, :, :, :]  # TODO: clarified variable (this is not thresholded!)

            # TODO: add comment about what the code is doing below
            if zz == z_step_keep[-1]:
                out.data[:, :, zz:] = pred_seg_th[:, :, :z_patch_extracted]
            else:
                out.data[:, :, zz:z_patch_size + zz] = pred_seg_th

    return out.data


def uncrop_image(ref_in, data_crop, x_crop_lst, y_crop_lst, z_crop_lst):
    """
    Reconstruct the data from the cropped segmentation.
    """
    seg_unCrop = zeros_like(ref_in, dtype=np.float32)
    crop_size_x, crop_size_y = data_crop.shape[:2]
    for i_z, zz in enumerate(z_crop_lst):
        pred_seg = data_crop[:, :, zz]
        x_start, y_start = int(x_crop_lst[i_z]), int(y_crop_lst[i_z])
        x_end = x_start + crop_size_x if x_start + crop_size_x < seg_unCrop.dim[0] else seg_unCrop.dim[0]
        y_end = y_start + crop_size_y if y_start + crop_size_y < seg_unCrop.dim[1] else seg_unCrop.dim[1]
        seg_unCrop.data[x_start:x_end, y_start:y_end, zz] = pred_seg[0:x_end - x_start, 0:y_end - y_start]
    return seg_unCrop


def deep_segmentation_spinalcord(im_image, contrast_type, ctr_algo='cnn', ctr_file=None, brain_bool=True,
                                 kernel_size='2d', threshold_seg=None, remove_temp_files=1, verbose=1):
    """
    Main pipeline for CNN-based segmentation of the spinal cord.
    :param im_image:
    :param contrast_type: {'t1', 't2', t2s', 'dwi'}
    :param ctr_algo:
    :param ctr_file:
    :param brain_bool:
    :param kernel_size:
    :param threshold_seg: Binarization threshold (between 0 and 1) to apply to the segmentation prediction. Set to -1
        for no binarization (i.e. soft segmentation output)
    :param remove_temp_files:
    :param verbose:
    :return:
    """
    if threshold_seg is None:
        threshold_seg = THR_DEEPSEG[contrast_type]

    # Display stuff
    logger.info("Config deepseg_sc:")
    logger.info("  Centerline algorithm: {}".format(ctr_algo))
    logger.info("  Brain in image: {}".format(brain_bool))
    logger.info("  Kernel dimension: {}".format(kernel_size))
    logger.info("  Contrast: {}".format(contrast_type))
    logger.info("  Threshold: {}".format(threshold_seg))

    # create temporary folder with intermediate results
    tmp_folder = sct.TempFolder(verbose=verbose)
    tmp_folder_path = tmp_folder.get_path()
    if ctr_algo == 'file':  # if the ctr_file is provided
        tmp_folder.copy_from(ctr_file)
        file_ctr = os.path.basename(ctr_file)
    else:
        file_ctr = None
    tmp_folder.chdir()

    # re-orient image to RPI
    logger.info("Reorient the image to RPI, if necessary...")
    original_orientation = im_image.orientation
    # fname_orient = 'image_in_RPI.nii'
    im_image.change_orientation('RPI')

    # Resample image to 0.5mm in plane
    im_image_res = \
        resampling.resample_nib(im_image, new_size=[0.5, 0.5, im_image.dim[6]], new_size_type='mm', interpolation='linear')

    fname_orient = 'image_in_RPI_res.nii'
    im_image_res.save(fname_orient)

    # find the spinal cord centerline - execute OptiC binary
    logger.info("Finding the spinal cord centerline...")
    _, im_ctl, im_labels_viewer = find_centerline(algo=ctr_algo,
                                                    image_fname=fname_orient,
                                                    contrast_type=contrast_type,
                                                    brain_bool=brain_bool,
                                                    folder_output=tmp_folder_path,
                                                    remove_temp_files=remove_temp_files,
                                                    centerline_fname=file_ctr)

    if ctr_algo == 'file':
        im_ctl = \
            resampling.resample_nib(im_ctl, new_size=[0.5, 0.5, im_image.dim[6]], new_size_type='mm', interpolation='linear')

    # crop image around the spinal cord centerline
    logger.info("Cropping the image around the spinal cord...")
    crop_size = 96 if (kernel_size == '3d' and contrast_type == 't2s') else 64
    X_CROP_LST, Y_CROP_LST, Z_CROP_LST, im_crop_nii = crop_image_around_centerline(im_in=im_image_res,
                                                                                   ctr_in=im_ctl,
                                                                                   crop_size=crop_size)

    # normalize the intensity of the images
    logger.info("Normalizing the intensity...")
    im_norm_in = apply_intensity_normalization(im_in=im_crop_nii)
    del im_crop_nii

    if kernel_size == '2d':
        # segment data using 2D convolutions
        logger.info("Segmenting the spinal cord using deep learning on 2D patches...")
        segmentation_model_fname = \
            os.path.join(sct.__sct_dir__, 'data', 'deepseg_sc_models', '{}_sc.h5'.format(contrast_type))
        seg_crop = segment_2d(model_fname=segmentation_model_fname,
                              contrast_type=contrast_type,
                              input_size=(crop_size, crop_size),
                              im_in=im_norm_in)
    elif kernel_size == '3d':
        # segment data using 3D convolutions
        logger.info("Segmenting the spinal cord using deep learning on 3D patches...")
        segmentation_model_fname = \
            os.path.join(sct.__sct_dir__, 'data', 'deepseg_sc_models', '{}_sc_3D.h5'.format(contrast_type))
        seg_crop = segment_3d(model_fname=segmentation_model_fname,
                              contrast_type=contrast_type,
                              im_in=im_norm_in)

    # Postprocessing
    seg_crop_postproc = np.zeros_like(seg_crop)
    x_cOm, y_cOm = None, None
    for zz in range(im_norm_in.dim[2]):
        # Fill holes (only for binary segmentations)
        if threshold_seg >= 0:
            pred_seg_th = fill_holes_2d((seg_crop[:, :, zz] > threshold_seg).astype(int))
            pred_seg_pp = keep_largest_object(pred_seg_th, x_cOm, y_cOm)
            # Update center of mass for slice i+1
            if 1 in pred_seg_pp:
                x_cOm, y_cOm = center_of_mass(pred_seg_pp)
                x_cOm, y_cOm = np.round(x_cOm), np.round(y_cOm)
        else:
            # If soft segmentation, do nothing
            pred_seg_pp = seg_crop[:, :, zz]

        seg_crop_postproc[:, :, zz] = pred_seg_pp  # dtype is float32

    # reconstruct the segmentation from the crop data
    logger.info("Reassembling the image...")
    im_seg = uncrop_image(ref_in=im_image_res,
                          data_crop=seg_crop_postproc,
                          x_crop_lst=X_CROP_LST,
                          y_crop_lst=Y_CROP_LST,
                          z_crop_lst=Z_CROP_LST)
    # seg_uncrop_nii.save(sct.add_suffix(fname_res, '_seg'))  # for debugging
    del seg_crop, seg_crop_postproc, im_norm_in

    # resample to initial resolution
    logger.info("Resampling the segmentation to the native image resolution using linear interpolation...")
    im_seg_r = resampling.resample_nib(im_seg, image_dest=im_image, interpolation='linear')

    if ctr_algo == 'viewer':  # for debugging
        im_labels_viewer.save(sct.add_suffix(fname_orient, '_labels-viewer'))

    # Binarize the resampled image (except for soft segmentation, defined by threshold_seg=-1)
    if threshold_seg >= 0:
        logger.info("Binarizing the resampled segmentation...")
        im_seg_r.data = (im_seg_r.data > 0.5).astype(np.uint8)

    # post processing step to z_regularized
    im_seg_r_postproc = post_processing_volume_wise(im_seg_r)

    # Change data type. By default, dtype is float32
    if threshold_seg >= 0:
        im_seg_r_postproc.change_type(np.uint8)

    tmp_folder.chdir_undo()

    # remove temporary files
    if remove_temp_files:
        logger.info("Remove temporary files...")
        tmp_folder.cleanup()

    # reorient to initial orientation
    im_seg_r_postproc.change_orientation(original_orientation)

    # copy q/sform from input image to output segmentation
    im_seg.copy_qform_from_ref(im_image)

    return im_seg_r_postproc, im_image_res, im_seg.change_orientation('RPI')
