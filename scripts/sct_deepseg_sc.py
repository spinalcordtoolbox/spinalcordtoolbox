#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Function to segment the spinal cord using convolutional neural networks
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener & Charley Gros
# Modified: 2018-06-05
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import division, absolute_import

import os
import sys
import numpy as np
from scipy.ndimage.measurements import center_of_mass, label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.exposure import rescale_intensity
from scipy.ndimage import distance_transform_edt

from spinalcordtoolbox.centerline import optic
import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from msct_parser import Parser
from sct_get_centerline import _call_viewer_centerline, _from_viewerLabels_to_centerline

import spinalcordtoolbox.resample.nipy_resample
from spinalcordtoolbox.deepseg_sc.cnn_models import nn_architecture_seg, nn_architecture_ctr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
BATCH_SIZE = 4


def get_parser():
    """Initialize the parser."""
    parser = Parser(__file__)
    parser.usage.set_description("""Spinal Cord Segmentation using convolutional networks. \n\nReference: C Gros, B De Leener, et al. Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional neural networks (2018). arxiv.org/abs/1805.06349""")

    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t1.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast.",
                      mandatory=True,
                      example=['t1', 't2', 't2s', 'dwi'])
    parser.add_option(name="-centerline",
                      type_value="multiple_choice",
                      description="Method used for extracting the centerline.\nsvm: automatic centerline detection, based on Support Vector Machine algorithm.\ncnn: automatic centerline detection, based on Convolutional Neural Network.\nviewer: semi-automatic centerline generation, based on manual selection of a few points using an interactive viewer, then approximation with NURBS.\nmanual: use an existing centerline by specifying its filename with flag -file_centerline (e.g. -file_centerline t2_centerline_manual.nii.gz).\n",
                      mandatory=False,
                      example=['svm', 'cnn', 'viewer', 'manual'],
                      default_value="svm")
    parser.add_option(name="-file_centerline",
                      type_value="image_nifti",
                      description="Input centerline file (to use with flag -centerline manual).",
                      mandatory=False,
                      example="t2_centerline_manual.nii.gz")
    parser.add_option(name="-brain",
                      type_value="multiple_choice",
                      description="indicate if the input image is expected to contain brain sections:\n1: contains brain section\n0: no brain section.\nTo indicate this parameter could speed the segmentation process. Note that this flag is only effective with -centerline cnn.",
                      mandatory=False,
                      example=["0", "1"])
    parser.add_option(name="-kernel",
                      type_value="multiple_choice",
                      description="choice of 2D or 3D kernels for the segmentation. Note that segmentation with 3D kernels is significantely longer than with 2D kernels.",
                      mandatory=False,
                      example=['2d', '3d'],
                      default_value="2d")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="output folder.",
                      mandatory=False,
                      example="My_Output_Folder/",
                      default_value="")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="remove temporary files.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="1: display on, 0: display off (default)",
                      mandatory=False,
                      example=["0", "1"],
                      default_value="1")
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=None)
    parser.add_option(name='-igt',
                      type_value='image_nifti',
                      description='File name of ground-truth segmentation.',
                      mandatory=False)
    return parser


def scale_intensity(data, out_min=0, out_max=255):
    """Scale intensity of data in a range defined by [out_min, out_max], based on the 2nd and 98th percentiles."""
    p2, p98 = np.percentile(data, (2, 98))
    return rescale_intensity(data, in_range=(p2, p98), out_range=(out_min, out_max))


def apply_intensity_normalization(img_path, fname_out, params=None):
    """Standardize the intensity range."""
    img = Image(img_path)

    img_normalized = img.change_type(np.float32)
    img_normalized.data = scale_intensity(img_normalized.data)
    img_normalized.save(fname_out)


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


def crop_image_around_centerline(filename_in, filename_ctr, filename_out, crop_size):
    """Crop the input image around the input centerline file."""
    im_in, data_ctr = Image(filename_in), Image(filename_ctr).data
    data_ctr = data_ctr if len(data_ctr.shape) >= 3 else np.expand_dims(data_ctr, 2)
    data_in = im_in.data.astype(np.float32)
    im_new = msct_image.empty_like(im_in) # but in fact we're going to crop it

    x_lst, y_lst = [], []
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

    im_new.data = data_im_new
    im_new.save(filename_out)

    return x_lst, y_lst


def _remove_extrem_holes(z_lst, end_z, start_z=0):
    """Remove extrem holes from the list of holes so that we will not interpolate on the extrem slices."""
    if start_z in z_lst:
        while start_z in z_lst:
            z_lst = z_lst[1:]
            start_z += 1
        if len(z_lst):
            z_lst.pop(0)

    if end_z in z_lst:
        while end_z in z_lst:
            z_lst = z_lst[:-1]
            end_z -= 1

    return z_lst


def _list2range(lst):
    tmplst = lst[:]
    tmplst.sort()
    start = tmplst[0]

    currentrange = [start, start + 1]

    for item in tmplst[1:]:
        if currentrange[1] == item:  # contiguous
            currentrange[1] += 1
        else:  # new range start
            yield list(currentrange)
            currentrange = [item, item + 1]

    yield list(currentrange)  # last range


def _fill_z_holes(zz_lst, data, z_spaccing):
    data_interpol = np.copy(data)

    for z_hole_start, z_hole_end in list(_list2range(zz_lst)):
        z_ref_start, z_ref_end = z_hole_start - 1, z_hole_end
        slice_ref_start, slice_ref_end = data[:, :, z_ref_start], data[:, :, z_ref_end]

        hole_cur_lst = list(range(z_hole_start, z_hole_end))
        lenght_hole = len(hole_cur_lst) + 1
        phys_lenght_hole = lenght_hole * z_spaccing

        denom_interpolation = (lenght_hole + 1)

        if phys_lenght_hole < 10:
            sct.log.warning('Filling a hole in the segmentation around z_slice #:' + str(z_ref_start))

            for idx_z, z_hole_cur in enumerate(hole_cur_lst):
                num_interpolation = (lenght_hole - idx_z - 1) * slice_ref_start  # Contribution of the bottom ref slice
                num_interpolation += (idx_z + 1) * slice_ref_end  # Contribution of the top ref slice

                slice_interpolation = num_interpolation * 1. / denom_interpolation
                slice_interpolation = (slice_interpolation > 0).astype(np.int)

                data_interpol[:, :, z_hole_cur] = slice_interpolation

    return data_interpol


def _remove_blobs(data):
    """Remove false positive blobs, likely occuring in brain sections."""
    labeled_obj, num_obj = label(data)
    if num_obj > 1:  # If there is more than one connected object
        bigger_obj = (labeled_obj == (np.bincount(labeled_obj.flat)[1:].argmax() + 1))

        data2clean = np.copy(data)

        # remove blobs only above the bigger connected object
        z_max = np.max(np.where(bigger_obj)[2])
        data2clean[:, :, :z_max + 1] = 0

        labeled_obj2clean, num_obj2clean = label(data2clean)
        if num_obj2clean:  # If there is connected object above the biffer connected one
            for obj_id in range(1, num_obj2clean + 1):
                # if the blob has a volume < 10% of the bigger connected object, then remove it
                if np.sum(labeled_obj2clean == obj_id) < 0.1 * np.sum(bigger_obj):
                    sct.printv('Removing small objects above slice#' + str(z_max))
                    data[np.where(labeled_obj2clean == obj_id)] = 0

    return data


def post_processing_volume_wise(fname_in):
    """Post processing function."""
    im_in = Image(fname_in)
    data_in = im_in.data.astype(np.int)

    data_in = _remove_blobs(data_in)

    zz_zeros = [zz for zz in range(im_in.dim[2]) if 1 not in list(np.unique(data_in[:, :, zz]))]
    zz_holes = _remove_extrem_holes(zz_zeros, im_in.dim[2] - 1, 0)
    # filling z_holes, i.e. interpolate for z_slice not segmented
    im_in.data = _fill_z_holes(zz_holes, data_in, im_in.dim[6]) if len(zz_holes) else data_in
    im_in.save(fname_in)


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

    # Put first the coord of the patch were the centerline is likely located so that the search could be faster for the next axial slices
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


def heatmap(filename_in, filename_out, model, patch_shape, mean_train, std_train, brain_bool=True):
    """Compute the heatmap with CNN_1 representing the SC localization."""
    im = Image(filename_in)
    data_im = im.data.astype(np.float32)
    im_out = msct_image.change_type(im, "uint8")
    del im
    data = np.zeros(im_out.data.shape)

    x_shape, y_shape = data_im.shape[:2]
    x_shape_block, y_shape_block = np.ceil(x_shape * 1.0 / patch_shape[0]).astype(np.int), np.int(y_shape * 1.0 / patch_shape[1])
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
        coord_lst = [[x_dim * patch_shape[0], 0, (x_dim + 1) * patch_shape[0], patch_shape[1]] for x_dim in range(x_shape_block)]
    # pad the input data in the R-L direction
    data_im = np.pad(data_im, ((0, x_pad), (0, 0), (0, 0)), 'constant')
    # scale intensities between 0 and 255
    data_im = scale_intensity(data_im)

    x_CoM, y_CoM = None, None
    z_sc_notDetected_cmpt = 0
    for zz in range(data_im.shape[2]):
        # if SC was detected at zz-1, we will start doing the detection on the block centered around the previously conputed center of mass (CoM)
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

        # if the SC was not detected at zz-1 or on the patch centered around CoM in slice zz, the entire cross-sectional slice is scaned
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

    im_out.data = data
    im_out.save(filename_out)
    del im_out

    # z_max is used to reject brain sections
    z_max = np.max(list(set(np.where(data)[2])))
    if z_max == data.shape[2] - 1:
        return None
    else:
        return z_max


def heatmap2optic(fname_heatmap, lambda_value, fname_out, z_max, algo='dpdt'):
    """Run OptiC on the heatmap computed by CNN_1."""
    import nibabel as nib
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_PAIR"

    optic_input = fname_heatmap.split('.nii')[0]

    cmd_optic = 'isct_spine_detect -ctype="%s" -lambda="%s" "%s" "%s" "%s"' % \
                (algo, str(lambda_value), "NONE", optic_input, optic_input)
    sct.run(cmd_optic, verbose=1)

    optic_hdr_filename = optic_input + '_ctr.hdr'
    img = nib.load(optic_hdr_filename)
    nib.save(img, fname_out)

    # crop the centerline if z_max < data.shape[2] and -brain == 1
    if z_max is not None:
        sct.printv('Cropping brain section.')
        ctr_nii = Image(fname_out)
        ctr_nii.data[:, :, z_max:] = 0
        ctr_nii.save()


def find_centerline(algo, image_fname, path_sct, contrast_type, brain_bool, folder_output, remove_temp_files, centerline_fname):

    if Image(image_fname).dim[2] == 1:  # isct_spine_detect requires nz > 1
        from sct_image import concat_data
        im_concat = concat_data([image_fname, image_fname], dim=2)
        im_concat.save(sct.add_suffix(image_fname, '_concat'))
        image_fname = sct.add_suffix(image_fname, '_concat')
        bool_2d = True
    else:
        bool_2d = False

    if algo == 'svm':
        # run optic on a heatmap computed by a trained SVM+HoG algorithm
        optic_models_fname = os.path.join(path_sct, 'data', 'optic_models', '{}_model'.format(contrast_type))
        _, centerline_filename = optic.detect_centerline(image_fname=image_fname,
                                                         contrast_type=contrast_type,
                                                         optic_models_path=optic_models_fname,
                                                         folder_output=folder_output,
                                                         remove_temp_files=remove_temp_files,
                                                         output_roi=False,
                                                         verbose=0)
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
        ctr_model_fname = os.path.join(path_sct, 'data', 'deepseg_sc_models', '{}_ctr.h5'.format(contrast_type))
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
        fname_heatmap = sct.add_suffix(image_fname, "_heatmap")
        img_filename = ''.join(sct.extract_fname(fname_heatmap)[:2])
        fname_heatmap_nii = img_filename + '.nii'
        z_max = heatmap(filename_in=image_fname,
                        filename_out=fname_heatmap_nii,
                        model=ctr_model,
                        patch_shape=dct_patch_ctr[contrast_type]['size'],
                        mean_train=dct_patch_ctr[contrast_type]['mean'],
                        std_train=dct_patch_ctr[contrast_type]['std'],
                        brain_bool=brain_bool)

        # run optic on the heatmap
        centerline_filename = sct.add_suffix(fname_heatmap, "_ctr")
        heatmap2optic(fname_heatmap=fname_heatmap_nii,
                      lambda_value=7 if contrast_type == 't2s' else 1,
                      fname_out=centerline_filename,
                      z_max=z_max if brain_bool else None)

    elif algo == 'viewer':
        centerline_filename = sct.add_suffix(image_fname, "_ctr")
        fname_labels_viewer = _call_viewer_centerline(fname_in=image_fname)
        _from_viewerLabels_to_centerline(fname_labels=fname_labels_viewer, fname_out=centerline_filename)

    elif algo == 'manual':
        centerline_filename = sct.add_suffix(image_fname, "_ctr")
        image_manual_centerline = Image(centerline_fname)
        # Re-orient and Re-sample the manual centerline
        image_centerline_reoriented = msct_image.change_orientation(image_manual_centerline, 'RPI').save(centerline_filename)
        input_resolution = image_centerline_reoriented.dim[4:7]
        new_resolution = 'x'.join(['0.5', '0.5', str(input_resolution[2])])
        spinalcordtoolbox.resample.nipy_resample.resample_file(centerline_filename, centerline_filename, new_resolution,
                                                           'mm', 'linear', verbose=0)

    else:
        sct.log.error('The parameter "-centerline" is incorrect. Please try again.')
        sys.exit(1)

    if bool_2d:
        from sct_image import split_data
        im_split_lst = split_data(Image(centerline_filename), dim=2)
        im_split_lst[0].save(centerline_filename)

    return centerline_filename


def _normalize_data(data, mean, std):
    """Util function to normalized data based on learned mean and std."""
    data -= mean
    data /= std
    return data


def segment_2d(model_fname, contrast_type, input_size, fname_in, fname_out):
    """Segment data using 2D convolutions."""
    seg_model = nn_architecture_seg(height=input_size[0],
                                    width=input_size[1],
                                    depth=2 if contrast_type != 't2' else 3,
                                    features=32,
                                    batchnorm=False,
                                    dropout=0.0)
    seg_model.load_weights(model_fname)

    image_normalized = Image(fname_in)
    seg_crop = msct_image.zeros_like(image_normalized, dtype=np.uint8)

    data_norm = image_normalized.data
    x_cOm, y_cOm = None, None
    # for zz in list(reversed(range(image_normalized.dim[2]))):
    for zz in range(image_normalized.dim[2]):
        pred_seg = seg_model.predict(np.expand_dims(np.expand_dims(data_norm[:, :, zz], -1), 0), batch_size=BATCH_SIZE)[0, :, :, 0]
        pred_seg_th = (pred_seg > 0.5).astype(int)
        pred_seg_pp = post_processing_slice_wise(pred_seg_th, x_cOm, y_cOm)
        seg_crop.data[:, :, zz] = pred_seg_pp

        if 1 in pred_seg_pp:
            x_cOm, y_cOm = center_of_mass(pred_seg_pp)
            x_cOm, y_cOm = np.round(x_cOm), np.round(y_cOm)

    seg_crop.save(fname_out)

    return seg_crop.data


def uncrop_image(fname_ref, fname_out, data_crop, x_crop_lst, y_crop_lst):
    """Reconstruc the data from the crop segmentation."""
    im = Image(fname_ref)
    seg_unCrop = msct_image.zeros_like(im, dtype=np.uint8)

    crop_size_x, crop_size_y = data_crop.shape[:2]

    for zz in range(len(x_crop_lst)):
        pred_seg = data_crop[:, :, zz]
        x_start, y_start = int(x_crop_lst[zz]), int(y_crop_lst[zz])
        x_end = x_start + crop_size_x if x_start + crop_size_x < seg_unCrop.dim[0] else seg_unCrop.dim[0]
        y_end = y_start + crop_size_y if y_start + crop_size_y < seg_unCrop.dim[1] else seg_unCrop.dim[1]
        seg_unCrop.data[x_start:x_end, y_start:y_end, zz] = pred_seg[0:x_end - x_start, 0:y_end - y_start]

    seg_unCrop.save(fname_out)


def post_processing_slice_wise(z_slice, x_cOm, y_cOm):
    """Keep the largest connected obejct per z_slice and fill little holes."""
    labeled_obj, num_obj = label(z_slice)
    if num_obj > 1:
        if x_cOm is None:  # slice 0
            z_slice = (labeled_obj == (np.bincount(labeled_obj.flat)[1:].argmax() + 1))
        else:
            idx_z_minus_1 = np.bincount(labeled_obj.flat)[1:].argmax() + 1
            for idx in range(1, num_obj + 1):
                z_idx = labeled_obj == idx
                if z_idx[int(x_cOm), int(y_cOm)]:
                    idx_z_minus_1 = idx
            z_slice = (labeled_obj == idx_z_minus_1)

    return binary_fill_holes(z_slice, structure=np.ones((3, 3))).astype(np.int)


def segment_3d(model_fname, contrast_type, fname_in, fname_out):
    """Perform segmentation with 3D convolutions."""
    from spinalcordtoolbox.deepseg_sc.cnn_models_3d import load_trained_model
    dct_patch_sc_3d = {'t2': {'size': (64, 64, 48), 'mean': 65.8562, 'std': 59.7999},
                        't2s': {'size': (96, 96, 48), 'mean': 87.0212, 'std': 64.425},
                        't1': {'size': (64, 64, 48), 'mean': 88.5001, 'std': 66.275}}
    # load 3d model
    seg_model = load_trained_model(model_fname)

    im = Image(fname_in)
    out = msct_image.zeros_like(im, dtype=np.uint8)

    # segment the spinal cord
    z_patch_size = dct_patch_sc_3d[contrast_type]['size'][2]
    z_step_keep = list(range(0, im.data.shape[2], z_patch_size))
    for zz in z_step_keep:
        if zz == z_step_keep[-1]:  # deal with instances where the im.data.shape[2] % patch_size_z != 0
            patch_im = np.zeros(dct_patch_sc_3d[contrast_type]['size'])
            z_patch_extracted = im.data.shape[2] - zz
            patch_im[:, :, :z_patch_extracted] = im.data[:, :, zz:]
        else:
            z_patch_extracted = z_patch_size
            patch_im = im.data[:, :, zz:z_patch_size + zz]

        if np.any(patch_im):  # Check if the patch is (not) empty, which could occur after a brain detection.
            patch_norm = _normalize_data(patch_im, dct_patch_sc_3d[contrast_type]['mean'], dct_patch_sc_3d[contrast_type]['std'])
            patch_pred_proba = seg_model.predict(np.expand_dims(np.expand_dims(patch_norm, 0), 0), batch_size=BATCH_SIZE)
            pred_seg_th = (patch_pred_proba > 0.5).astype(int)[0, 0, :, :, :]

            x_cOm, y_cOm = None, None
            for zz_pp in range(z_patch_size):
                pred_seg_pp = post_processing_slice_wise(pred_seg_th[:, :, zz_pp], x_cOm, y_cOm)
                pred_seg_th[:, :, zz_pp] = pred_seg_pp
                x_cOm, y_cOm = center_of_mass(pred_seg_pp)
                x_cOm, y_cOm = np.round(x_cOm), np.round(y_cOm)

            if zz == z_step_keep[-1]:
                out.data[:, :, zz:] = pred_seg_th[:, :, :z_patch_extracted]
            else:
                out.data[:, :, zz:z_patch_size + zz] = pred_seg_th

    out.save(fname_out)


def deep_segmentation_spinalcord(fname_image, contrast_type, output_folder, ctr_algo='cnn', ctr_file=None, brain_bool=True, kernel_size='2d', remove_temp_files=1, verbose=1):
    """Pipeline."""
    path_script = os.path.dirname(__file__)
    path_sct = os.path.dirname(path_script)

    # create temporary folder with intermediate results
    sct.log.info("Creating temporary folder...")
    file_fname = os.path.basename(fname_image)
    tmp_folder = sct.TempFolder()
    tmp_folder_path = tmp_folder.get_path()
    fname_image_tmp = tmp_folder.copy_from(fname_image)
    if ctr_algo == 'manual':  # if the ctr_file is provided
        tmp_folder.copy_from(ctr_file)
        file_ctr = os.path.basename(ctr_file)
    else:
        file_ctr = None
    tmp_folder.chdir()

    # orientation of the image, should be RPI
    sct.log.info("Reorient the image to RPI, if necessary...")
    fname_orient = sct.add_suffix(file_fname, '_RPI')
    im_2orient = Image(file_fname)
    original_orientation = im_2orient.orientation
    if original_orientation != 'RPI':
        im_orient = msct_image.change_orientation(im_2orient, 'RPI').save(fname_orient)
    else:
        im_orient = im_2orient
        sct.copy(fname_image_tmp, fname_orient)

    # resampling RPI image
    sct.log.info("Resample the image to 0.5 mm isotropic resolution...")
    fname_res = sct.add_suffix(fname_orient, '_resampled')
    im_2res = im_orient
    input_resolution = im_2res.dim[4:7]
    new_resolution = 'x'.join(['0.5', '0.5', str(input_resolution[2])])
    spinalcordtoolbox.resample.nipy_resample.resample_file(fname_orient, fname_res, new_resolution,
                                                           'mm', 'linear', verbose=0)

    # find the spinal cord centerline - execute OptiC binary
    sct.log.info("Finding the spinal cord centerline...")
    centerline_filename = find_centerline(algo=ctr_algo,
                                          image_fname=fname_res,
                                          path_sct=path_sct,
                                          contrast_type=contrast_type,
                                          brain_bool=brain_bool,
                                          folder_output=tmp_folder_path,
                                          remove_temp_files=remove_temp_files,
                                          centerline_fname=file_ctr)
    
    # crop image around the spinal cord centerline
    sct.log.info("Cropping the image around the spinal cord...")
    fname_crop = sct.add_suffix(fname_res, '_crop')
    crop_size = 96 if (kernel_size == '3d' and contrast_type == 't2s') else 64
    X_CROP_LST, Y_CROP_LST = crop_image_around_centerline(filename_in=fname_res,
                                                          filename_ctr=centerline_filename,
                                                          filename_out=fname_crop,
                                                          crop_size=crop_size)

    # normalize the intensity of the images
    sct.log.info("Normalizing the intensity...")
    fname_norm = sct.add_suffix(fname_crop, '_norm')
    apply_intensity_normalization(img_path=fname_crop, fname_out=fname_norm)

    if kernel_size == '2d':
        # segment data using 2D convolutions
        sct.log.info("Segmenting the spinal cord using deep learning on 2D patches...")
        segmentation_model_fname = os.path.join(path_sct, 'data', 'deepseg_sc_models', '{}_sc.h5'.format(contrast_type))
        fname_seg_crop = sct.add_suffix(fname_norm, '_seg')
        seg_crop_data = segment_2d(model_fname=segmentation_model_fname,
                                contrast_type=contrast_type,
                                input_size=(crop_size, crop_size),
                                fname_in=fname_norm,
                                fname_out=fname_seg_crop)
    elif kernel_size == '3d':
        # resample to 0.5mm isotropic
        fname_res3d = sct.add_suffix(fname_norm, '_resampled3d')
        spinalcordtoolbox.resample.nipy_resample.resample_file(fname_norm, fname_res3d, '0.5x0.5x0.5',
                                                               'mm', 'linear', verbose=0)

        # segment data using 3D convolutions
        sct.log.info("Segmenting the spinal cord using deep learning on 3D patches...")
        segmentation_model_fname = os.path.join(path_sct, 'data', 'deepseg_sc_models', '{}_sc_3D.h5'.format(contrast_type))
        fname_seg_crop_res = sct.add_suffix(fname_res3d, '_seg')
        segment_3d(model_fname=segmentation_model_fname,
                    contrast_type=contrast_type,
                    fname_in=fname_res3d,
                    fname_out=fname_seg_crop_res)

        # resample to the initial pz resolution
        fname_seg_res2d = sct.add_suffix(fname_seg_crop_res, '_resampled2d')
        initial_2d_resolution = 'x'.join(['0.5', '0.5', str(input_resolution[2])])
        spinalcordtoolbox.resample.nipy_resample.resample_file(fname_seg_crop_res, fname_seg_res2d, initial_2d_resolution,
                                                               'mm', 'linear', verbose=0)
        seg_crop_data = Image(fname_seg_res2d).data

    # reconstruct the segmentation from the crop data
    sct.log.info("Reassembling the image...")
    fname_seg_res_RPI = sct.add_suffix(file_fname, '_res_RPI_seg')
    uncrop_image(fname_ref=fname_res,
                fname_out=fname_seg_res_RPI,
                data_crop=seg_crop_data,
                x_crop_lst=X_CROP_LST,
                y_crop_lst=Y_CROP_LST)

    # resample to initial resolution
    sct.log.info("Resampling the segmentation to the original image resolution...")
    fname_seg_RPI = sct.add_suffix(file_fname, '_RPI_seg')
    initial_resolution = 'x'.join([str(input_resolution[0]), str(input_resolution[1]), str(input_resolution[2])])
    spinalcordtoolbox.resample.nipy_resample.resample_file(fname_seg_res_RPI, fname_seg_RPI, initial_resolution,
                                                           'mm', 'linear', verbose=0)

    # binarize the resampled image to remove interpolation effects
    sct.log.info("Binarizing the segmentation to avoid interpolation effects...")
    thr = '0.0001' if contrast_type in ['t1', 'dwi'] else '0.5'
    sct.run(['sct_maths', '-i', fname_seg_RPI, '-bin', thr, '-o', fname_seg_RPI], verbose=0)

    # post processing step to z_regularized
    post_processing_volume_wise(fname_in=fname_seg_RPI)

    # reorient to initial orientation
    sct.log.info("Reorienting the segmentation to the original image orientation...")
    fname_seg = sct.add_suffix(file_fname, '_seg')
    if original_orientation != 'RPI':
        im_seg_orient = msct_image.change_orientation(Image(fname_seg_RPI), original_orientation).save(fname_seg)
    else:
        sct.copy(fname_seg_RPI, fname_seg)

    tmp_folder.chdir_undo()

    # copy image from temporary folder into output folder
    sct.copy(os.path.join(tmp_folder_path, fname_seg), output_folder)

    # remove temporary files
    if remove_temp_files:
        sct.log.info("Remove temporary files...")
        tmp_folder.cleanup()

    return os.path.join(output_folder, fname_seg)


def generate_qc(fn_in, fn_seg, args, path_qc):
    """Generate a QC entry allowing to quickly review the segmentation process."""
    import spinalcordtoolbox.reports.qc as qc
    import spinalcordtoolbox.reports.slice as qcslice

    qc.add_entry(
     src=fn_in,
     process="sct_deepseg_sc",
     args=args,
     path_qc=path_qc,
     plane='Axial',
     qcslice=qcslice.Axial([Image(fn_in), Image(fn_seg)]),
     qcslice_operations=[qc.QcImage.listed_seg],
     qcslice_layout=lambda x: x.mosaic(),
    )


def main():
    """Main function."""
    sct.init_sct()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    fname_image = arguments['-i']
    contrast_type = arguments['-c']

    ctr_algo = arguments["-centerline"]
    if "-centerline" not in args and contrast_type == 't1':
        ctr_algo = 'cnn'

    if "-brain" not in args:
        if contrast_type in ['t2s', 'dwi']:
            brain_bool = False
        if contrast_type in ['t1', 't2']:
            brain_bool = True
    else:
        brain_bool = bool(int(arguments["-brain"]))

    kernel_size = arguments["-kernel"]
    if kernel_size == '3d' and contrast_type == 'dwi':
        kernel_size = '2d'
        sct.printv('3D kernel model for dwi contrast is not available. 2D kernel model is used instead.', type="warning")

    if '-ofolder' not in args:
        output_folder = os.getcwd()
    else:
        output_folder = arguments["-ofolder"]

    if ctr_algo == 'manual' and "-file_centerline" not in args:
		sct.log.error('Please use the flag -file_centerline to indicate the centerline filename.')
		sys.exit(1)
    
    if "-file_centerline" in args:
        manual_centerline_fname = arguments["-file_centerline"]
        ctr_algo = 'manual'
    else:
        manual_centerline_fname = None

    remove_temp_files = int(arguments['-r'])

    verbose = arguments['-v']

    path_qc = arguments.get("-qc", None)

    algo_config_stg = '\nMethod:'
    algo_config_stg += '\n\tCenterline algorithm: ' + str(ctr_algo)
    algo_config_stg += '\n\tAssumes brain section included in the image: ' + str(brain_bool)
    algo_config_stg += '\n\tDimension of the segmentation kernel convolutions: ' + kernel_size + '\n'
    sct.printv(algo_config_stg)

    fname_seg = deep_segmentation_spinalcord(fname_image, contrast_type, output_folder,
                                            ctr_algo=ctr_algo, ctr_file=manual_centerline_fname,
                                            brain_bool=brain_bool, kernel_size=kernel_size,
                                            remove_temp_files=remove_temp_files, verbose=verbose)

    if path_qc is not None:
        generate_qc(fname_image, fname_seg, args, os.path.abspath(path_qc))

    sct.display_viewer_syntax([fname_image, os.path.join(output_folder, fname_seg)], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == "__main__":
    main()
