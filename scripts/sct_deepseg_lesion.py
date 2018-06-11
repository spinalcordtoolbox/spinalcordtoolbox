#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Function to segment the multiple sclerosis lesions using convolutional neural networks
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Charley Gros
# Modified: 2018-06-11
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys
import numpy as np
from scipy.ndimage.measurements import center_of_mass, label
from scipy.ndimage import distance_transform_edt
from skimage.exposure import rescale_intensity
from scipy.interpolate.interpolate import interp1d

from spinalcordtoolbox.centerline import optic
import sct_utils as sct
from msct_image import Image
from msct_parser import Parser
from sct_image import set_orientation

import spinalcordtoolbox.resample.nipy_resample
from spinalcordtoolbox.deepseg_sc.cnn_models import nn_architecture_ctr


def get_parser():
    """Initialize the parser."""
    parser = Parser(__file__)
    parser.usage.set_description("""MS lesion Segmentation using convolutional networks. \n\nReference: C Gros, B De Leener, et al. Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional neural networks (2018). arxiv.org/abs/1805.06349""")

    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t1.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast.",
                      mandatory=True,
                      example=['t2', 't2s'])
    parser.add_option(name="-centerline",
                      type_value="multiple_choice",
                      description="choice of spinal cord centerline algorithm.",
                      mandatory=False,
                      example=['svm', 'cnn'],
                      default_value="svm")
    parser.add_option(name="-brain",
                      type_value="multiple_choice",
                      description="indicate if the input image is expected to contain brain sections: 1: contains brain section, 0: no brain section. To indicate this parameter could speed the segmentation process.",
                      mandatory=False,
                      example=["0", "1"],
                      default_value="1")
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
    parser.add_option(name='-igt',
                      type_value='image_nifti',
                      description='File name of ground-truth segmentation.',
                      mandatory=False)
    return parser


def scale_intensity(data, out_min=0, out_max=255):
    """Scale intensity of data in a range defined by [out_min, out_max], based on the 2nd and 98th percentiles."""
    p2, p98 = np.percentile(data, (2, 98))
    return rescale_intensity(data, in_range=(p2, p98), out_range=(out_min, out_max))


def apply_intensity_normalization_model(img, landmarks_lst):
    """Description: apply the learned intensity landmarks to the input image."""
    percent_decile_lst = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    vals = list(img)
    landmarks_lst_cur = np.percentile(vals, q=percent_decile_lst)

    # create linear mapping models for the percentile segments to the learned standard intensity space
    linear_mapping = interp1d(landmarks_lst_cur, landmarks_lst, bounds_error=False)

    # transform the input image intensity values
    output = linear_mapping(img)

    # treat image intensity values outside of the cut-off percentiles range separately
    below_mapping = exp_model(landmarks_lst_cur[:2], landmarks_lst[:2], landmarks_lst[0])
    output[img < landmarks_lst_cur[0]] = below_mapping(img[img < landmarks_lst_cur[0]])

    above_mapping = exp_model(landmarks_lst_cur[-3:-1], landmarks_lst[-3:-1], landmarks_lst[-1])
    output[img > landmarks_lst_cur[-1]] = above_mapping(img[img > landmarks_lst_cur[-1]])

    return output.astype(np.float32)


def exp_model((x1, x2), (y1, y2), s2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)
    mu90 = x2

    # y2 = alpha + beta * exp(gamma * x)
    alpha = s2

    omega = m * mu90 - s2 + b
    beta = omega * np.exp(-m * mu90 * 1.0 / omega)

    gamma = m * 1.0 / omega

    return lambda x: alpha + beta * np.exp(gamma * x)


def apply_intensity_normalization(img_path, fname_out, contrast):
    """Standardize the intensity range."""
    img = Image(img_path)
    img_normalized = img.copy()
    data2norm = img.data.astype(np.float32)

    dct_norm = {'t2': [0.000000, 136.832187, 312.158435, 448.968030, 568.657779, 696.671586, 859.221138, 1074.463414, 1373.289174, 1811.522669, 2611.000000],
                't2s': [0.000000, 123.246969, 226.422561, 338.361023, 532.341924, 788.693675, 1096.975553, 1407.979466, 1716.524530, 2079.788451, 2611.000000]}

    img_normalized.data = apply_intensity_normalization_model(data2norm, dct_norm[contrast])

    img_normalized.changeType('float32')
    img_normalized.setFileName(fname_out)
    img_normalized.save()


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
    im_in, data_ctr = Image(filename_in), Image(filename_ctr).data.astype(np.int8)
    data_in = im_in.data.astype(np.float32)

    # z_step_keep = range(0, len(range(data_in.shape[2])), crop_size)
    # z_data_crop_max = max(z_step_keep) + crop_size

    # im_data_crop = np.zeros((crop_size, crop_size, z_data_crop_max))
    im_data_crop = np.zeros((crop_size, crop_size, im_in.dim[2]))

    im_new = im_in.copy()
    # im_new.dim = tuple([crop_size, crop_size, z_data_crop_max] + list(im_in.dim[3:]))
    im_new.dim = tuple([crop_size, crop_size, im_in.dim[2]] + list(im_in.dim[3:]))

    x_lst, y_lst = [], []
    for zz in range(im_in.dim[2]):
        if 1 in np.array(data_ctr[:, :, zz]):
            x_ctr, y_ctr = center_of_mass(np.array(data_ctr[:, :, zz]))

            x_start, x_end = _find_crop_start_end(x_ctr, crop_size, im_in.dim[0])
            y_start, y_end = _find_crop_start_end(y_ctr, crop_size, im_in.dim[1])

            crop_im = np.zeros((crop_size, crop_size))
            x_shape, y_shape = data_in[x_start:x_end, y_start:y_end, zz].shape
            crop_im[:x_shape, :y_shape] = data_in[x_start:x_end, y_start:y_end, zz]

            im_data_crop[:, :, zz] = crop_im

            x_lst.append(str(x_start))
            y_lst.append(str(y_start))

    im_new.data = im_data_crop
    im_new.setFileName(filename_out)
    im_new.save()
    del im_in, im_new

    return x_lst, y_lst


def scan_slice(z_slice, model, mean_train, std_train, coord_lst, patch_shape, z_out_dim):
    """Scan the entire axial slice to detect the centerline."""
    z_slice_out = np.zeros(z_out_dim)
    sum_lst = []
    # loop across all the non-overlapping blocks of a cross-sectional slice
    for idx, coord in enumerate(coord_lst):
        block = z_slice[coord[0]:coord[2], coord[1]:coord[3]]
        block_nn = np.expand_dims(np.expand_dims(block, 0), -1)
        block_nn_norm = _normalize_data(block_nn, mean_train, std_train)
        block_pred = model.predict(block_nn_norm)

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
    im_out = im.copy()
    im_out.changeType('uint8')
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
            block_pred = model.predict(block_nn_norm)

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
                sct.printv('\nBrain section detected.')
                break

        # distance transform to deal with the harsh edges of the prediction boundaries (Dice)
        data[:, :, zz][np.where(data[:, :, zz] < 0.5)] = 0
        data[:, :, zz] = distance_transform_edt(data[:, :, zz])

    im_out.data = data
    im_out.setFileName(filename_out)
    im_out.save()
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
        sct.printv('\nCropping brain section.')
        ctr_nii = Image(fname_out)
        ctr_nii.data[:, :, z_max:] = 0
        ctr_nii.save()


def _normalize_data(data, mean, std):
    """Util function to normalized data based on learned mean and std."""
    data -= mean
    data /= std
    return data


def uncrop_image(fname_ref, fname_out, data_crop, x_crop_lst, y_crop_lst):
    """Reconstruc the data from the crop segmentation."""
    im = Image(fname_ref)
    seg_unCrop = im.copy()
    seg_unCrop.data *= 0
    seg_unCrop.changeType('uint8')

    crop_size_x, crop_size_y = data_crop.shape[:2]

    for zz in range(len(x_crop_lst)):
        pred_seg = data_crop[:, :, zz]
        x_start, y_start = int(x_crop_lst[zz]), int(y_crop_lst[zz])
        x_end = x_start + crop_size_x if x_start + crop_size_x < seg_unCrop.dim[0] else seg_unCrop.dim[0]
        y_end = y_start + crop_size_y if y_start + crop_size_y < seg_unCrop.dim[1] else seg_unCrop.dim[1]
        seg_unCrop.data[x_start:x_end, y_start:y_end, zz] = pred_seg[0:x_end - x_start, 0:y_end - y_start]

    seg_unCrop.setFileName(fname_out)
    seg_unCrop.save()


def segment_3d(model_fname, contrast_type, fname_in, fname_out):
    """Perform segmentation with 3D convolutions."""
    from spinalcordtoolbox.deepseg_sc.cnn_models_3d import load_trained_model
    dct_patch_3d = {'t2': {'size': (48, 48, 48), 'mean': 871.309, 'std': 557.916},
                    't2s': {'size': (48, 48, 48), 'mean': 1011.31, 'std': 678.985}}

    # load 3d model
    seg_model = load_trained_model(model_fname)

    im = Image(fname_in)
    out = im.copy()
    out.data *= 0
    out.changeType('uint8')

    # segment the spinal cord
    z_patch_size = dct_patch_3d[contrast_type]['size'][2]
    z_step_keep = range(0, im.data.shape[2], z_patch_size)
    for zz in z_step_keep:
        if zz == z_step_keep[-1]:  # deal with instances where the im.data.shape[2] % patch_size_z != 0
            patch_im = np.zeros(dct_patch_3d[contrast_type]['size'])
            z_patch_extracted = im.data.shape[2] - zz
            patch_im[:, :, :z_patch_extracted] = im.data[:, :, zz:]
        else:
            z_patch_extracted = z_patch_size
            patch_im = im.data[:, :, zz:z_patch_size + zz]

        if np.sum(patch_im):  # Check if the patch is (not) empty, which could occur after a brain detection.
            patch_norm = _normalize_data(patch_im, dct_patch_3d[contrast_type]['mean'], dct_patch_3d[contrast_type]['std'])
            patch_pred_proba = seg_model.predict(np.expand_dims(np.expand_dims(patch_norm, 0), 0))
            pred_seg_th = (patch_pred_proba > 0.5).astype(int)[0, 0, :, :, :]

            if zz == z_step_keep[-1]:
                out.data[:, :, zz:] = pred_seg_th[:, :, :z_patch_extracted]
            else:
                out.data[:, :, zz:z_patch_size + zz] = pred_seg_th

    out.setFileName(fname_out)
    out.save()
    del im, out


def deep_segmentation_MSlesion(fname_image, contrast_type, output_folder, ctr_algo='svm', brain_bool=True, remove_temp_files=1, verbose=1):
    """Pipeline."""
    path_script = os.path.dirname(__file__)
    path_sct = os.path.dirname(path_script)

    # create temporary folder with intermediate results
    sct.log.info("\nCreating temporary folder...")
    file_fname = os.path.basename(fname_image)
    tmp_folder = sct.TempFolder()
    tmp_folder_path = tmp_folder.get_path()
    fname_image_tmp = tmp_folder.copy_from(fname_image)
    tmp_folder.chdir()

    # orientation of the image, should be RPI
    sct.log.info("\nReorient the image to RPI, if necessary...")
    fname_orient = sct.add_suffix(file_fname, '_RPI')
    im_2orient = Image(file_fname)
    original_orientation = im_2orient.orientation
    if original_orientation != 'RPI':
        im_orient = set_orientation(im_2orient, 'RPI')
        im_orient.setFileName(fname_orient)
        im_orient.save()
    else:
        im_orient = im_2orient
        sct.copy(fname_image_tmp, fname_orient)

    # resampling RPI image
    sct.log.info("\nResample the image to 0.5 mm isotropic resolution...")
    fname_res = sct.add_suffix(fname_orient, '_resampled')
    im_2res = im_orient
    input_resolution = im_2res.dim[4:7]
    new_resolution = 'x'.join(['0.5', '0.5', str(input_resolution[2])])
    spinalcordtoolbox.resample.nipy_resample.resample_file(fname_orient, fname_res, new_resolution,
                                                           'mm', 'linear', verbose=0)

    # find the spinal cord centerline - execute OptiC binary
    sct.log.info("\nFinding the spinal cord centerline...")
    if ctr_algo == 'svm':
        # run optic on a heatmap computed by a trained SVM+HoG algorithm
        optic_models_fname = os.path.join(path_sct, 'data', 'optic_models', '{}_model'.format(contrast_type))
        _, centerline_filename = optic.detect_centerline(image_fname=fname_res,
                                                         contrast_type=contrast_type,
                                                         optic_models_path=optic_models_fname,
                                                         folder_output=tmp_folder_path,
                                                         remove_temp_files=remove_temp_files,
                                                         output_roi=False,
                                                         verbose=0)
    elif ctr_algo == 'cnn':
        # CNN parameters
        dct_patch_ctr = {'t2': {'size': (80, 80), 'mean': 51.1417, 'std': 57.4408},
                            't2s': {'size': (80, 80), 'mean': 68.8591, 'std': 71.4659}}
        dct_params_ctr = {'t2': {'features': 16, 'dilation_layers': 2},
                            't2s': {'features': 8, 'dilation_layers': 3}}

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
        fname_heatmap = sct.add_suffix(fname_res, "_heatmap")
        img_filename = ''.join(sct.extract_fname(fname_heatmap)[:2])
        fname_heatmap_nii = img_filename + '.nii'
        z_max = heatmap(filename_in=fname_res,
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

    # crop image around the spinal cord centerline
    sct.log.info("\nCropping the image around the spinal cord...")
    fname_crop = sct.add_suffix(fname_res, '_crop')
    crop_size = 48
    X_CROP_LST, Y_CROP_LST = crop_image_around_centerline(filename_in=fname_res,
                                                          filename_ctr=centerline_filename,
                                                          filename_out=fname_crop,
                                                          crop_size=crop_size)

    # normalize the intensity of the images
    sct.log.info("\nNormalizing the intensity...")
    fname_norm = sct.add_suffix(fname_crop, '_norm')
    apply_intensity_normalization(img_path=fname_crop, fname_out=fname_norm, contrast=contrast_type)

    # resample to 0.5mm isotropic
    fname_res3d = sct.add_suffix(fname_norm, '_resampled3d')
    spinalcordtoolbox.resample.nipy_resample.resample_file(fname_norm, fname_res3d, '0.5x0.5x0.5',
                                                               'mm', 'linear', verbose=0)

    # segment data using 3D convolutions
    sct.log.info("\nSegmenting the MS lesions using deep learning on 3D patches...")
    segmentation_model_fname = os.path.join(path_sct, 'data', 'deepseg_lesion_models', '{}_lesion.h5'.format(contrast_type))
    fname_seg_crop_res = sct.add_suffix(fname_res3d, '_lesionseg')
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
    sct.log.info("\nReassembling the image...")
    fname_seg_res_RPI = sct.add_suffix(file_fname, '_res_RPI_seg')
    uncrop_image(fname_ref=fname_res,
                fname_out=fname_seg_res_RPI,
                data_crop=seg_crop_data,
                x_crop_lst=X_CROP_LST,
                y_crop_lst=Y_CROP_LST)

    # resample to initial resolution
    sct.log.info("\nResampling the segmentation to the original image resolution...")
    fname_seg_RPI = sct.add_suffix(file_fname, '_RPI_lesionseg')
    initial_resolution = 'x'.join([str(input_resolution[0]), str(input_resolution[1]), str(input_resolution[2])])
    spinalcordtoolbox.resample.nipy_resample.resample_file(fname_seg_res_RPI, fname_seg_RPI, initial_resolution,
                                                           'mm', 'linear', verbose=0)

    # binarize the resampled image to remove interpolation effects
    sct.log.info("\nBinarizing the segmentation to avoid interpolation effects...")
    thr = '0.5'
    sct.run(['sct_maths', '-i', fname_seg_RPI, '-bin', thr, '-o', fname_seg_RPI], verbose=0)

    # reorient to initial orientation
    sct.log.info("\nReorienting the segmentation to the original image orientation...")
    fname_seg = sct.add_suffix(file_fname, '_lesionseg')
    if original_orientation != 'RPI':
        im_seg_orient = set_orientation(Image(fname_seg_RPI), original_orientation)
        im_seg_orient.setFileName(fname_seg)
        im_seg_orient.save()
    else:
        sct.copy(fname_seg_RPI, fname_seg)

    tmp_folder.chdir_undo()

    # copy image from temporary folder into output folder
    sct.copy(os.path.join(tmp_folder_path, fname_seg), output_folder)

    # remove temporary files
    if remove_temp_files:
        sct.log.info("\nRemove temporary files...")
        tmp_folder.cleanup()

    return os.path.join(output_folder, fname_seg)


def main():
    """Main function."""
    sct.init_sct()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    fname_image = arguments['-i']
    contrast_type = arguments['-c']

    ctr_algo = arguments["-centerline"]

    brain_bool = bool(int(arguments["-brain"]))
    if "-brain" not in args and contrast_type in ['t2s', 'dwi']:
        brain_bool = False

    if '-ofolder' not in args:
        output_folder = os.getcwd()
    else:
        output_folder = arguments["-ofolder"]

    remove_temp_files = int(arguments['-r'])

    verbose = arguments['-v']

    algo_config_stg = '\nMethod:'
    algo_config_stg += '\n\tCenterline algorithm: ' + ctr_algo
    algo_config_stg += '\n\tAssumes brain section included in the image: ' + str(brain_bool) + '\n'
    sct.printv(algo_config_stg)

    fname_seg = deep_segmentation_MSlesion(fname_image, contrast_type, output_folder,
                                            ctr_algo=ctr_algo, brain_bool=brain_bool,
                                            remove_temp_files=remove_temp_files, verbose=verbose)

    sct.display_viewer_syntax([fname_image, os.path.join(output_folder, fname_seg)], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == "__main__":
    main()
