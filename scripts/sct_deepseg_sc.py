#!/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Function to segment the spinal cord using deep convolutional networks
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener & Charley Gros
# Modified: 2018-01-22
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os, sys

import numpy as np
from scipy.ndimage.measurements import center_of_mass, label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.exposure import rescale_intensity
from scipy.ndimage import distance_transform_edt

from spinalcordtoolbox.centerline import optic
import sct_utils as sct
from msct_image import Image
from msct_parser import Parser
from sct_image import set_orientation

import spinalcordtoolbox.resample.nipy_resample
from spinalcordtoolbox.deepseg_sc.cnn_models import nn_architecture_seg, nn_architecture_ctr


def get_parser():
    # Initialize the parser
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
    parser.add_option(name="-ctr",
                      type_value="multiple_choice",
                      description="choice of spinal cord centerline detector.",
                      mandatory=False,
                      example=['svm', 'cnn'],
                      default_value="cnn")
    parser.add_option(name="-brain",
                      type_value="multiple_choice",
                      description="indicate if the input image contains brain sections: 1: contains brain section, 0: no brain section. To indicate this parameter could speed the segmentation process.",
                      mandatory=False,
                      example=["0", "1"],
                      default_value="1")
    parser.add_option(name="-kernel",
                      type_value="multiple_choice",
                      description="choice of 2D or 3D kernels for the segmentation. Note that segmentation with 3D kernels is significantely longer than with 2D kernels.",
                      mandatory=False,
                      example=["2d", "3d"],
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


def scale_intensity(data):
    p2, p98 = np.percentile(data, (2, 98))
    return rescale_intensity(data, in_range=(p2, p98), out_range=(0, 255))


def apply_intensity_normalization(img_path, fname_out):
    img = Image(img_path)
    img_normalized = img.copy()
    p2, p98 = np.percentile(img.data, (2, 98))
    img_normalized.data = scale_intensity(img.data.astype(np.float32))
    img_normalized.changeType('float32')
    img_normalized.setFileName(fname_out)
    img_normalized.save()


def _find_crop_start_end(coord_ctr, crop_size, im_dim):
    half_size = crop_size // 2
    coord_start, coord_end = int(coord_ctr) - half_size + 1, int(coord_ctr) + half_size + 1

    if coord_end > im_dim:
        coord_end = im_dim
        coord_start = im_dim - crop_size if im_dim >= crop_size else 0
    if coord_start < 0:
        coord_start = 0
        coord_end = crop_size if im_dim >= crop_size else im_dim

    return coord_start, coord_end


def crop_image_around_centerline(im_in, ctr_in, im_out, crop_size, x_dim_half, y_dim_half):
    im_in, data_ctr = Image(im_in), Image(ctr_in).data

    im_new = im_in.copy()
    im_new.dim = tuple([crop_size, crop_size, im_in.dim[2]] + list(im_in.dim[3:]))

    data_im_new = np.zeros((crop_size, crop_size, im_in.dim[2]))

    x_lst, y_lst = [], []
    for zz in range(im_in.dim[2]):
        if 1 in np.array(data_ctr[:, :, zz]):
            x_ctr, y_ctr = center_of_mass(np.array(data_ctr[:, :, zz]))

            x_start, x_end = _find_crop_start_end(x_ctr, crop_size, im_in.dim[0])
            y_start, y_end = _find_crop_start_end(y_ctr, crop_size, im_in.dim[1])

            crop_im = np.zeros((crop_size, crop_size))
            x_shape, y_shape = im_in.data[x_start:x_end, y_start:y_end, zz].shape
            crop_im[:x_shape, :y_shape] = im_in.data[x_start:x_end, y_start:y_end, zz]

            data_im_new[:, :, zz] = crop_im

            x_lst.append(str(x_start))
            y_lst.append(str(y_start))

    im_new.data = data_im_new

    im_new.setFileName(im_out)

    im_new.save()

    del im_in
    del im_new

    return x_lst, y_lst


def _remove_extrem_holes(z_lst, end_z, start_z=0):
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

        hole_cur_lst = range(z_hole_start, z_hole_end)
        lenght_hole = len(hole_cur_lst) + 1
        phys_lenght_hole = lenght_hole * z_spaccing

        denom_interpolation = (lenght_hole + 1)

        if phys_lenght_hole < 10:
            sct.log.warning('Filling an hole in the segmentation around z_slice #:' + str(z_ref_start))

            for idx_z, z_hole_cur in enumerate(hole_cur_lst):
                num_interpolation = (lenght_hole - idx_z - 1) * slice_ref_start  # Contribution of the bottom ref slice
                num_interpolation += (idx_z + 1) * slice_ref_end  # Contribution of the top ref slice

                slice_interpolation = num_interpolation * 1. / denom_interpolation
                slice_interpolation = (slice_interpolation > 0).astype(np.int)

                data_interpol[:, :, z_hole_cur] = slice_interpolation

    return data_interpol


def fill_z_holes(fname_in):
    im_in = Image(fname_in)
    data_in = im_in.data.astype(np.int)

    zz_zeros = [zz for zz in range(im_in.dim[2]) if 1 not in list(np.unique(data_in[:, :, zz]))]
    zz_holes = _remove_extrem_holes(zz_zeros, im_in.dim[2] - 1, 0)

    # filling z_holes, i.e. interpolate for z_slice not segmented
    im_in.data = _fill_z_holes(zz_holes, data_in, im_in.dim[6]) if len(zz_holes) else data_in

    im_in.setFileName(fname_in)
    im_in.save()
    del im_in


def scan_slice(z_slice, model, mean_train, std_train, coord_lst, patch_shape, y_crop, z_out_dim):
    z_slice_out = np.zeros(z_out_dim)
    sum_lst = []
    # loop across all the non-overlapping blocks of a cross-sectional slice
    for idx, coord in enumerate(coord_lst):
        block = z_slice[coord[0]:coord[2], coord[1]:coord[3]]
        block_nn = np.expand_dims(np.expand_dims(block, 0), -1)
        block_nn_norm = normalize_data(block_nn, mean_train, std_train)
        block_pred = model.predict(block_nn_norm)

        if coord[2] > z_out_dim[0]:
            x_end = patch_shape[0] - (coord[2] - z_out_dim[0])
        else:
            x_end = patch_shape[0]

        z_slice_out[coord[0]:coord[2], coord[1]:coord[3]] = block_pred[0, :x_end, :, 0]
        sum_lst.append(np.sum(block_pred[0, :x_end, :, 0]))

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
    im = Image(filename_in)
    data_im = im.data.astype(np.float32)
    im_out = im.copy()
    im_out.changeType('uint8')
    del im
    data = np.zeros(im_out.data.shape)

    x_shape, y_shape = data_im.shape[:2]
    x_shape_block, y_shape_block = np.ceil(x_shape * 1.0 / patch_shape[0]).astype(np.int), np.int(y_shape * 1.0 / patch_shape[1])
    x_pad = int(x_shape_block * patch_shape[0] - x_shape)
    y_crop = y_shape - y_shape_block * patch_shape[1]
    # slightly crop the input data in the P-A direction so that data_im.shape[1] % patch_shape[1] == 0
    data_im = data_im[:, :y_shape - y_crop, :]
    # pad the input data in the R-L direction
    data_im = np.pad(data_im, ((0, x_pad), (0, 0), (0, 0)), 'constant')

    # scale intensities between 0 and 255
    data_im = scale_intensity(data_im)

    # coordinates of the blocks to scan during the detection, in the cross-sectional plane
    coord_lst = [[x_dim * patch_shape[0], y_dim * patch_shape[1],
                   (x_dim + 1) * patch_shape[0], (y_dim + 1) * patch_shape[1]]
                    for y_dim in range(y_shape_block) for x_dim in range(x_shape_block)]

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
            block_nn_norm = normalize_data(block_nn, mean_train, std_train)
            block_pred = model.predict(block_nn_norm)

            # coordinates manipulation due to the above padding and cropping
            if x_1 > data.shape[0]:
                x_end = data.shape[0]
                x_1 = data.shape[0]
                x_0 = data.shape[0] - patch_shape[0] if data.shape[0] > patch_shape[0] else 0
            else:
                x_end = patch_shape[0]

            data[x_0:x_1, y_0:y_1, zz] = block_pred[0, :x_end, :, 0]

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
                                                coord_lst, patch_shape, y_crop, data.shape[:2])
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
    im_out.setFileName(filename_out)
    im_out.save()
    del im_out

    z_max = np.max(list(set(np.where(data)[2])))
    if z_max == data.shape[2] - 1:
        return None
    else:
        return z_max


def heatmap2optic(fname_heatmap, lambda_value, fname_out, z_max, algo='dpdt'):
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
        ctr_nii = Image(fname_out)
        ctr_nii.data[:, :, z_max:] = 0
        ctr_nii.save()


def normalize_data(data, mean, std):
    data -= mean
    data /= std
    return data


def segment_2d(model_fname, contrast_type, input_size, fname_in, fname_out):

    dct_params_seg = {
                    # 't2': {'mean': 92.3024483132, 'std': 57.9122089031,
                    #         'features': 32, 'depth': 2, 'batchnorm': True,
                    #         'dropout': 0.0},
                    't1': {'mean': 84.5119262632, 'std': 39.607477199,
                            'features': 8, 'depth': 3, 'batchnorm': True,
                            'dropout': 0.0},
                    'dwi': {'mean': 84.8337225877, 'std': 54.6299357786,
                            'features': 24, 'depth': 2, 'batchnorm': True,
                            'dropout': 0.0}
                    }

    seg_model = nn_architecture_seg(height=input_size[0],
                                    width=input_size[1],
                                    depth=dct_params_seg[contrast_type]['depth'],
                                    features=dct_params_seg[contrast_type]['features'],
                                    batchnorm=dct_params_seg[contrast_type]['batchnorm'],
                                    dropout=dct_params_seg[contrast_type]['dropout'])
    seg_model.load_weights(model_fname)

    # segment the spinal cord
    sct.log.info("Segmenting the spinal cord using deep learning on 2D images...")
    image_normalized = Image(fname_in)
    seg_crop = image_normalized.copy()
    seg_crop.data *= 0
    seg_crop.changeType('uint8')
    for zz in list(reversed(range(image_normalized.dim[2]))):
        z_slice = normalize_data(image_normalized.data[:, :, zz],
                                dct_params_seg[contrast_type]['mean'],
                                dct_params_seg[contrast_type]['std'])
        pred_seg = seg_model.predict(np.expand_dims(np.expand_dims(z_slice, -1), 0))[0, :, :, 0]
        pred_seg_th = (pred_seg > 0.5).astype(int)

        # keep the largest connected obejct per z_slice
        labeled_obj, num_obj = label(pred_seg_th)
        if num_obj > 1:
            pred_seg_th = (labeled_obj == (np.bincount(labeled_obj.flat)[1:].argmax() + 1))

        pred_seg_th = binary_fill_holes(pred_seg_th, structure=np.ones((3, 3))).astype(np.int)

        seg_crop.data[:, :, zz] = pred_seg_th
    seg_crop.setFileName(fname_out)
    seg_crop.save()

    return seg_crop.data


def uncrop_image(fname_ref, fname_out, data_crop, x_crop_lst, y_crop_lst):

    sct.log.info("Reassembling the image...")
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


def main():
    sct.init_sct()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    fname_image = arguments['-i']
    contrast_type = arguments['-c']

    if "-ctr" in arguments:
        ctr_algo = arguments["-ctr"]

    if "-brain" in arguments:
        brain_bool = bool(int(arguments["-brain"]))
    else:
        brain_bool = False if contrast_type in ['t2s', 'dwi'] else True

    if "-kernel" in arguments:
        kernel_size = arguments["-kernel"]

    if "-ofolder" in arguments:
        output_folder = arguments["-ofolder"]
    else:
        output_folder = os.getcwd()

    if '-r' in arguments:
        remove_temp_files = int(arguments['-r'])

    if '-v' in arguments:
        verbose = arguments['-v']

    path_qc = arguments.get("-qc", None)

    fname_seg = deep_segmentation_spinalcord(fname_image, contrast_type, output_folder,
                                            ctr_algo=ctr_algo, brain_bool=brain_bool, kernel_size=kernel_size,
                                            remove_temp_files=remove_temp_files, verbose=verbose)

    if path_qc is not None:
        generate_qc(fname_image, fname_seg, args, os.path.abspath(path_qc))

    sct.display_viewer_syntax([fname_image, os.path.join(output_folder, fname_seg)], colormaps=['gray', 'red'], opacities=['', '0.7'])


def generate_qc(fn_in, fn_seg, args, path_qc):
    """
    Generate a QC entry allowing to quickly review the segmentation process.
    """

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


def deep_segmentation_spinalcord(fname_image, contrast_type, output_folder, ctr_algo='cnn', brain_bool=True, kernel_size='2d', remove_temp_files=1, verbose=1):
    # initalizing parameters
    crop_size = 64  # TODO: this parameter should actually be passed by the model, as it is one of its fixed parameter

    # loading models required to perform the segmentation
    # this step can be long, as some models (e.g., DL) are heavy
    sct.log.info("Loading models...")
    path_script = os.path.dirname(__file__)
    path_sct = os.path.dirname(path_script)

    # create temporary folder with intermediate results
    sct.log.info("Creating temporary folder...")
    file_fname = os.path.basename(fname_image)
    tmp_folder = sct.TempFolder()
    tmp_folder_path = tmp_folder.get_path()
    fname_image_tmp = tmp_folder.copy_from(fname_image)
    tmp_folder.chdir()

    # orientation of the image, should be RPI
    sct.log.info("Reorient the image to RPI, if necessary...")
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
    sct.log.info("Resample the image to 0.5 mm isotropic resolution...")
    fname_res = sct.add_suffix(fname_orient, '_resampled')
    im_2res = im_orient
    input_resolution = im_2res.dim[4:7]
    new_resolution = 'x'.join(['0.5', '0.5', str(input_resolution[2])])
    spinalcordtoolbox.resample.nipy_resample.resample_file(fname_orient, fname_res, new_resolution,
                                                           'mm', 'linear', verbose=0)

    # find the spinal cord centerline - execute OptiC binary
    sct.log.info("Finding the spinal cord centerline...")
    if ctr_algo == 'svm':
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
                        't2s': {'size': (80, 80), 'mean': 68.8591, 'std': 71.4659},
                        't1': {'size': (80, 80), 'mean': 55.7359, 'std': 64.3149},
                        'dwi': {'size': (80, 80), 'mean': 55.744, 'std': 45.003}
                        }
        dct_params_ctr = {'t2': {'height': 80, 'width': 80, 'channels': 1, 'classes': 1,
                                'features': 16, 'depth': 2, 'padding': 'same', 'batchnorm': True,
                                'dropout': 0.0, 'dilation_layers': 2},
                        't2s': {'height': 80, 'width': 80, 'channels': 1, 'classes': 1,
                                'features': 8, 'depth': 2, 'padding': 'same', 'batchnorm': True,
                                'dropout': 0.0, 'dilation_layers': 3},
                        't1': {'height': 80, 'width': 80, 'channels': 1, 'classes': 1,
                                'features': 24, 'depth': 2, 'padding': 'same', 'batchnorm': True,
                                'dropout': 0.0, 'dilation_layers': 3},
                        'dwi': {'height': 80, 'width': 80, 'channels': 1, 'classes': 1,
                                'features': 8, 'depth': 2, 'padding': 'same', 'batchnorm': True,
                                'dropout': 0.0, 'dilation_layers': 2}
                        }
        params_ctr = dct_params_ctr[contrast_type]

        # load model
        ctr_model_fname = os.path.join(path_sct, 'data', 'deepseg_sc_models', '{}_ctr.h5'.format(contrast_type))
        ctr_model = nn_architecture_ctr(height=params_ctr['height'],
                                        width=params_ctr['width'],
                                        channels=params_ctr['channels'],
                                        classes=params_ctr['classes'],
                                        features=params_ctr['features'],
                                        depth=params_ctr['depth'],
                                        temperature=1.0,
                                        padding=params_ctr['padding'],
                                        batchnorm=params_ctr['batchnorm'],
                                        dropout=params_ctr['dropout'],
                                        dilation_layers=params_ctr['dilation_layers'])
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
                      lambda_value=1,
                      fname_out=centerline_filename,
                      z_max=z_max if brain_bool else None)

    # crop image around the spinal cord centerline
    sct.log.info("Cropping the image around the spinal cord...")
    fname_crop = sct.add_suffix(fname_res, '_crop')
    X_CROP_LST, Y_CROP_LST = crop_image_around_centerline(im_in=fname_res, ctr_in=centerline_filename, im_out=fname_crop,
                                                          crop_size=crop_size,
                                                          x_dim_half=crop_size // 2, y_dim_half=crop_size // 2)

    # normalize the intensity of the images
    sct.log.info("Normalizing the intensity...")
    fname_norm = sct.add_suffix(fname_crop, '_norm')
    apply_intensity_normalization(img_path=fname_crop, fname_out=fname_norm)


    if kernel_size == '2d':
        segmentation_model_fname = os.path.join(path_sct, 'data', 'deepseg_sc_models', '{}_sc.h5'.format(contrast_type))
        # segmentation_model_fname = os.path.join(path_sct, 'data', 'deepseg_sc_models', '{}_seg_sc.h5'.format(contrast_type))
        fname_seg_crop = sct.add_suffix(fname_norm, '_seg')
        seg_crop_data = segment_2d(model_fname=segmentation_model_fname,
                                contrast_type=contrast_type,
                                input_size=(crop_size, crop_size),
                                fname_in=fname_norm,
                                fname_out=fname_seg_crop)

        fname_seg_res_RPI = sct.add_suffix(file_fname, '_res_RPI_seg')
        uncrop_image(fname_ref=fname_res,
                    fname_out=fname_seg_res_RPI,
                    data_crop=seg_crop_data,
                    x_crop_lst=X_CROP_LST,
                    y_crop_lst=Y_CROP_LST)

    elif kernel_size == '3d':
        pass

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
    fill_z_holes(fname_in=fname_seg_RPI)

    # reorient to initial orientation
    sct.log.info("Reorienting the segmentation to the original image orientation...")
    fname_seg = sct.add_suffix(file_fname, '_seg')
    if original_orientation != 'RPI':
        im_orient = set_orientation(Image(fname_seg_RPI), original_orientation)
        im_orient.setFileName(fname_seg)
        im_orient.save()
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


if __name__ == "__main__":
    main()
