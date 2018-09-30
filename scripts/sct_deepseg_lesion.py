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

from __future__ import print_function, absolute_import, division

import os
import sys

import numpy as np
from scipy.ndimage.measurements import center_of_mass, label
from scipy.ndimage import distance_transform_edt
from scipy.interpolate.interpolate import interp1d

from spinalcordtoolbox.centerline import optic
import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
from msct_parser import Parser
from sct_deepseg_sc import find_centerline, crop_image_around_centerline, uncrop_image, _normalize_data

import spinalcordtoolbox.resample.nipy_resample
from spinalcordtoolbox.deepseg_sc.cnn_models import nn_architecture_ctr

BATCH_SIZE = 4


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
                      example=['t2', 't2_ax', 't2s'])
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


def exp_model(xs, ys, s2):
    x1, x2 = xs
    y1, y2 = ys
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
    data2norm = img.data.astype(np.float32)

    dct_norm = {'t2': [0.000000, 136.832187, 312.158435, 448.968030, 568.657779, 696.671586, 859.221138, 1074.463414, 1373.289174, 1811.522669, 2611.000000],
                't2_ax': [0.000000, 112.195357, 291.611185, 446.727066, 581.103970, 702.979079, 833.318257, 1011.856313, 1268.801813, 1687.137075, 2611.000000],
                't2s': [0.000000, 123.246969, 226.422561, 338.361023, 532.341924, 788.693675, 1096.975553, 1407.979466, 1716.524530, 2079.788451, 2611.000000]}

    img_normalized = msct_image.empty_like(img)
    img_normalized.data = apply_intensity_normalization_model(data2norm, dct_norm[contrast])
    img_normalized.save(fname_out, dtype="float32")


def segment_3d(model_fname, contrast_type, fname_in, fname_out):
    """Perform segmentation with 3D convolutions."""
    from spinalcordtoolbox.deepseg_sc.cnn_models_3d import load_trained_model
    dct_patch_3d = {'t2': {'size': (48, 48, 48), 'mean': 871.309, 'std': 557.916},
                    't2_ax': {'size': (48, 48, 48), 'mean': 835.592, 'std': 528.386},
                    't2s': {'size': (48, 48, 48), 'mean': 1011.31, 'std': 678.985}}

    # load 3d model
    seg_model = load_trained_model(model_fname)

    im = Image(fname_in)

    out = msct_image.zeros_like(im, dtype=np.uint8)

    # segment the spinal cord
    z_patch_size = dct_patch_3d[contrast_type]['size'][2]
    z_step_keep = list(range(0, im.data.shape[2], z_patch_size))
    for zz in z_step_keep:
        if zz == z_step_keep[-1]:  # deal with instances where the im.data.shape[2] % patch_size_z != 0
            patch_im = np.zeros(dct_patch_3d[contrast_type]['size'])
            z_patch_extracted = im.data.shape[2] - zz
            patch_im[:, :, :z_patch_extracted] = im.data[:, :, zz:]
        else:
            z_patch_extracted = z_patch_size
            patch_im = im.data[:, :, zz:z_patch_size + zz]

        if np.any(patch_im):  # Check if the patch is (not) empty, which could occur after a brain detection.
            patch_norm = _normalize_data(patch_im, dct_patch_3d[contrast_type]['mean'], dct_patch_3d[contrast_type]['std'])
            patch_pred_proba = seg_model.predict(np.expand_dims(np.expand_dims(patch_norm, 0), 0), batch_size=BATCH_SIZE)
            pred_seg_th = (patch_pred_proba > 0.1).astype(int)[0, 0, :, :, :]
            if zz == z_step_keep[-1]:
                out.data[:, :, zz:] = pred_seg_th[:, :, :z_patch_extracted]
            else:
                out.data[:, :, zz:z_patch_size + zz] = pred_seg_th

    out.save(fname_out)


def deep_segmentation_MSlesion(fname_image, contrast_type, output_folder, ctr_algo='svm', ctr_file=None, brain_bool=True, remove_temp_files=1, verbose=1):
    """Pipeline."""
    path_script = os.path.dirname(__file__)
    path_sct = os.path.dirname(path_script)

    # create temporary folder with intermediate results
    sct.log.info("\nCreating temporary folder...")
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
    sct.log.info("\nReorient the image to RPI, if necessary...")
    fname_orient = sct.add_suffix(file_fname, '_RPI')
    im_2orient = Image(file_fname)
    original_orientation = im_2orient.orientation
    if original_orientation != 'RPI':
        im_orient = msct_image.change_orientation(im_2orient, 'RPI').save(fname_orient)
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
    contrast_type_ctr = contrast_type.split('_')[0]
    centerline_filename = find_centerline(algo=ctr_algo,
                                      image_fname=fname_res,
                                      path_sct=path_sct,
                                      contrast_type=contrast_type_ctr,
                                      brain_bool=brain_bool,
                                      folder_output=tmp_folder_path,
                                      remove_temp_files=remove_temp_files,
                                      centerline_fname=file_ctr)

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
    thr = '0.1'
    sct.run(['sct_maths', '-i', fname_seg_RPI, '-bin', thr, '-o', fname_seg_RPI], verbose=0)

    # reorient to initial orientation
    sct.log.info("\nReorienting the segmentation to the original image orientation...")
    fname_seg = sct.add_suffix(file_fname, '_lesionseg')
    if original_orientation != 'RPI':
        im_seg_orient = Image(fname_seg_RPI) \
         .change_orientation(original_orientation) \
         .save(fname_seg)

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
    if "-brain" not in args and contrast_type in ['t2s', 't2_ax']:
        brain_bool = False

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

    algo_config_stg = '\nMethod:'
    algo_config_stg += '\n\tCenterline algorithm: ' + str(ctr_algo)
    algo_config_stg += '\n\tAssumes brain section included in the image: ' + str(brain_bool) + '\n'
    sct.printv(algo_config_stg)

    fname_seg = deep_segmentation_MSlesion(fname_image, contrast_type, output_folder,
                                            ctr_algo=ctr_algo, ctr_file=manual_centerline_fname, brain_bool=brain_bool,
                                            remove_temp_files=remove_temp_files, verbose=verbose)

    sct.display_viewer_syntax([fname_image, os.path.join(output_folder, fname_seg)], colormaps=['gray', 'red'], opacities=['', '0.7'])


if __name__ == "__main__":
    main()
