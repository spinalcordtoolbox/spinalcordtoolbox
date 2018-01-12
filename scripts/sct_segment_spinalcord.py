#!/usr/bin/env python
#########################################################################################
#
# Function to segment the spinal cord using deep convolutional networks
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener & Charley Gros
# Modified: 2017-12-14
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import numpy as np
import shutil
from scipy.ndimage.measurements import center_of_mass
import pickle
from scipy.interpolate.interpolate import interp1d
from skimage.exposure import rescale_intensity


import os
import sys
from spinalcordtoolbox.centerline import optic
import sct_utils as sct
from msct_image import Image
from msct_parser import Parser

from keras.models import load_model
import spinalcordtoolbox.resample.nipy_resample
from spinalcordtoolbox.segmentation.cnn_models import dice_coef, dice_coef_loss



def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description("""This function allows the extraction of the spinal cord centerline. Two methods are available: OptiC (automatic) and Viewer (manual).""")

    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t1.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast.",
                      mandatory=False,
                      example=['t1', 't2', 't2s', 'dwi'])
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
    return parser


def apply_intensity_normalization_model(img_path, landmarks_pd, fname_out, max_interp='exp'):
    # Description: apply the learned intensity landmarks to the input image

    img = Image(img_path)

    img_data = np.asarray(img.data)

    percent_decile_lst = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    vals = list(img_data)
    landmarks_lst_cur = np.percentile(vals, q=percent_decile_lst)

    # treat single intensity accumulation error
    if not len(np.unique(landmarks_lst_cur)) == len(landmarks_lst_cur):
        output = rescale_intensity(img_data, out_range=(0, 255))

    else:
        # create linear mapping models for the percentile segments to the learned standard intensity space
        linear_mapping = interp1d(landmarks_lst_cur, landmarks_pd.values, bounds_error = False)

        # transform the input image intensity values
        output = linear_mapping(img_data)

        # treat image intensity values outside of the cut-off percentiles range separately
        below_mapping = exp_model(landmarks_lst_cur[:2], landmarks_pd.values[:2], landmarks_pd.values[0])
        output[img_data < landmarks_lst_cur[0]] = below_mapping(img_data[img_data < landmarks_lst_cur[0]])

        if max_interp == 'exp':
            above_mapping = exp_model(landmarks_lst_cur[-3:-1], landmarks_pd.values[-3:-1], landmarks_pd.values[-1])
        elif max_interp == 'linear':
            above_mapping = linear_model(landmarks_lst_cur[-2:], landmarks_pd.values[-2:])
        elif max_interp == 'flat':
            above_mapping = lambda x: landmarks_pd.values[-1]
        else:
            print 'No model was chosen, will use flat'
            above_mapping = lambda x: landmarks_pd.values[-1]
        output[img_data > landmarks_lst_cur[-1]] = above_mapping(img_data[img_data > landmarks_lst_cur[-1]])

    #print np.min(output), np.max(output)

    # save resulting image
    img_normalized = img.copy()
    img_normalized.data = output
    img_normalized.setFileName(fname_out)
    img_normalized.save()

    return img_normalized


def linear_model((x1, x2), (y1, y2)):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)
    return lambda x: m * x + b


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


class SingleIntensityAccumulationError(Exception):
    """
    Thrown when an image shows an unusual single-intensity peaks which would obstruct
    both, training and transformation.
    """


def crop_image_around_centerline(im_in, ctr_in, im_out, crop_size, x_dim_half, y_dim_half):
    im_in, im_ctr = Image(im_in), Image(ctr_in)

    im_new = im_in.copy()
    im_new.dim = tuple([crop_size, crop_size, im_in.dim[2]] + list(im_in.dim[3:]))

    data_im_new = np.zeros((crop_size, crop_size, im_in.dim[2]))

    x_lst, y_lst = [], []
    for zz in range(im_in.dim[2]):
        x_ctr, y_ctr = center_of_mass(im_ctr.data[:, :, zz])

        x_start, x_end = int(x_ctr) - x_dim_half + 1, int(x_ctr) + x_dim_half + 1
        y_start, y_end = int(y_ctr) - y_dim_half + 1, int(y_ctr) + y_dim_half + 1

        if y_start < 0:
            y_start, y_end = 0, crop_size
        if y_end > im_in.dim[1]:
            y_start, y_end = im_in.dim[1] - crop_size, im_in.dim[1]
        if x_start < 0:
            x_start, x_end = 0, crop_size
        if x_end > im_in.dim[0]:
            x_start, x_end = im_in.dim[0] - crop_size, im_in.dim[0]

        crop_im = np.zeros((crop_size, crop_size))
        x_shape, y_shape = im_in.data[x_start:x_end, y_start:y_end, zz].shape
        crop_im[:x_shape, :y_shape] = im_in.data[x_start:x_end, y_start:y_end, zz]

        data_im_new[:, :, zz] = crop_im

        x_lst.append(str(x_start))
        y_lst.append(str(y_start))

    im_new.data = data_im_new

    im_new.setFileName(im_out)

    im_new.save()

    del im_in, im_ctr
    del im_new

    return x_lst, y_lst


def main():
    sct.start_stream_logger()
    parser = get_parser()
    args = sys.argv[1:]
    arguments = parser.parse(args)

    fname_image = arguments['-i']
    contrast_type = arguments['-c']

    if "-ofolder" in arguments:
        output_folder = arguments["-ofolder"]
    else:
        output_folder = os.getcwd()

    if '-r' in arguments:
        remove_temp_files = arguments['-r']

    if '-v' in arguments:
        verbose = arguments['-v']

    deep_segmentation_spinalcord(fname_image, contrast_type, output_folder, remove_temp_files, verbose)


def deep_segmentation_spinalcord(fname_image, contrast_type, output_folder, remove_temp_files=1, verbose=1):
    # initalizing parameters
    crop_size = 64  # TODO: this parameter should actually be passed by the model, as it is one of its fixed parameter

    # initializing objects
    custom_objects = {'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}

    # loading models required to perform the segmentation
    # this step can be long, as some models (e.g., DL) are heavy
    path_script = os.path.dirname(__file__)
    path_sct = os.path.dirname(path_script)
    optic_models_fname = os.path.join(path_sct, 'data/optic_models', '{}_model'.format(contrast_type))

    intensity_norm_model_fname = os.path.join(path_sct, 'data/deepscseg_models', 'intensity_norm_model.pkl')
    intensity_norm_model = pickle.load(open(intensity_norm_model_fname, "rb"))[contrast_type]

    segmentation_model_fname = os.path.join(path_sct, 'data/deepscseg_models', '{}_seg_sc.h5'.format(contrast_type))
    seg_model = load_model(segmentation_model_fname, custom_objects=custom_objects)

    # create temporary folder with intermediate results
    file_fname = os.path.basename(fname_image)
    tmp_folder = sct.TempFolder()
    tmp_folder_path = tmp_folder.get_path()
    fname_image_tmp = tmp_folder.copy_from(fname_image)
    tmp_folder.chdir()

    # orientation of the image, should be RPI
    fname_orient = sct.add_suffix(file_fname, '_RPI')
    im_2orient = Image(file_fname)
    original_orientation = im_2orient.orientation
    if original_orientation != 'RPI':
        sct.run('sct_image -i ' + file_fname + ' -setorient RPI -o ' + fname_orient)
    else:
        shutil.copyfile(fname_image_tmp, fname_orient)

    # resampling RPI image
    fname_res = sct.add_suffix(fname_orient, '_resampled')
    im_2res = Image(fname_orient)
    input_resolution = im_2res.dim[4:7]
    new_resolution = 'x'.join(['0.5', '0.5', str(input_resolution[2])])
    spinalcordtoolbox.resample.nipy_resample.resample_file(fname_orient, fname_res, new_resolution,
                                                           'mm', 'linear', verbose)

    # find the spinal cord centerline - execute OptiC binary
    _, centerline_filename = optic.detect_centerline(image_fname=fname_res,
                                                     contrast_type=contrast_type,
                                                     optic_models_path=optic_models_fname,
                                                     folder_output=tmp_folder_path,
                                                     remove_temp_files=remove_temp_files,
                                                     output_roi=False,
                                                     verbose=verbose)

    # crop image around the spinal cord centerline
    fname_crop = sct.add_suffix(fname_res, '_crop')
    X_CROP_LST, Y_CROP_LST = crop_image_around_centerline(im_in=fname_res, ctr_in=centerline_filename, im_out=fname_crop,
                                                          crop_size=crop_size,
                                                          x_dim_half=crop_size // 2, y_dim_half=crop_size // 2)

    # normalize the intensity of the images
    fname_norm = sct.add_suffix(fname_crop, '_norm')
    image_normalized = apply_intensity_normalization_model(img_path=fname_crop,
                                                           landmarks_pd=intensity_norm_model,
                                                           fname_out=fname_norm)

    # segment the spinal cord
    fname_seg_crop = sct.add_suffix(fname_norm, '_seg')
    seg_crop = image_normalized.copy()
    seg_crop.data *= 0.0
    for zz in range(image_normalized.dim[2]):
        pred_seg = seg_model.predict(np.expand_dims(np.expand_dims(image_normalized.data[:, :, zz], -1), 0))[0, :, :, 0]
        pred_seg = (pred_seg > 0.5).astype(int)
        seg_crop.data[:, :, zz] = pred_seg
    seg_crop.setFileName(fname_seg_crop)
    seg_crop.save()

    fname_seg_res_RPI = sct.add_suffix(file_fname, '_res_RPI_seg')
    im = Image(fname_res)
    seg_unCrop = im.copy()
    seg_unCrop.data *= 0

    for zz in range(seg_unCrop.dim[2]):
        pred_seg = seg_crop.data[:, :, zz]
        x_start, y_start = int(X_CROP_LST[zz]), int(Y_CROP_LST[zz])
        x_end = x_start + crop_size if x_start + crop_size < seg_unCrop.dim[0] else seg_unCrop.dim[0]
        y_end = y_start + crop_size if y_start + crop_size < seg_unCrop.dim[1] else seg_unCrop.dim[1]
        seg_unCrop.data[x_start:x_end, y_start:y_end, zz] = pred_seg[0:x_end - x_start, 0:y_end - y_start]

    seg_unCrop.setFileName(fname_seg_res_RPI)
    seg_unCrop.save()

    # resample to initial resolution
    fname_seg_RPI = sct.add_suffix(file_fname, '_RPI_seg')
    initial_resolution = 'x'.join([str(input_resolution[0]), str(input_resolution[1]), str(input_resolution[2])])
    spinalcordtoolbox.resample.nipy_resample.resample_file(fname_seg_res_RPI, fname_seg_RPI, initial_resolution,
                                                           'mm', 'linear', verbose)

    # binarize the resampled image to remove interpolation effects
    sct.run('sct_maths -i ' + fname_seg_RPI + ' -bin 0.75 -o ' + fname_seg_RPI)

    # reorient to initial orientation
    fname_seg = sct.add_suffix(file_fname, '_seg')
    if original_orientation != 'RPI':
        sct.run('sct_image -i ' + fname_seg_RPI + ' -setorient ' + original_orientation + ' -o ' + fname_seg)
    else:
        shutil.copyfile(fname_seg_RPI, fname_seg)

    tmp_folder.chdir_undo()

    # copy image from temporary folder into output folder
    shutil.copyfile(tmp_folder_path + '/' + fname_seg, output_folder + '/' + fname_seg)

    sct.display_viewer_syntax([fname_image, output_folder + '/' + fname_seg], colormaps=['gray', 'red'], opacities=['', '0.7'])

    # TODO: add QC report


if __name__ == "__main__":
    main()
