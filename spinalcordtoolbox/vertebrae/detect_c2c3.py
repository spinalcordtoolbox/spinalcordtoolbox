# coding: utf-8
# This is the interface API to detect the posterior edge of C2-C3 disc.
#
# The models have been trained as explained in (Gros et al. 2018, MIA, doi.org/10.1016/j.media.2017.12.001),
# in section 2.1.2, except that the cords are not straightened for the C2-C3 disc detection task.
#
# To train a new model:
# - Edit "$SCT_DIR/dev/detect_c2c3/config_file.py" according to your needs, then save the file.
# - Run "source sct_launcher" in a terminal
# - Run the script "$SCT_DIR/dev/detect_c2c3/train.py"
#
# To use this model when running the module "detect_c2c3" (herebelow) and "sct_label_vertebrae":
# - Save the trained model in "$SCT_DIR/data/c2c3_disc_models/"
#
# Author: charley
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file LICENSE.TXT

from __future__ import absolute_import

import os
import sys
import logging
import sct_utils as sct
from sct_flatten_sagittal import flatten_sagittal
import numpy as np
import nibabel as nib
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label as label_regions

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image, zeros_like

logger = logging.getLogger(__name__)


def detect_c2c3(nii_im, nii_seg, contrast, nb_sag_avg=7.0, verbose=1):
    """
    Detect the posterior edge of C2-C3 disc.
    :param nii_im:
    :param nii_seg:
    :param contrast:
    :param verbose:
    :return:
    """
    # path to the pmj detector
    path_model = os.path.join(sct.__data_dir__, 'c2c3_disc_models', '{}_model'.format(contrast))
    # check if model exists
    if not os.path.isfile(path_model+'.yml'):
        raise FileNotFoundError(
            "The model file {} does not exist. Please download it using sct_download_data".format(path_model+'.yml'))

    orientation_init = nii_im.orientation
    z_seg_max = np.max(np.where(nii_seg.change_orientation('PIR').data)[1])

    # Flatten sagittal
    nii_im = flatten_sagittal(nii_im, nii_seg,verbose=verbose)
    nii_seg_flat = flatten_sagittal(nii_seg, nii_seg, verbose=verbose)

    # create temporary folder with intermediate results
    logger.info("Creating temporary folder...")
    tmp_folder = sct.TempFolder()
    tmp_folder.chdir()

    # Extract mid-slice
    nii_im.change_orientation('PIR')
    nii_seg_flat.change_orientation('PIR')
    mid_RL = int(np.rint(nii_im.dim[2] * 1.0 / 2))
    nb_sag_avg_half = int(nb_sag_avg / 2 / nii_im.dim[6])
    midSlice = np.mean(nii_im.data[:, :, mid_RL-nb_sag_avg_half:mid_RL+nb_sag_avg_half+1], 2) # average 7 slices
    midSlice_seg = nii_seg_flat.data[:, :, mid_RL]
    nii_midSlice = msct_image.zeros_like(nii_im)
    nii_midSlice.data = midSlice
    nii_midSlice.save('data_midSlice.nii')

    # Run detection
    logger.info('Run C2-C3 detector...')
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_PAIR"
    cmd_detection = 'isct_spine_detect -ctype=dpdt "%s" "%s" "%s"' % \
                    (path_model, 'data_midSlice', 'data_midSlice_pred')
    # The command below will fail, but we don't care because it will output an image (prediction), which we
    # will use later on.
    s, o = sct.run(cmd_detection, verbose=0, is_sct_binary=True, raise_exception=False)
    pred = nib.load('data_midSlice_pred_svm.hdr').get_data()
    if verbose >= 2:
        # copy the "prediction data before post-processing" in an Image object
        nii_pred_before_postPro = nii_midSlice.copy()
        nii_pred_before_postPro.data = pred  # 2D data with orientation, mid sag slice of the original data
        nii_pred_before_postPro.save("pred_midSlice_before_postPro.nii.gz")  # save it)
    # DEBUG trick: check if the detection succeed by running: fsleyes data_midSlice data_midSlice_pred_svm -cm red -dr 0 100
    # If a "red cluster" is observed in the neighbourhood of C2C3, then the model detected it.

    # Create mask along centerline
    midSlice_mask = np.zeros(midSlice_seg.shape)
    mask_halfSize = int(np.rint(25.0 / nii_midSlice.dim[4]))
    for z in range(midSlice_mask.shape[1]):
        row = midSlice_seg[:, z]  # 2D data with PI orientation, mid sag slice of the original data
        if np.any(row > 0):
            med_y = int(np.rint(np.median(np.where(row > 0))))
            midSlice_mask[med_y-mask_halfSize:med_y+mask_halfSize, z] = 1  # 2D data with PI orientation, mid sag slice of the original data
    if verbose >= 2:
        # copy the created mask in an Image object
        nii_postPro_mask = nii_midSlice.copy()
        nii_postPro_mask.data = midSlice_mask  # 2D data with PI orientation, mid sag slice of the original data
        nii_postPro_mask.save("mask_midSlice.nii.gz")  # save it

    # mask prediction
    pred[midSlice_mask == 0] = 0
    pred[:, z_seg_max:] = 0  # Mask above SC segmentation
    if verbose >= 2:
        # copy the "prediction data after post-processing" in an Image object
        nii_pred_after_postPro = nii_midSlice.copy()
        nii_pred_after_postPro.data = pred
        nii_pred_after_postPro.save("pred_midSlice_after_postPro.nii.gz")  # save it

    # assign label to voxel
    nii_c2c3 = zeros_like(nii_seg_flat)  # 3D data with PIR orientaion
    if np.any(pred > 0):
        logger.info('C2-C3 detected...')

        pred_bin = (pred > 0).astype(np.int_)
        coord_max = np.where(pred == np.max(pred))
        pa_c2c3, is_c2c3 = coord_max[0][0], coord_max[1][0]
        nii_seg.change_orientation('PIR')
        rl_c2c3 = int(np.rint(center_of_mass(np.array(nii_seg.data[:, is_c2c3, :]))[1]))
        nii_c2c3.data[pa_c2c3, is_c2c3, rl_c2c3] = 3
    else:
        logger.warning('C2-C3 not detected...')

    # remove temporary files
    tmp_folder.chdir_undo()
    if verbose < 2:
        logger.info("Remove temporary files...")
        tmp_folder.cleanup()
    else:
        logger.info("Temporary files saved to "+tmp_folder.get_path())

    nii_c2c3.change_orientation(orientation_init)
    return nii_c2c3


def detect_c2c3_from_file(fname_im, fname_seg, contrast, fname_c2c3=None, verbose=1):
    """
    Detect the posterior edge of C2-C3 disc.
    :param fname_im:
    :param fname_seg:
    :param contrast:
    :param fname_c2c3:
    :param verbose:
    :return: fname_c2c3
    """
    # load data
    logger.info('Load data...')
    nii_im = Image(fname_im)
    nii_seg = Image(fname_seg)

    # detect C2-C3
    nii_c2c3 = detect_c2c3(nii_im.copy(), nii_seg, contrast, verbose=verbose)

    # Output C2-C3 disc label
    # by default, output in the same directory as the input images
    logger.info('Generate output file...')
    if fname_c2c3 is None:
        fname_c2c3 = os.path.join(os.path.dirname(nii_im.absolutepath), "label_c2c3.nii.gz")
    nii_c2c3.save(fname_c2c3)

    return fname_c2c3
