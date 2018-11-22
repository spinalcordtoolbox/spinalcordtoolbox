# coding: utf-8
# This is the interface API to detect the posterior edge of C2-C3 disc.
# Author: charley
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file LICENSE.TXT

from __future__ import absolute_import

import os
import sct_utils as sct
import numpy as np
import nibabel as nib

from ..image import Image, zeros_like

PATH_MODEL = 'XX'

def detect_c2c3(nii_im, nii_seg, contrast, verbose=1):
    """
    Detect the posterior edge of C2-C3 disc.
    :param nii_im:
    :param nii_seg:
    :param contrast:
    :param verbose:
    :return:
    """
    orientation_init = nii_im.orientation

    # Flatten sagittal
    #### TODO refactor sct_flatten_sagittal to be able to call the function with matrices as input

    # Extract mid-slice
    nii_im.change_orientation('PIR')
    mid_RL = int(np.rint(nii_im.dim[2] * 1.0 / 2))
    midSlice = nii_im.data[:, :, mid_RL]
    nii_midSlice = zeros_like(nii_im)
    nii_midSlice.data = midSlice
    nii_midSlice.save('data_midSlice.nii')

    # Run detection
    sct.printv('Run C2-C3 detector...', verbose)
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_PAIR"
    cmd_detection = 'isct_spine_detect -ctype=dpdt "%s" "%s" "%s"' % \
                    (PATH_MODEL.split('.yml')[0], 'data_midSlice', 'data_midSlice_pred')
    os.system(cmd_detection)
    img = nib.load('data_midSlice_pred_svm.hdr')

    # Create mask along centerline
    # create_mask2d

    # remove img and hdr



def detect_c2c3_from_file(fname_im, fname_seg, contrast, verbose=1):
    """
    Detect the posterior edge of C2-C3 disc.
    :param fname_im:
    :param fname_seg:
    :param contrast:
    :param verbose:
    :return: fname_
    """
    # load data
    sct.printv('Load data...', verbose)
    nii_im = Image(fname_im)
    nii_seg = Image(fname_seg)

    # detect C2-C3
    nii_c2c3 = detect_c2c3(nii_im, nii_seg, contrast, verbose=verbose)