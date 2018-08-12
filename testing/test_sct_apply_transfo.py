#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_apply_transfo
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: generate warping field for dmri that makes sense (dmri --> T2).

from __future__ import absolute_import

import os

import numpy as np

import sct_utils as sct
import spinalcordtoolbox.image as msct_image

def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i template/template/PAM50_small_t2.nii.gz -d t2/t2.nii.gz -w t2/warp_template2anat.nii.gz',
                    '-i dmri/dmri.nii.gz -d t2/t2.nii.gz -w t2/warp_template2anat.nii.gz']

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """

    fname_src = param_test.dict_args_with_path["-i"]
    fname_ref = param_test.dict_args_with_path["-d"]
    fname_dst = sct.add_suffix(os.path.basename(fname_src), "_reg")
    #fname_dst = "output.nii.gz"
    img_src = msct_image.Image(fname_src)
    img_ref = msct_image.Image(fname_ref)
    img_dst = msct_image.Image(fname_dst)

    if img_dst.orientation != img_ref.orientation:
        param_test.output += "\nImage has wrong orientation (%s -> %s)" \
         % (img_ref.orientation, img_dst.orientation)
        param_test.status = 1

    if len(img_src.data.shape) > 3:
        # Allowed failure for now
        return param_test

    if not (img_dst.data != 0).any():
        param_test.output += "\nImage is garbage (all zeros)"
        param_test.status = 1


    return param_test
