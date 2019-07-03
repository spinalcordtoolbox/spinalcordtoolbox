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

from spinalcordtoolbox.image import Image


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i template/template/PAM50_small_t2.nii.gz -d t2/t2.nii.gz -w t2/warp_template2anat.nii.gz',
                    '-i dmri/dmri.nii.gz -d t2/t2.nii.gz -w t2/warp_template2anat.nii.gz']
    param_test.input = 'template/template/PAM50_small_t2.nii.gz'
    param_test.ref = 't2/t2.nii.gz'
    param_test.out = 'PAM50_small_t2_reg.nii.gz'

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    img_src = Image(param_test.input)
    img_ref = Image(param_test.ref)
    img_output = Image(param_test.out)

    if img_output.orientation != img_ref.orientation:
        param_test.output += "\nImage has wrong orientation (%s -> %s)" % (img_ref.orientation, img_output.orientation)
        param_test.status = 99

    if len(img_src.data.shape) > 3:
        # Allowed failure for now
        return param_test

    if not (img_output.data != 0).any():
        param_test.output += "\nImage is garbage (all zeros)"
        param_test.status = 99

    return param_test
