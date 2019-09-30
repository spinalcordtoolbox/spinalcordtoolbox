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
    param_test.fname_out = [
        'PAM50_small_t2_reg.nii',
        'PAM50_small_t2_reg-crop1.nii',
        'PAM50_small_t2_reg-crop2.nii',
        'PAM50_small_t2_reg-concatWarp.nii',
        'PAM50_small_t2_reg-4D.nii',
    ]
    default_args = [
        '-i template/template/PAM50_small_t2.nii.gz -d t2/t2.nii.gz -w t2/warp_template2anat.nii.gz -o {}'.format(param_test.fname_out[0]),
        '-i template/template/PAM50_small_t2.nii.gz -d t2/t2.nii.gz -w t2/warp_template2anat.nii.gz -crop 1 -o {}'.format(param_test.fname_out[1]),
        '-i template/template/PAM50_small_t2.nii.gz -d t2/t2.nii.gz -w t2/warp_template2anat.nii.gz -crop 2 -o {}'.format(param_test.fname_out[2]),
        '-i template/template/PAM50_small_t2.nii.gz -d t2/t2.nii.gz -w t2/warp_template2anat.nii.gz t2/warp_template2anat.nii.gz -o {}'.format(param_test.fname_out[3]),
        '-i dmri/dmri.nii.gz -d t2/t2.nii.gz -w t2/warp_template2anat.nii.gz -o {}'.format(param_test.fname_out[4]),
    ]
    param_test.input = 'template/template/PAM50_small_t2.nii.gz'
    param_test.ref = 't2/t2.nii.gz'

    # assign default params
    if not param_test.args:
        param_test.args = default_args

    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # find which test is performed
    index_args = param_test.default_args.index(param_test.args)

    img_src = Image(param_test.input)
    img_ref = Image(param_test.ref)
    img_output = Image(param_test.fname_out[index_args])

    if img_output.orientation != img_ref.orientation:
        param_test.output += "\nImage has wrong orientation (%s -> %s)" % (img_ref.orientation, img_output.orientation)
        param_test.status = 99

    if not (img_output.data != 0).any():
        param_test.output += "\nImage is garbage (all zeros)"
        param_test.status = 99

    # Only checking the first 3 dimensions because one test involves a 4D volume
    if not img_ref.dim[0:3] == img_output.dim[0:3]:
        param_test.output += "\nRef and output images don't have the same first 3 dimensions."
        param_test.status = 99

    # Checking the 4th dim (which should be the same as the input image, not the reference image)
    if not img_src.dim[3] == img_output.dim[3]:
        param_test.output += "\nInput and output images don't have the same 4th dimension' size."
        param_test.status = 99


    if param_test.status == 99:
        param_test.output += "\n--> FAILED"
    else:
        param_test.output += "\n--> PASSED"

    return param_test
