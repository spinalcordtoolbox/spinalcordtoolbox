#!/usr/bin/env python
#########################################################################################
#
# Test function sct_apply_transfo
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: generate warping field for dmri that makes sense (dmri --> T2).

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
    param_test.output += '\nNot implemented.'
    return param_test
