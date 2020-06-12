#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_dmri_moco
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from pandas import read_csv
from numpy import allclose

import sct_utils as sct


def init(param_test):
    """
    Initialize class: param_test
    """
    # Reorient image to sagittal for testing another orientation (and crop to save time)
    sct.run('sct_image -i dmri/dmri.nii.gz -setorient AIL -o dmri/dmri_AIL.nii', verbose=0)
    sct.run('sct_crop_image -i dmri/dmri_AIL.nii -zmin 19 -zmax 21 -o dmri/dmri_AIL_crop.nii', verbose=0)
    # Create Gaussian mask for testing
    sct.run('sct_create_mask -i dmri/dmri_T0000.nii.gz -p center -size 5mm -f gaussian -o dmri/mask.nii', verbose=0)

    # initialization
    default_args = [
        '-i dmri/dmri.nii.gz -bvec dmri/bvecs.txt -g 3 -x nn -ofolder dmri_test1 -r 0',
        '-i dmri/dmri.nii.gz -bvec dmri/bvecs.txt -g 3 -m dmri/mask.nii -ofolder dmri_test2 -r 0',
        '-i dmri/dmri_AIL_crop.nii -bvec dmri/bvecs.txt -x nn -ofolder dmri_test3 -r 0',
        ]

    # Output moco param files
    param_test.file_mocoparam = [
        'dmri_test1/moco_params.tsv',
        'dmri_test2/moco_params.tsv',
        None,
        ]

    # Ground truth value for integrity testing (corresponds to X motion parameters column)
    param_test.groundtruth = [
        [0.00047529041677414337, -1.1970542445283172e-05, -1.1970542445283172e-05, -1.1970542445283172e-05, -0.1296642741802682,
        -0.1296642741802682, -0.1296642741802682],
        [0.008491259664160821, 0.0029020670607778237, 0.0029020670607778237, 0.0029020670607778237, -0.014405996135095476,
        -0.014405996135095476, -0.014405996135095476],
        None,
        ]

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

    # Open motion parameters and compare with ground truth
    # NB: We skip the test for sagittal images (*_AIL) because there is no output moco params
    if param_test.file_mocoparam[index_args] is not None:
        df = read_csv(param_test.file_mocoparam[index_args], sep="\t")
        lresults = list(df['X'][:])
        lgroundtruth = param_test.groundtruth[index_args]
        if allclose(lresults, lgroundtruth):
            param_test.output += "\n--> PASSED"
        else:
            param_test.output += "\nMotion parameters do not match: " \
                                 "  results: {}" \
                                 "  ground truth: {}".format(lresults, lgroundtruth)
            param_test.status = 99
            param_test.output += "\n--> FAILED"

    return param_test
