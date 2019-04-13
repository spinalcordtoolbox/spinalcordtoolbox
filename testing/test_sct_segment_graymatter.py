#!/usr/bin/env python
#########################################################################################
#
# Test function sct_segment_graymatter
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# modified: 2015/08/31
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

import os

import numpy as np

import sct_utils as sct


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2s/t2s.nii.gz -s t2s/t2s_seg.nii.gz -vertfile t2s/MNI-Poly-AMU_level_crop.nii.gz -ref t2s/t2s_gmseg_manual.nii.gz']

    param_test.hd_threshold = 3  # in mm
    param_test.wm_dice_threshold = 0.8

    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """

    target_name = sct.extract_fname(param_test.file_input)[1]

    dice_fname = os.path.join(param_test.path_output, 'dice_coefficient_' + target_name + '.txt')
    hausdorff_fname = os.path.join(param_test.path_output, 'hausdorff_dist_' + target_name + '.txt')

    # Extracting dice results:
    dice = open(dice_fname, 'r')
    dice_lines = dice.readlines()
    dice.close()
    gm_start = dice_lines.index('Dice coefficient on the Gray Matter segmentation:\n')
    wm_start = dice_lines.index('Dice coefficient on the White Matter segmentation:\n')

    # extracting dice on GM
    gm_dice_lines = dice_lines[gm_start:wm_start - 1]
    gm_dice_lines = gm_dice_lines[gm_dice_lines.index('2D Dice coefficient by slice:\n') + 1:-1]

    null_slices = []
    gm_dice = []
    for line in gm_dice_lines:
        try:
            n_slice, dc = line.split(' ')
            # remove \n from dice result
            dc = dc[:-1]
            dc = dc[:-4] if '[0m' in dc else dc

            if dc == '0' or dc == 'nan':
                null_slices.append(n_slice)
            else:
                try:
                    gm_dice.append(float(dc))
                except ValueError:
                    gm_dice.append(float(dc[:-4]))
        except ValueError:
            # Hack to avoid "too many values to unpack" due to the recent addition of "Total processing time:" at the
            # end of a process. In the future we should simply output dice results as external csv file.
            logger.debug("ValueError: Ignoring this line.")
    result_dice_gm = np.mean(gm_dice)

    # extracting dice on WM
    wm_dice_lines = dice_lines[wm_start:]
    wm_dice_lines = wm_dice_lines[wm_dice_lines.index('2D Dice coefficient by slice:\n') + 1:]
    wm_dice = []
    for line in wm_dice_lines:
        try:
            n_slice, dc = line.split(' ')
            # remove \n from dice result
            if line is not wm_dice_lines[-1]:
                dc = dc[:-1]
            if n_slice not in null_slices:
                try:
                    wm_dice.append(float(dc))
                except ValueError:
                    wm_dice.append(float(dc[:-4]))
        except ValueError:
            # Hack to avoid "too many values to unpack" due to the recent addition of "Total processing time:" at the
            # end of a process. In the future we should simply output dice results as external csv file.
            logger.debug("ValueError: Ignoring this line.")
    result_dice_wm = np.mean(wm_dice)

    # Extracting hausdorff distance results
    hd = open(hausdorff_fname, 'r')
    hd_lines = hd.readlines()
    hd.close()

    # remove title of columns and last empty/non important lines
    hd_lines = hd_lines[1:-4]

    hausdorff = []
    max_med = []
    for line in hd_lines:
        slice_id, res = line.split(':')
        slice, n_slice = slice_id.split(' ')
        if n_slice not in null_slices:
            hd, med1, med2 = res[:-1].split(' - ')
            hd, med1, med2 = float(hd), float(med1), float(med2)
            hausdorff.append(hd)
            max_med.append(max(med1, med2))

    result_hausdorff = np.mean(hausdorff)
    result_median_dist = np.mean(max_med)

    # Integrity check
    if result_hausdorff > param_test.hd_threshold or result_dice_wm < param_test.wm_dice_threshold:
        param_test.status = 99
        param_test.output += '\nResulting segmentation is too different from manual segmentation:\n' \
                             'WM dice: ' + str(result_dice_wm) + '\n' \
                             'Hausdorff distance: ' + str(result_hausdorff) + '\n'

    # update Panda structure
    param_test.results['dice_gm'] = result_dice_gm
    param_test.results['dice_wm'] = result_dice_wm
    param_test.results['hausdorff'] = result_hausdorff
    param_test.results['med_dist'] = result_median_dist

    return param_test
