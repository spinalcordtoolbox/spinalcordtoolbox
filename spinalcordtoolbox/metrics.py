#!/usr/bin/env python
#########################################################################################
#
# SCT Metrics API
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import numpy as np

def compute_dice(image1, image2, mode='3d', label=1, zboundaries=False):
    """
    This function computes the Dice coefficient between two binary images.
    Args:
        image1: object Image
        image2: object Image
        mode: mode of computation of Dice.
                3d: compute Dice coefficient over the full 3D volume
                2d-slices: compute the 2D Dice coefficient for each slice of the volumes
        label: binary label for which Dice coefficient will be computed. Default=1
        zboundaries: True/False. If True, the Dice coefficient is computed over a Z-ROI where both segmentations are
                     present. Default=False.

    Returns: Dice coefficient as a float between 0 and 1. Raises ValueError exception if an error occurred.

    """
    MODES = ['3d', '2d-slices']
    if mode not in MODES:
        raise ValueError('\n\nERROR: mode must be one of these values:' + ',  '.join(MODES))

    dice = 0.0  # default value of dice is 0

    # check if images are in the same coordinate system
    assert image1.data.shape == image2.data.shape, "\n\nERROR: the data (" + image1.absolutepath + " and " + image2.absolutepath + ") don't have the same size.\nPlease use  \"sct_register_multimodal -i im1.nii.gz -d im2.nii.gz -identity 1\"  to put the input images in the same space"

    # if necessary, change orientation of images to RPI and compute segmentation boundaries
    if mode == '2d-slices' or (mode == '3d' and zboundaries):
        # changing orientation to RPI if necessary
        if image1.orientation != 'RPI':
            image1_c = image1.copy()
            image1_c.change_orientation('RPI')
            image1 = image1_c
        if image2.orientation != 'RPI':
            image2_c = image2.copy()
            image2_c.change_orientation('RPI')
            image2 = image2_c

        zmin, zmax = 0, image1.data.shape[2] - 1
        if zboundaries:
            # compute Z-ROI for which both segmentations are present.
            for z in range(zmin, zmax + 1):  # going from inferior to superior
                if np.any(image1.data[:, :, z]) and np.any(image2.data[:, :, z]):
                    zmin = z
                    break
            for z in range(zmax, zmin + 1, -1):  # going from superior to inferior
                if np.any(image1.data[:, :, z]) and np.any(image2.data[:, :, z]):
                    zmax = z
                    break

        if zmin > zmax:
            # segmentations do not overlap
            return 0.0

        if mode == '3d':
            # compute dice coefficient over Z-ROI
            data1 = image1.data[:, :, zmin:zmax]
            data2 = image2.data[:, :, zmin:zmax]

            dice = np.sum(data2[data1 == label]) * 2.0 / (np.sum(data1) + np.sum(data2))

        elif mode == '2d-slices':
            raise ValueError('2D slices Dice coefficient feature is not implemented yet')

    elif mode == '3d':
        # compute 3d dice coefficient
        dice = np.sum(image2.data[image1.data == label]) * 2.0 / (np.sum(image1.data) + np.sum(image2.data))

    return dice
