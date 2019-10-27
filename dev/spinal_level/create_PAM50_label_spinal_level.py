#!/usr/bin/env python
#########################################################################################
# 
# - For each PAM50/spinal_levels/spinal_levels_*.nii.gz, compute the center of mass of the distribution.
# - Generate a single nifti file (PAM50_label_spinal_levels.nii.gz).
# - Create a voxel for each of the identified spinal level center of mass.
#	The value of the level is: 2 for spinal level 2, etc.
#
# Usage: python create_PAM50_label_spinal_level.py
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley
# Modified: 2019-10-28
#
#########################################################################################

import os
import numpy as np
from scipy import ndimage

import sct_utils as sct

from spinalcordtoolbox.image import Image, zeros_like

def main():
    path_PAM50 = os.path.join(sct.__data_dir__, 'PAM50')
    path_spinal_levels = os.path.join(path_PAM50, 'spinal_levels')
    path_template = os.path.join(path_PAM50, 'template')

    # init output image
    path_label_disc = os.path.join(path_template, 'PAM50_label_disc.nii.gz')
    path_label_spinal_level = os.path.join(path_template, 'PAM50_label_spinal_levels.nii.gz')
    im_disc = Image(path_label_disc)
    im_lvl = zeros_like(im_disc)
    del im_disc

    # loop across spinal levels
    for i_lvl in range(1, 21):
        # open the ith spinal level distribution
        path_i_lvl = os.path.join(path_spinal_levels, 'spinal_level_'+str(i_lvl).zfill(2)+'.nii.gz')
        im_i_lvl = Image(path_i_lvl)
        data_i_lvl = im_i_lvl.data
        del im_i_lvl

        if not (data_i_lvl == 0).all():
            # compute the center of mass of the distribution
            x_i_lvl, y_i_lvl, z_i_lvl = ndimage.measurements.center_of_mass(data_i_lvl)
            # round values to make indices
            x_i_lvl, y_i_lvl, z_i_lvl = int(np.round(x_i_lvl)), int(np.round(y_i_lvl)), \
                                            int(np.round(z_i_lvl))
            # assign the spinal level label to the computed voxel coordinates
            im_lvl.data[x_i_lvl, y_i_lvl, z_i_lvl] = i_lvl

    # save file
    im_lvl.save(path_label_spinal_level)
    print('Save in: '+path_label_spinal_level)

if __name__ == "__main__":
    main()
