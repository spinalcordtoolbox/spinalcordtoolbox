#!/usr/bin/env python
#########################################################################################
#
# Test function sct_crop_image
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# modified: 2015-03-30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
import msct_image


def test(data_path):

    # parameters
    folder_data = 't2/'
    file_data = ['t2.nii.gz', 't2_seg.nii.gz']

    # test normal crop
    cmd = 'sct_crop_image -i ' + data_path + folder_data + file_data[0] \
          + ' -o cropped_normal.nii.gz -dim 1 -start 10 -end 50'
    status, output = sct.run(cmd, 0)

    if status == 0:
        # check if cropping was correct
        nx, ny, nz, nt, px, py, pz, pt = msct_image.Image('cropped_normal.nii.gz').dim
        if (ny != 41):
            status = 5
    return status, output


if __name__ == "__main__":
    # call main function
    test()
