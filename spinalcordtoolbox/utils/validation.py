#!/usr/bin/env python
# -*- coding: utf-8
# Validation related utilites


def check_dimensions_match(input_im, im_vertebral_labeling):
    nx, ny, nz = input_im.data.shape
    nx_vertebral, ny_vertebral, nz_vertebral = im_vertebral_labeling.data.shape

    # Check dimensions consistency between the input and the vertebral labeling file
    if (nx, ny, nz) != (nx_vertebral, ny_vertebral, nz_vertebral):
        return False
    else:
        return True