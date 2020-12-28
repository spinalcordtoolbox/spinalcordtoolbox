#!/usr/bin/env python
# -*- coding: utf-8
# Validation related utilites


def check_dimensions_match(input_metric, im_vertebral_labeling):
    # Check dimensions consistency between the input and the vertebral labeling file
    if (input_metric.data.shape) != (im_vertebral_labeling.data.shape):
        return False
    else:
        return True