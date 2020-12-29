#!/usr/bin/env python
# -*- coding: utf-8
# Validation related utilites


def check_dimensions_match(metric, img):
    """
    This function checks dimensions consistency between the input metric and image

    :param metric: the object Metric to compare
    :param img: the object Image to compare
    """
    if (metric.data.shape) != (img.data.shape):
        raise ValueError(f"The input metric's dimension is {metric.data.shape}, "
    f"but the input image's dimension is {img.data.shape}, which doesn't match.")