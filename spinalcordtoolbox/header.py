#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""
Utility functions for processing the headers of Nifti images.
"""

__author__ = "Joshua Newton"
__email__ = "joshuacwnewton@gmail.com"
__copyright__ = "Copyright (c) 2021 Polytechnique Montreal <www.neuro.polymtl.ca>"

# --------------------------------------------------------------------------------

import math

from contrib import fslhd

DISPLAY_FORMATS = ('sct', 'fslhd', 'nibabel')


def format_header(image, output_format='sct'):
    """
    Generate a string with formatted header fields for pretty-printing.

    :param image: Input image to take header from.
    :param output_format: Specify how to format the output header.
    """
    if output_format == 'sct':
        formatted_fields = _apply_sct_header_formatting(fslhd.generate_nifti_fields(image))
        aligned_string = _align_dict(formatted_fields)
    elif output_format == 'fslhd':
        formatted_fields = fslhd.generate_nifti_fields(image)
        aligned_string = _align_dict(formatted_fields)
    elif output_format == 'nibabel':
        formatted_fields = {k: v[()] for k, v in dict(image.header).items()}
        aligned_string = _align_dict(formatted_fields, use_tabs=False, delimiter=": ")
    else:
        raise ValueError(f"Can't format header using '{output_format}' format. Available formats: {DISPLAY_FORMATS}")

    return aligned_string


def _apply_sct_header_formatting(fslhd_fields):
    """
    Tweak fslhd's header fields using SCT's visual preferences.

    :param fslhd_fields: Dict with fslhd's header fields.
    :return modified_fields: Dict with modified header fields.
    """
    modified_fields = {}
    dim, pixdim = [], []
    for key, value in fslhd_fields.items():
        # Replace split dim fields with one-line dim field
        if key.startswith('dim'):
            dim.append(value)
            if key == 'dim7':
                modified_fields['dim'] = dim
        # Replace split pixdim fields with one-line pixdim field
        elif key.startswith('pixdim'):
            pixdim.append(float(value))
            if key == 'pixdim7':
                modified_fields['pixdim'] = pixdim
        # Leave all other fields
        else:
            modified_fields[key] = value

    return modified_fields


def _align_dict(dictionary, use_tabs=True, delimiter=""):
    """
    Create a string with aligned padding from a dict's keys and values.

    :param dictionary: Variable of type dict.
    :param use_tabs: Whether to use tabs instead of spaces for padding.

    :return: String containing padded dict key/values.
    """
    len_max = max([len(str(name)) for name in dictionary.keys()]) + 2
    out = []
    for k, v in dictionary.items():
        if use_tabs:
            len_max = int(8 * round(float(len_max)/8))  # Round up to the nearest 8 to align with tab stops
            padding = "\t" * math.ceil((len_max - len(k))/8)
        else:
            padding = " " * (len_max - len(k))
        out.append(f"{k}{padding}{delimiter}{v}")
    return '\n'.join(out)
