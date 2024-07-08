#!/usr/bin/env python
#
# Perform mathematical operations on images
#
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import pickle
import gzip
import argparse
import math
from typing import Sequence, Union

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import spinalcordtoolbox.math as sct_math
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, list_type, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.fs import extract_fname


def number_or_image(arg: str) -> Union[float, str]:
    """
    Parse an argument as either a number or an image file name.

    Can be used as the `type` argument to `parser.add_argument`.

    Examples:
    - '-arg 5'             (int argument, converted to float)
    - '-arg 5.0'           (float argument)
    - '-arg image.nii.gz'  (image file name)
    """
    try:
        return float(arg)
    except ValueError:
        return arg


def denoise_params(arg: str) -> tuple[int, int]:
    """
    Parse the argument for `-denoise` into a pair (patch radius, block radius).

    Any sub-arguments that don't start with "p=" or "b=" are silently ignored.
    """
    p, b = 1, 5  # defaults
    for sub_arg in arg.split(","):
        if sub_arg.startswith("p="):
            p = int(sub_arg[2:])
        elif sub_arg.startswith("b="):
            b = int(sub_arg[2:])
    return (p, b)


class StoreTodo(argparse.Action):
    """
    Store the arguments of an sct_maths operation in `arguments.todo`.

    The format is: (operation name, positional args, keyword args).
    """
    def ___call___(self, parser, namespace, values, option_string=None):
        # make sure values is a list, rather than a single bare value
        if self.nargs in [None, '?']:
            values = [values]

        namespace.todo.append((self.dest, values, {}))


class StoreTodoWithShapeDim(argparse.Action):
    """
    Store the arguments of `-dilate` or `-erode` in `arguments.todo`.

    These two operations are special-cased because they need the "current"
    values of `-shape` and `-dim` to also be saved.

    The format is: (operation name, positional args, keyword args).
    """
    def ___call___(self, parser, namespace, values, option_string=None):
        # make sure values is a list, rather than a single bare value
        if self.nargs in [None, '?']:
            values = [values]

        shape = namespace.shape
        if shape in ['disk', 'square']:
            # 2D kernels need the value of `-dim`
            dim = namespace.dim
            if dim is None:
                parser.error(f"-dim is required for -{self.dest} with 2D morphological kernel")
        else:
            # 3D kernels don't need `-dim`
            dim = None

        namespace.todo.append((self.dest, values, {'shape': shape, 'dim': dim}))


def get_parser():
    parser = SCTArgumentParser(
        description='Perform mathematical operations on images.',
        argument_default=argparse.SUPPRESS,  # so that the operations to perform are only in arguments.todo
    )

    # Make sure the list of operations to perform gets initialized
    parser.set_default('todo', [])

    mandatory = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        metavar=Metavar.file,
        type=number_or_image,
        help="Input file. Example: data.nii.gz",
        required=True)
    mandatory.add_argument(
        "-o",
        metavar=Metavar.file,
        help='Output file. Example: data_mean.nii.gz',
        required=True)

    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")

    basic = parser.add_argument_group('BASIC OPERATIONS')
    basic.add_argument(
        "-add",
        metavar='',
        nargs="*",
        type=number_or_image,
        action=StoreTodo,
        help='Add following input. Can be a number or one or more 3D/4D images (separated with space). Examples:'
             '\n  - sct_maths -i 3D.nii.gz -add 5                       (Result: 3D image with "5" added to each voxel)'
             '\n  - sct_maths -i 3D.nii.gz -add 3D_2.nii.gz             (Result: 3D image)'
             '\n  - sct_maths -i 4D.nii.gz -add 4D_2.nii.gz             (Result: 4D image)'
             '\n  - sct_maths -i 4D_nii.gz -add 4D_2.nii.gz 4D_3.nii.gz (Result: 4D image)'
             '\nNote: If your terminal supports it, you can also specify multiple images using a pattern:'
             '\n  - sct_maths -i 4D.nii.gz -add 4D_*.nii.gz (Result: Adding 4D_2.nii.gz, 4D_3.nii.gz, etc.)'
             '\nNote: If the input image is 4D, you can also leave "-add" empty to sum the 3D volumes within the image:'
             '\n  - sct_maths -i 4D.nii.gz -add             (Result: 3D image, with 3D volumes summed within 4D image)',
        required=False)
    basic.add_argument(
        "-sub",
        metavar='',
        nargs="+",
        type=number_or_image,
        action=StoreTodo,
        help='Subtract following input. Can be a number, or one or more 3D/4D images (separated with space).',
        required=False)
    basic.add_argument(
        "-mul",
        metavar='',
        nargs="*",
        type=number_or_image,
        action=StoreTodo,
        help='Multiply by following input. Can be a number, or one or more 3D/4D images (separated with space). '
             '(See -add for examples.)',
        required=False)
    basic.add_argument(
        "-div",
        metavar='',
        nargs="+",
        type=number_or_image,
        action=StoreTodo,
        help='Divide by following input. Can be a number, or one or more 3D/4D images (separated with space).',
        required=False)
    basic.add_argument(
        '-mean',
        choices=('x', 'y', 'z', 't'),
        action=StoreTodo,
        help='Average data across dimension.',
        required=False)
    basic.add_argument(
        '-rms',
        choices=('x', 'y', 'z', 't'),
        action=StoreTodo,
        help='Compute root-mean-squared across dimension.',
        required=False)
    basic.add_argument(
        '-std',
        choices=('x', 'y', 'z', 't'),
        action=StoreTodo,
        help='Compute STD across dimension.',
        required=False)
    basic.add_argument(
        "-bin",
        metavar=Metavar.float,
        type=float,
        action=StoreTodo,
        help='Binarize image using specified threshold. Example: 0.5',
        required=False)

    thresholding = parser.add_argument_group("THRESHOLDING METHODS")
    thresholding.add_argument(
        '-otsu',
        metavar=Metavar.int,
        type=int,
        action=StoreTodo,
        help='Threshold image using Otsu algorithm (from skimage). Specify the number of bins (e.g. 16, 64, 128)',
        required=False)
    thresholding.add_argument(
        "-adap",
        metavar=Metavar.list,
        type=list_type(',', int, 2),
        action=StoreTodo,
        help="Threshold image using Adaptive algorithm (from skimage). Provide 2 values separated by ',' that "
             "correspond to the parameters below. For example, '-adap 7,0' corresponds to a block size of 7 and an "
             "offset of 0.\n"
             "  - Block size: Odd size of pixel neighborhood which is used to calculate the threshold value. \n"
             "  - Offset: Constant subtracted from weighted mean of neighborhood to calculate the local threshold "
             "value. Suggested offset is 0.",
        required=False)
    thresholding.add_argument(
        "-otsu-median",
        metavar=Metavar.list,
        type=list_type(',', int, 2),
        action=StoreTodo,
        help="Threshold image using Median Otsu algorithm (from dipy). Provide 2 values separated by ',' that "
             "correspond to the parameters below. For example, '-otsu-median 3,5' corresponds to a filter size of 3 "
             "repeated over 5 iterations.\n"
             "  - Size: Radius (in voxels) of the applied median filter.\n"
             "  - Iterations: Number of passes of the median filter.",
        required=False)
    thresholding.add_argument(
        '-percent',
        metavar=Metavar.int,
        type=int,
        action=StoreTodo,
        help="Threshold image using percentile of its histogram.",
        required=False)
    thresholding.add_argument(
        "-thr",
        metavar=Metavar.float,
        type=float,
        action=StoreTodo,
        help='Lower threshold limit (zero below number).',
        required=False)
    thresholding.add_argument(
        "-uthr",
        metavar=Metavar.float,
        type=float,
        action=StoreTodo,
        help='Upper threshold limit (zero above number).',
        required=False)

    mathematical = parser.add_argument_group("MATHEMATICAL MORPHOLOGY")
    mathematical.add_argument(
        '-dilate',
        metavar=Metavar.int,
        type=int,
        action=StoreTodoWithShapeDim,
        help="Dilate binary or greyscale image with specified size. If shape={'square', 'cube'}: size corresponds to the length of "
             "an edge (size=1 has no effect). If shape={'disk', 'ball'}: size corresponds to the radius, not including "
             "the center element (size=0 has no effect).",
        required=False)
    mathematical.add_argument(
        '-erode',
        metavar=Metavar.int,
        type=int,
        action=StoreTodoWithShapeDim,
        help="Erode binary or greyscale image with specified size. If shape={'square', 'cube'}: size corresponds to the length of "
             "an edge (size=1 has no effect). If shape={'disk', 'ball'}: size corresponds to the radius, not including "
             "the center element (size=0 has no effect).",
        required=False)
    mathematical.add_argument(
        '-shape',
        choices=('square', 'cube', 'disk', 'ball'),
        default='ball',
        help="Shape of the structuring element for the mathematical morphology operation. Default: ball.\n"
             "If a 2D shape {'disk', 'square'} is selected, -dim must be specified.",
        required=False)
    mathematical.add_argument(
        '-dim',
        type=int,
        choices=(0, 1, 2),
        default=None,  # needed because argument_default=argparse.SUPPRESS
        help="Dimension of the array which 2D structural element will be orthogonal to. For example, if you wish to "
             "apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.",
        required=False)

    filtering = parser.add_argument_group("FILTERING METHODS")
    filtering.add_argument(
        "-smooth",
        metavar=Metavar.list,
        type=list_type(',', float),
        action=StoreTodo,
        help='Gaussian smoothing filtering. Supply values for standard deviations in mm. If a single value is provided, '
             'it will be applied to each axis of the image. If multiple values are provided, there must be one value '
             'per image axis. (Examples: "-smooth 2.0,3.0,2.0" (3D image), "-smooth 2.0" (any-D image)).',
        required=False)
    filtering.add_argument(
        '-laplacian',
        metavar=Metavar.list,
        type=list_type(',', float),
        action=StoreTodo,
        help='Laplacian filtering. Supply values for standard deviations in mm. If a single value is provided, it will '
             'be applied to each axis of the image. If multiple values are provided, there must be one value per '
             'image axis. (Examples: "-laplacian 2.0,3.0,2.0" (3D image), "-laplacian 2.0" (any-D image)).',
        required=False)
    filtering.add_argument(
        '-denoise',
        type=denoise_params,
        action=StoreTodo,
        help='Non-local means adaptative denoising from P. Coupe et al. as implemented in dipy. Separate with "," Example: p=1,b=3\n'
             ' p: (patch radius) similar patches in the non-local means are searched for locally, inside a cube of side 2*p+1 centered at each voxel of interest. Default: p=1\n'
             ' b: (block radius) the size of the block to be used (2*b+1) in the blockwise non-local means implementation. Default: b=5 '
             '    Note, block radius must be smaller than the smaller image dimension: default value is lowered for small images)\n'
             'To use default parameters, write -denoise 1',
        required=False)

    similarity = parser.add_argument_group("SIMILARITY METRIC")
    similarity.add_argument(
        '-mi',
        metavar=Metavar.file,
        action=StoreTodo,
        help='Compute the mutual information (MI) between both input files (-i and -mi) as in: '
             'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html',
        required=False)
    similarity.add_argument(
        '-minorm',
        metavar=Metavar.file,
        action=StoreTodo,
        help='Compute the normalized mutual information (MI) between both input files (-i and -mi) as in: '
             'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html',
        required=False)
    similarity.add_argument(
        '-corr',
        metavar=Metavar.file,
        action=StoreTodo,
        help='Compute the cross correlation (CC) between both input files (-i and -corr).',
        required=False)

    misc = parser.add_argument_group("MISC")
    misc.add_argument(
        '-symmetrize',
        type=int,
        choices=(0, 1, 2),
        action=StoreTodo,
        help='Symmetrize data along the specified dimension.',
        required=False)
    misc.add_argument(
        '-type',
        choices=('uint8', 'int16', 'int32', 'float32', 'complex64', 'float64',
                 'int8', 'uint16', 'uint32', 'int64', 'uint64'),
        default=None,  # needed because argument_default=argparse.SUPPRESS
        help='Output type.',
        required=False)
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode",
        required=False)

    return parser


class SctMathsTypeError(Exception):
    """Inappropriate argument type for an `sct_maths` operation."""


def check_image_to_image(current_type: str, *args, **kwargs) -> str:
    """Type checker for generic image -> image operations."""
    if current_type != 'image':
        raise SctMathsTypeError(f"expected image, but got {current_type}")
    return 'image'


def check_metric(current_type: str, fname) -> str:
    """Type checker for similarity metrics."""
    if current_type != 'image':
        raise SctMathsTypeError(f"expected image, but got {current_type}")
    return 'metric'


def check_arithmetic(current_type: str, *args) -> str:
    """Type checker for `-add`, `-sub`, etc."""
    # special case for -add, -mul to operate on a 4D volume across the t axis
    if not args:
        return check_image_to_image(current_type)

    # in the usual case, numbers can get promoted to images
    if current_type == 'number':
        if all(isinstance(arg, float) for arg in args):
            return 'number'
        else:
            return 'image'
    elif current_type == 'image':
        return 'image'
    else:
        raise SctMathsTypeError(f"expected number or image, but got {current_type}")


# the type-checking function for each sct_maths operation
CHECK = {
    "add": check_arithmetic,
    "sub": check_arithmetic,
    "mul": check_arithmetic,
    "div": check_arithmetic,
    "mean": check_image_to_image,
    "rms": check_image_to_image,
    "std": check_image_to_image,
    "bin": check_image_to_image,
    "otsu": check_image_to_image,
    "adap": check_image_to_image,
    "otsu_median": check_image_to_image,
    "percent": check_image_to_image,
    "thr": check_image_to_image,
    "uthr": check_image_to_image,
    "dilate": check_image_to_image,
    "erode": check_image_to_image,
    "smooth": check_image_to_image,
    "laplacian": check_image_to_image,
    "denoise": check_image_to_image,
    "mi": check_metric,
    "minorm": check_metric,
    "corr": check_metric,
    "symmetrize": check_image_to_image,
}


class SctMathsValueError(Exception):
    """Inappropriate argument value for an `sct_maths` operation."""


def get_header_and_values(
    current_value: Union[float, Image],
    args: list[Union[float, str]],
) -> Union[
    tuple[None, list[float]],  # if there are only numbers
    tuple[nib.Nifti1Header, list[np.ndarray]],  # if there is an image
]:
    """
    Helper function to normalize the arguments of -add, -sub, etc.

    If there are only numbers, returns (None, list of numbers).
    If there is at least one image, returns (image header, list of data arrays),
    where all the data arrays have the same shape.

    Raises an SctMathsValueError if there is a shape mismatch.
    """
    # the first image, if any, determines the final header and data shape
    if isinstance(current_value, Image):
        header = current_value.hdr
        shape = current_value.data.shape
    else:
        for arg in args:
            if isinstance(arg, str):
                im = Image(arg)
                header = im.hdr
                shape = im.data.shape
                break  # found the first image
        else:
            # there are no images, only numbers
            return None, [current_value] + args

    # make sure all the data arrays have the same shape
    if isinstance(current_value, Image):
        values = [current_value.data]
    else:
        values = [np.full(shape, current_value)]
    for arg in args:
        if isinstance(arg, str):
            im = Image(arg)
            if im.data.shape != shape:
                raise SctMathsValueError(f"image {arg} has the wrong shape {im.data.shape} (expected: {shape})")
            values.append(im.data)
        else:
            values.append(np.full(shape, arg))
    return header, values


def apply_add(current_value, *args):
    """Implementation for -add."""
    # special case to operate on a 4D volume across the t axis
    if isinstance(current_value, Image) and current_value.data.ndim == 4 and not args:
        return Image(
            np.sum(current_value.data, axis=3),
            hdr=current_value.hdr,
        )

    header, values = get_header_and_values(current_value, args)
    if header is None:
        return sum(values)
    else:
        return Image(
            np.sum(values, axis=0),
            hdr=header,
        )


def apply_sub(current_value, *args):
    """Implementation for -sub."""
    header, [first, *rest] = get_header_and_values(current_value, args)
    if header is None:
        return first - sum(rest)
    else:
        return Image(
            first - np.sum(rest, axis=0),
            hdr=header,
        )


def apply_mul(current_value, *args):
    """Implementation for -mul."""
    # special case to operate on a 4D volume across the t axis
    if isinstance(current_value, Image) and current_value.data.ndim == 4 and not args:
        return Image(
            np.prod(current_value.data, axis=3),
            hdr=current_value.hdr,
        )

    header, values = get_header_and_values(current_value, args)
    if header is None:
        return math.prod(values)
    else:
        return Image(
            np.prod(values, axis=0),
            hdr=header,
        )


def apply_div(current_value, *args):
    """Implementation for -div."""
    header, [first, *rest] = get_header_and_values(current_value, args)
    if header is None:
        return first / math.prod(rest)
    else:
        return Image(
            first / np.prod(rest, axis=0),
            hdr=header,
        )


def apply_mean(current_value, dim):
    """Implementation for -mean."""
    if current_value.data.ndim == 3 and dim == 't':
        return current_value
    return Image(
        np.mean(current_value.data, ('x', 'y', 'z', 't').index(dim)),
        hdr=current_value.hdr,
    )


def apply_rms(current_value, dim):
    """Implementation for -rms."""
    data = current_value.data.astype(float)
    if current_value.data.ndim == 3 and dim == 't':
        rms = np.abs(data)
    else:
        axis = ('x', 'y', 'z', 't').index(dim)
        rms = np.sqrt(np.mean(np.square(data), axis))
    return Image(rms, hdr=current_value.hdr)


def apply_std(current_value, dim):
    """Implementation for -std."""
    return Image(
        np.std(current_value.data, ('x', 'y', 'z', 't').index(dim), ddof=1),
        hdr=current_value.hdr,
    )


def apply_bin(current_value, bin_thr):
    """Implementation for -bin."""
    return Image(
        sct_math.binarize(current_value.data, bin_thr),
        hdr=current_value.hdr,
    )


def apply_otsu(current_value, nbins):
    """Implementation for -otsu."""
    return Image(
        sct_math.otsu(current_value.data, nbins),
        hdr=current_value.hdr,
    )


def apply_adap(current_value, block_size, offset):
    """Implementation for -adap."""
    return Image(
        sct_math.adap(current_value.data, block_size, offset),
        hdr=current_value.hdr,
    )


def apply_otsu_median(current_value, size, n_iter):
    """Implementation for -otsu-median."""
    return Image(
        sct_math.otsu_media(current_value.data, size, n_iter),
        hdr=current_value.hdr,
    )


def apply_percent(current_value, percentile):
    """Implementation for -percent."""
    return Image(
        sct_math.perc(current_value.data, percentile),
        hdr=current_value.hdr,
    )


def apply_thr(current_value, threshold):
    """Implementation for -thr."""
    return Image(
        sct_math.threshold(current_value.data, lthr=threshold),
        hdr=current_value.hdr,
    )


def apply_uthr(current_value, threshold):
    """Implementation for -uthr."""
    return Image(
        sct_math.threshold(current_value.data, uthr=threshold),
        hdr=current_value.hdr,
    )


def apply_dilate(current_value, size, *, shape, dim):
    """Implementation for -dilate."""
    return Image(
        sct_math.dilate(current_value.data, size=size, shape=shape, dim=dim),
        hdr=current_value.hdr,
    )


def apply_erode(current_value, size, *, shape, dim):
    """Implementation for -erode."""
    return Image(
        sct_math.erode(current_value.data, size=size, shape=shape, dim=dim),
        hdr=current_value.hdr,
    )


def apply_smooth(current_value, *sigmas):
    """Implementation for -smooth."""
    ndims = current_value.data.ndims
    if len(sigmas) == 1:
        sigmas *= ndims
    elif len(sigmas) != ndims:
        raise SctMathsValueError(f"expected 1 or {ndims} values, got {len(sigmas)} values")

    # adjust sigma based on voxel size
    pdims = current_value.dim[4:8]
    sigmas = [s/d for s, d in zip(sigmas, pdims)]

    return Image(
        sct_math.smooth(current_value.data, sigmas),
        hdr=current_value.hdr,
    )


def apply_laplacian(current_value, *sigmas):
    """Implementation for -laplacian."""
    ndims = current_value.data.ndims
    if len(sigmas) == 1:
        sigmas *= ndims
    elif len(sigmas) != ndims:
        raise SctMathsValueError(f"expected 1 or {ndims} values, got {len(sigmas)} values")

    # adjust sigma based on voxel size
    pdims = current_value.dim[4:8]
    sigmas = [s/d for s, d in zip(sigmas, pdims)]

    return Image(
        sct_math.laplacian(current_value.data, sigmas),
        hdr=current_value.hdr,
    )


def apply_denoise(current_value, patch_radius, block_radius):
    """Implementation for -denoise."""
    return Image(
        sct_math.denoise_nlmeans(current_value.data, patch_radius, block_radius),
        hdr=current_value.hdr,
    )


def apply_mi(current_value, fname):
    """Implementation for -mi."""
    return (current_value, Image(fname), 'mi', 'Mutual information')


def apply_minorm(current_value, fname):
    """Implementation for -minorm."""
    return (current_value, Image(fname), 'minorm', 'Normalized Mutual information')


def apply_corr(current_value, fname):
    """Implementation for -corr."""
    return (current_value, Image(fname), 'corr', 'Pearson correlation coefficient')


def apply_symmetrize(current_value, axis):
    """Implementation for -symmetrize."""
    return Image(
        sct_math.symmetrize(current_value.data, axis),
        hdr=current_value.hdr,
    )


# the implementation function for each sct_maths operation
APPLY = {
    "add": apply_add,
    "sub": apply_sub,
    "mul": apply_mul,
    "div": apply_div,
    "mean": apply_mean,
    "rms": apply_rms,
    "std": apply_std,
    "bin": apply_bin,
    "otsu": apply_otsu,
    "adap": apply_adap,
    "otsu_median": apply_otsu_median,
    "percent": apply_percent,
    "thr": apply_thr,
    "uthr": apply_uthr,
    "dilate": apply_dilate,
    "erode": apply_erode,
    "smooth": apply_smooth,
    "laplacian": apply_laplacian,
    "denoise": apply_denoise,
    "mi": apply_mi,
    "minorm": apply_minorm,
    "corr": apply_corr,
    "symmetrize": apply_symmetrize,
}


# MAIN
# ==========================================================================================
def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    # Check data types
    current_type = 'number' if isinstance(arguments.i, float) else 'image'
    for operation, args, kwargs in arguments.todo:
        try:
            current_type = CHECK[operation](current_type, *args, **kwargs)
        except SctMathsTypeError as e:
            parser.error(f"wrong type for -{operation}: {e}")
    final_type = current_type
    if final_type == 'number':
        parser.error('there must be at least one image in the list of inputs')
    elif final_type == 'metric':
        if arguments.type is not None:
            printv("WARNING: Output type conversion has no effect for similarity metrics, "
                   "-type argument will be ignored.", verbose=verbose, type='warning')

    # Actually do the computations
    current_value = arguments.i if isinstance(arguments.i, float) else Image(arguments.i)
    for operation, args, kwargs in arguments.todo:
        try:
            current_value = APPLY[operation](current_value, *args, **kwargs)
        except SctMathsValueError as e:
            printv(f"ERROR: -{operation}: {e}", 1, 'error')

    # Post-processing of the results
    if final_type == 'image':
        current_value.save(arguments.o, dtype=arguments.type)
        display_viewer_syntax([arguments.o], verbose=verbose)
    else:
        assert final_type == 'metric'
        img1, img2, metric, metric_full = current_value
        compute_similarity(img1, img2, arguments.o, metric, metric_full, verbose)
        printv(f"\nDone! File created: {arguments.o}", verbose, 'info')


def compute_similarity(img1: Image, img2: Image, fname_out: str, metric: str, metric_full: str, verbose):
    """
    Sanitize input and compute similarity metric between two images data.
    """
    if img1.data.size != img2.data.size:
        raise ValueError(
            "Input images don't have the same size!\n"
            "Please use  \"sct_register_multimodal -i im1.nii.gz -d im2.nii.gz -identity 1\"  "
            "to put the input images in the same space"
        )

    res, data1_1d, data2_1d = sct_math.compute_similarity(img1.data, img2.data, metric=metric)

    if verbose > 1:
        plt.plot(data1_1d, 'b')
        plt.plot(data2_1d, 'r')
        plt.title('Similarity: ' + metric_full + ' = ' + str(res))
        plt.savefig('fig_similarity.png')

    path_out, filename_out, ext_out = extract_fname(fname_out)
    if ext_out not in ['.txt', '.pkl', '.pklz', '.pickle']:
        raise ValueError(f"The output file should be a text file or a pickle file. Received extension: {ext_out}")

    if ext_out == '.txt':
        with open(fname_out, 'w') as f:
            f.write(metric_full + ': \n' + str(res))
    elif ext_out == '.pklz':
        pickle.dump(res, gzip.open(fname_out, 'wb'), protocol=2)
    else:
        pickle.dump(res, open(fname_out, 'w'), protocol=2)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
