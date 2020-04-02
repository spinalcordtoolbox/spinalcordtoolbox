#!/usr/bin/env python
#########################################################################################
#
# Perform mathematical operations on images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import division, absolute_import

import os
import sys
import numpy as np
import argparse

import spinalcordtoolbox as sct
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import Metavar, SmartFormatter
import spinalcordtoolbox.math

from sct_utils import printv, extract_fname, display_viewer_syntax, init_sct


ALMOST_ZERO = 0.000000001


def get_parser():

    parser = argparse.ArgumentParser(
        description='Perform mathematical operations on images. Some inputs can be either a number or a 4d image or '
                    'several 3d images separated with ","',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatory = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        metavar=Metavar.file,
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
        nargs="+",
        help='Add following input. Can be a number or multiple images (separated with space).',
        required=False)
    basic.add_argument(
        "-sub",
        metavar='',
        nargs="+",
        help='Subtract following input. Can be a number or an image.',
        required=False)
    basic.add_argument(
        "-mul",
        metavar='',
        nargs="+",
        help='Multiply by following input. Can be a number or multiple images (separated with space).',
        required=False)
    basic.add_argument(
        "-div",
        metavar='',
        nargs="+",
        help='Divide by following input. Can be a number or an image.',
        required=False)
    basic.add_argument(
        '-mean',
        help='Average data across dimension.',
        required=False,
        choices=('x', 'y', 'z', 't'))
    basic.add_argument(
        '-rms',
        help='Compute root-mean-squared across dimension.',
        required=False,
        choices=('x', 'y', 'z', 't'))
    basic.add_argument(
        '-std',
        help='Compute STD across dimension.',
        required=False,
        choices=('x', 'y', 'z', 't'))
    basic.add_argument(
        "-bin",
        type=float,
        metavar=Metavar.float,
        help='Binarize image using specified threshold. Example: 0.5',
        required=False)

    thresholding = parser.add_argument_group("THRESHOLDING METHODS")
    thresholding.add_argument(
        '-otsu',
        type=int,
        metavar=Metavar.int,
        help='Threshold image using Otsu algorithm (from skimage). Specify the number of bins (e.g. 16, 64, 128)',
        required=False)
    thresholding.add_argument(
        "-adap",
        metavar=Metavar.list,
        help="R|Threshold image using Adaptive algorithm (from skimage). Separate following arguments with ',':"
             "\n Block size: Odd size of pixel neighborhood which is used to calculate the threshold value (e.g. 3, 7, 21, ...)"
             "\n Offset: Constant subtracted from weighted mean of neighborhood to calculate the local threshold value. Suggested offset is 0.",
        required=False)
    thresholding.add_argument(
        "-otsu-median",
        help="R|Threshold image using Median Otsu algorithm. Separate following arguments with ',':"
             "\n Size of the median filter (e.g. 2, 3)"
             "\n Number of iterations (e.g. 3, 4, 5)\n",
        metavar=Metavar.list,
        required=False)
    thresholding.add_argument(
        '-percent',
        type=int,
        help="Threshold image using percentile of its histogram.",
        metavar=Metavar.int,
        required=False)
    thresholding.add_argument(
        "-thr",
        type=float,
        help='Use following number to threshold image (zero below number).',
        metavar=Metavar.float,
        required=False)

    mathematical = parser.add_argument_group("MATHEMATICAL MORPHOLOGY")
    mathematical.add_argument(
        '-dilate',
        type=int,
        metavar=Metavar.int,
        help="Dilate binary or greyscale image with specified size. If shape={'square', 'cube'}: size corresponds to the length of "
             "an edge (size=1 has no effect). If shape={'disk', 'ball'}: size corresponds to the radius, not including "
             "the center element (size=0 has no effect).",
        required=False)
    mathematical.add_argument(
        '-erode',
        type=int,
        metavar=Metavar.int,
        help="Erode binary or greyscale image with specified size. If shape={'square', 'cube'}: size corresponds to the length of "
             "an edge (size=1 has no effect). If shape={'disk', 'ball'}: size corresponds to the radius, not including "
             "the center element (size=0 has no effect).",
        required=False)
    mathematical.add_argument(
        '-shape',
        help="Shape of the structuring element for the mathematical morphology operation. Default: ball.",
        required=False,
        choices=('square', 'cube', 'disk', 'ball'),
        default='ball')
    mathematical.add_argument(
        '-dim',
        type=int,
        metavar=Metavar.int,
        help="Dimension of the array which 2D structural element will be orthogonal to. For example, if you wish to "
             "apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.",
        required=False,
        choices=(0, 1, 2))

    filtering = parser.add_argument_group("FILTERING METHODS")
    filtering.add_argument(
        "-smooth",
        metavar='',
        help='Gaussian smoothing filter with specified standard deviations in mm for each axis (Example: 2,2,1) or '
             'single value for all axis (Example: 2).',
        required = False)
    filtering.add_argument(
        '-laplacian',
        nargs="+",
        metavar='',
        help='Laplacian filtering with specified standard deviations in mm for all axes (Example: 2).',
        required = False)
    filtering.add_argument(
        '-denoise',
        help='R|Non-local means adaptative denoising from P. Coupe et al. as implemented in dipy. Separate with ". Example: p=1,b=3\n'
             ' p: (patch radius) similar patches in the non-local means are searched for locally, inside a cube of side 2*p+1 centered at each voxel of interest. Default: p=1\n'
             ' b: (block radius) the size of the block to be used (2*b+1) in the blockwise non-local means implementation. Default: b=5 '
             '    Note, block radius must be smaller than the smaller image dimension: default value is lowered for small images)\n'
             'To use default parameters, write -denoise 1',
        required=False)

    similarity = parser.add_argument_group("SIMILARITY METRIC")
    similarity.add_argument(
        '-mi',
        metavar=Metavar.file,
        help='Compute the mutual information (MI) between both input files (-i and -mi) as in: '
             'http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html',
        required=False)
    similarity.add_argument(
        '-minorm',
        metavar=Metavar.file,
        help='Compute the normalized mutual information (MI) between both input files (-i and -mi) as in: '
             'http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html',
        required=False)
    similarity.add_argument(
        '-corr',
        metavar=Metavar.file,
        help='Compute the cross correlation (CC) between both input files (-i and -cc).',
        required=False)

    misc = parser.add_argument_group("MISC")
    misc.add_argument(
        '-symmetrize',
        type=int,
        help='Symmetrize data along the specified dimension.',
        required=False,
        choices=(0, 1, 2))
    misc.add_argument(
        '-type',
        required=False,
        help='Output type.',
        choices=('uint8', 'int16', 'int32', 'float32', 'complex64', 'float64', 'int8', 'uint16', 'uint32', 'int64',
                 'uint64'))
    misc.add_argument(
        "-v",
        type=int,
        help="Verbose. 0: nothing. 1: basic. 2: extended.",
        required=False,
        default=1,
        choices=(0, 1, 2))

    return parser


# MAIN
# ==========================================================================================
def main(args=None):
    """
    Main function
    :param args:
    :return:
    """
    dim_list = ['x', 'y', 'z', 't']

    # Get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    parser = get_parser()
    arguments = parser.parse_args(args=args)
    fname_in = arguments.i
    fname_out = arguments.o
    verbose = arguments.v
    init_sct(log_level=verbose, update=True)  # Update log level
    if '-type' in arguments:
        output_type = arguments.type
    else:
        output_type = None

    # Open file(s)
    im = Image(fname_in)
    data = im.data  # 3d or 4d numpy array
    dim = im.dim

    # run command
    if arguments.otsu is not None:
        param = arguments.otsu
        data_out = otsu(data, param)

    elif arguments.adap is not None:
        param = convert_list_str(arguments.adap, "int")
        data_out = adap(data, param[0], param[1])

    elif arguments.otsu_median is not None:
        param = convert_list_str(arguments.otsu_median, "int")
        data_out = otsu_median(data, param[0], param[1])

    elif arguments.thr is not None:
        param = arguments.thr
        data_out = threshold(data, param)

    elif arguments.percent is not None:
        param = arguments.percent
        data_out = perc(data, param)

    elif arguments.bin is not None:
        bin_thr = arguments.bin
        data_out = binarise(data, bin_thr=bin_thr)

    elif arguments.add is not None:
        from numpy import sum
        data2 = get_data_or_scalar(arguments.add, data)
        data_concat = concatenate_along_4th_dimension(data, data2)
        data_out = sum(data_concat, axis=3)

    elif arguments.sub is not None:
        data2 = get_data_or_scalar(arguments.sub, data)
        data_out = data - data2

    elif arguments.laplacian is not None:
        sigmas = convert_list_str(arguments.laplacian, "float")
        if len(sigmas) == 1:
            sigmas = [sigmas for i in range(len(data.shape))]
        elif len(sigmas) != len(data.shape):
            printv(parser.error('ERROR: -laplacian need the same number of inputs as the number of image dimension OR only one input'))
        # adjust sigma based on voxel size
        sigmas = [sigmas[i] / dim[i + 4] for i in range(3)]
        # smooth data
        data_out = laplacian(data, sigmas)

    elif arguments.mul is not None:
        from numpy import prod
        data2 = get_data_or_scalar(arguments.mul, data)
        data_concat = concatenate_along_4th_dimension(data, data2)
        data_out = prod(data_concat, axis=3)

    elif arguments.div is not None:
        from numpy import divide
        data2 = get_data_or_scalar(arguments.div, data)
        data_out = divide(data, data2)

    elif arguments.mean is not None:
        from numpy import mean
        dim = dim_list.index(arguments.mean)
        if dim + 1 > len(np.shape(data)):  # in case input volume is 3d and dim=t
            data = data[..., np.newaxis]
        data_out = mean(data, dim)

    elif arguments.rms is not None:
        from numpy import mean, sqrt, square
        dim = dim_list.index(arguments.rms)
        if dim + 1 > len(np.shape(data)):  # in case input volume is 3d and dim=t
            data = data[..., np.newaxis]
        data_out = sqrt(mean(square(data.astype(float)), dim))

    elif arguments.std is not None:
        from numpy import std
        dim = dim_list.index(arguments.std)
        if dim + 1 > len(np.shape(data)):  # in case input volume is 3d and dim=t
            data = data[..., np.newaxis]
        data_out = std(data, dim, ddof=1)

    elif arguments.smooth is not None:
        sigmas = convert_list_str(arguments.smooth, "float")
        if len(sigmas) == 1:
            sigmas = [sigmas[0] for i in range(len(data.shape))]
        elif len(sigmas) != len(data.shape):
            printv(parser.error('ERROR: -smooth need the same number of inputs as the number of image dimension OR only one input'))
        # adjust sigma based on voxel size
        sigmas = [sigmas[i] / dim[i + 4] for i in range(3)]
        # smooth data
        data_out = smooth(data, sigmas)

    elif arguments.dilate is not None:
        data_out = sct.math.dilate(data, size=arguments.dilate, shape=arguments.shape, dim=arguments.dim)

    elif arguments.erode is not None:
        data_out = sct.math.erode(data, size=arguments.erode, shape=arguments.shape, dim=arguments.dim)

    elif arguments.denoise is not None:
        # parse denoising arguments
        p, b = 1, 5  # default arguments
        list_denoise = (arguments.denoise).split(",")
        for i in list_denoise:
            if 'p' in i:
                p = int(i.split('=')[1])
            if 'b' in i:
                b = int(i.split('=')[1])
        data_out = denoise_nlmeans(data, patch_radius=p, block_radius=b)

    elif arguments.symmetrize is not None:
        data_out = (data + data[list(range(data.shape[0] - 1, -1, -1)), :, :]) / float(2)

    elif arguments.mi is not None:
        # input 1 = from flag -i --> im
        # input 2 = from flag -mi
        im_2 = Image(arguments.mi)
        compute_similarity(im.data, im_2.data, fname_out, metric='mi', verbose=verbose)
        data_out = None

    elif arguments.minorm is not None:
        im_2 = Image(arguments.minorm)
        compute_similarity(im.data, im_2.data, fname_out, metric='minorm', verbose=verbose)
        data_out = None

    elif arguments.corr is not None:
        # input 1 = from flag -i --> im
        # input 2 = from flag -mi
        im_2 = Image(arguments.corr)
        compute_similarity(im.data, im_2.data, fname_out, metric='corr', verbose=verbose)
        data_out = None

    # if no flag is set
    else:
        data_out = None
        printv(parser.error('ERROR: you need to specify an operation to do on the input image'))

    if data_out is not None:
        # Write output
        nii_out = Image(fname_in)  # use header of input file
        nii_out.data = data_out
        nii_out.save(fname_out, dtype=output_type)
    # TODO: case of multiple outputs
    # assert len(data_out) == n_out
    # if n_in == n_out:
    #     for im_in, d_out, fn_out in zip(nii, data_out, fname_out):
    #         im_in.data = d_out
    #         im_in.absolutepath = fn_out
    #         if "-w" in arguments:
    #             im_in.hdr.set_intent('vector', (), '')
    #         im_in.save()
    # elif n_out == 1:
    #     nii[0].data = data_out[0]
    #     nii[0].absolutepath = fname_out[0]
    #     if "-w" in arguments:
    #             nii[0].hdr.set_intent('vector', (), '')
    #     nii[0].save()
    # elif n_out > n_in:
    #     for dat_out, name_out in zip(data_out, fname_out):
    #         im_out = nii[0].copy()
    #         im_out.data = dat_out
    #         im_out.absolutepath = name_out
    #         if "-w" in arguments:
    #             im_out.hdr.set_intent('vector', (), '')
    #         im_out.save()
    # else:
    #     printv(parser.usage.generate(error='ERROR: not the correct numbers of inputs and outputs'))

    # display message
    if data_out is not None:
        display_viewer_syntax([fname_out], verbose=verbose)
    else:
        printv('\nDone! File created: ' + fname_out, verbose, 'info')


def convert_list_str(string_list, type='int'):
    """
    Receive a string and then converts it into a list of selected type.
    Example: "2,2,3" --> [2, 2, 3]
    :param string_list: List of comma-separated string
    :param type: string: int, float
    :return:
    """
    new_type_list = (string_list).split(",")
    for inew_type_list, ele in enumerate(new_type_list):
        if type is "int":
            new_type_list[inew_type_list] = int(ele)
        elif type is "float":
            new_type_list[inew_type_list] = float(ele)

    return new_type_list


def otsu(data, nbins):
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(data, nbins)
    return data > thresh


def adap(data, block_size, offset):
    from skimage.filters import threshold_local
    mask = data
    for iz in range(data.shape[2]):
        adaptive_thresh = threshold_local(data[:, :, iz], block_size, method='gaussian', offset=offset)
        mask[:, :, iz] = mask[:, :, iz] > adaptive_thresh
    return mask


def otsu_median(data, size, n_iter):
    from dipy.segment.mask import median_otsu
    data, mask = median_otsu(data, size, n_iter)
    return mask


def threshold(data, thr_value):
    data[data < thr_value] = 0
    return data


def perc(data, perc_value):
    from numpy import percentile
    perc = percentile(data, perc_value)
    return data > perc


def binarise(data, bin_thr=0):
    return data > bin_thr


def get_data(list_fname):
    """
    Get data from list of file names
    :param list_fname:
    :return: 3D or 4D numpy array.
    """
    try:
        nii = [Image(f_in) for f_in in list_fname]
    except Exception as e:
        printv(str(e), 1, 'error')  # file does not exist, exit program
    data0 = nii[0].data
    data = nii[0].data
    # check that every images have same shape
    for i in range(1, len(nii)):
        if not np.shape(nii[i].data) == np.shape(data0):
            printv('\nWARNING: shape(' + list_fname[i] + ')=' + str(np.shape(nii[i].data)) + ' incompatible with shape(' + list_fname[0] + ')=' + str(np.shape(data0)), 1, 'warning')
            printv('\nERROR: All input images must have same dimensions.', 1, 'error')
        else:
            data = concatenate_along_4th_dimension(data, nii[i].data)
    return data


def get_data_or_scalar(argument, data_in):
    """
    Get data from list of file names (scenario 1) or scalar (scenario 2)
    :param argument: list of file names of scalar
    :param data_in: if argument is scalar, use data to get np.shape
    :return: 3d or 4d numpy array
    """
    # try to convert argument in float
    try:
        # build data2 with same shape as data
        data_out = data_in[:, :, :] * 0 + float(argument[0])
    # if conversion fails, it should be a string (i.e. file name)
    except ValueError:
        data_out = get_data(argument)
    return data_out


def concatenate_along_4th_dimension(data1, data2):
    """
    Concatenate two data along 4th dimension.
    :param data1: 3d or 4d array
    :param data2: 3d or 4d array
    :return data_concat: concate(data1, data2)
    """
    if len(np.shape(data1)) == 3:
        data1 = data1[..., np.newaxis]
    if len(np.shape(data2)) == 3:
        data2 = data2[..., np.newaxis]
    return np.concatenate((data1, data2), axis=3)


def denoise_nlmeans(data_in, patch_radius=1, block_radius=5):
    """
    data_in: nd_array to denoise
    for more info about patch_radius and block radius, please refer to the dipy website: http://nipy.org/dipy/reference/dipy.denoise.html#dipy.denoise.nlmeans.nlmeans
    """
    from dipy.denoise.nlmeans import nlmeans
    from dipy.denoise.noise_estimate import estimate_sigma
    from numpy import asarray
    data_in = asarray(data_in)

    block_radius_max = min(data_in.shape) - 1
    block_radius = block_radius_max if block_radius > block_radius_max else block_radius

    sigma = estimate_sigma(data_in)
    denoised = nlmeans(data_in, sigma, patch_radius=patch_radius, block_radius=block_radius)

    return denoised


def smooth(data, sigmas):
    """
    Smooth data by convolving Gaussian kernel
    :param data: input 3D numpy array
    :param sigmas: Kernel SD in voxel
    :return:
    """
    assert len(data.shape) == len(sigmas)
    from scipy.ndimage.filters import gaussian_filter
    return gaussian_filter(data.astype(float), sigmas, order=0, truncate=4.0)


def laplacian(data, sigmas):
    """
    Apply Laplacian filter
    """
    assert len(data.shape) == len(sigmas)
    from scipy.ndimage.filters import gaussian_laplace
    return gaussian_laplace(data.astype(float), sigmas)
    # from scipy.ndimage.filters import laplace
    # return laplace(data.astype(float))


def compute_similarity(data1, data2, fname_out='', metric='', verbose=1):
    '''
    Compute a similarity metric between two images data
    :param data1: numpy.array 3D data
    :param data2: numpy.array 3D data
    :param fname_out: file name of the output file. Output file should be either a text file ('.txt') or a pickle file ('.pkl', '.pklz' or '.pickle')
    :param metric: 'mi' for mutual information or 'corr' for pearson correlation coefficient
    :return: None
    '''
    assert data1.size == data2.size, "\n\nERROR: the data don't have the same size.\nPlease use  \"sct_register_multimodal -i im1.nii.gz -d im2.nii.gz -identity 1\"  to put the input images in the same space"
    data1_1d = data1.ravel()
    data2_1d = data2.ravel()
    # get indices of non-null voxels from the intersection of both data
    data_mult = data1_1d * data2_1d
    ind_nonnull = np.where(data_mult > ALMOST_ZERO)[0]
    # set new variables with non-null voxels
    data1_1d = data1_1d[ind_nonnull]
    data2_1d = data2_1d[ind_nonnull]
    # compute similarity metric
    if metric == 'mi':
        res = mutual_information(data1_1d, data2_1d, normalized=False)
        metric_full = 'Mutual information'
    if metric == 'minorm':
        res = mutual_information(data1_1d, data2_1d, normalized=True)
        metric_full = 'Normalized Mutual information'
    if metric == 'corr':
        res = correlation(data1_1d, data2_1d)
        metric_full = 'Pearson correlation coefficient'
    # qc output
    if verbose > 1:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot(data1_1d, 'b')
        plt.plot(data2_1d, 'r')
        plt.grid
        plt.title('Similarity: ' + metric_full + ' = ' + str(res))
        plt.savefig('fig_similarity.png')

    printv('\n' + metric_full + ': ' + str(res), verbose, 'info')

    path_out, filename_out, ext_out = extract_fname(fname_out)
    if ext_out not in ['.txt', '.pkl', '.pklz', '.pickle']:
        printv('ERROR: the output file should a text file or a pickle file. Received extension: ' + ext_out, 1, 'error')

    elif ext_out == '.txt':
        file_out = open(fname_out, 'w')
        file_out.write(metric_full + ': \n' + str(res))
        file_out.close()

    else:
        import pickle, gzip
        if ext_out == '.pklz':
            pickle.dump(res, gzip.open(fname_out, 'wb'), protocol=2)
        else:
            pickle.dump(res, open(fname_out, 'w'), protocol=2)


def mutual_information(x, y, nbins=32, normalized=False):
    """
    Compute mutual information
    :param x: 1D numpy.array : flatten data from an image
    :param y: 1D numpy.array : flatten data from an image
    :param nbins: number of bins to compute the contingency matrix (only used if normalized=False)
    :return: float non negative value : mutual information
    """
    import sklearn.metrics
    if normalized:
        mi = sklearn.metrics.normalized_mutual_info_score(x, y)
    else:
        c_xy = np.histogram2d(x, y, nbins)[0]
        mi = sklearn.metrics.mutual_info_score(None, None, contingency=c_xy)
    # mi = adjusted_mutual_info_score(None, None, contingency=c_xy)
    return mi


def correlation(x, y, type='pearson'):
    """
    Compute pearson or spearman correlation coeff
    Pearson's R is parametric whereas Spearman's R is non parametric (less sensitive)
    :param x: 1D numpy.array : flatten data from an image
    :param y: 1D numpy.array : flatten data from an image
    :param type: str:  'pearson' or 'spearman': type of R correlation coeff to compute
    :return: float value : correlation coefficient (between -1 and 1)
    """
    from scipy.stats import pearsonr, spearmanr

    if type == 'pearson':
        corr = pearsonr(x, y)[0]
    if type == 'spearman':
        corr = spearmanr(x, y)[0]

    return corr


# def check_shape(data):
#     """
#     Make sure all elements of the list (given by first axis) have same shape. If data is 4d, convert to list and switch first and last axis.
#     :param data_list:
#     :return: data_list_out
#     """
#     from numpy import shape
#     # check that element of the list have same shape
#     for i in range(1, shape(data)[0]):
#         if not shape(data[0]) == shape(data[i]):
#             printv('ERROR: all input images must have same dimensions.', 1, 'error')
#     # if data are 4d (hence giving 5d list), rearrange to list of 3d data
#     if len(shape(data)) == 5:
#         from numpy import squeeze
#         from scipy import swapaxes
#         data = squeeze(swapaxes(data, 0, 4)).tolist()
#     return data

    # # random_walker
    # from skimage.segmentation import random_walker
    # import numpy as np
    # markers = np.zeros(data.shape, dtype=np.uint)
    # perc = np.percentile(data, 95)
    # markers[data < perc] = 1
    # markers[data > perc] = 2
    # mask = random_walker(data, markers, beta=10, mode='bf')

    # # spectral clustering
    # from sklearn.feature_extraction import image
    # from sklearn.cluster import spectral_clustering
    # import numpy as np
    # data2d = data[:, :, 8]
    # graph = image.img_to_graph(data2d)
    # graph.data = np.exp(-graph.data / graph.data.std())
    # mask = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
    # # label_im = -np.ones(data.shape)
    # # label_im[mask] = labels

    # Hough transform for ellipse
    # from skimage import data, color
    # from skimage.feature import canny, morphology
    # from skimage.transform import hough_ellipse
    # # detect edges
    # data2d = data3d[:, :, 8]
    # edges = canny(data2d, sigma=3.0)
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    # result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
    # result = hough_ellipse(edges, accuracy=20, min_size=5, max_size=20)
    # result.sort(order='accumulator')
    # # Estimated parameters for the ellipse
    # best = list(result[-1])
    # yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    # orientation = best[5]
    # # Draw the ellipse on the original image
    # from matplotlib.pylab import *
    # from skimage.draw import ellipse_perimeter
    # cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    # # image_rgb[cy, cx] = (0, 0, 255)
    # # Draw the edge (white) and the resulting ellipse (red)
    # # edges = color.gray2rgb(edges)
    # data2d[cy, cx] = 1000

    # # detect edges
    # from skimage.feature import canny
    # from skimage import morphology, measure
    # data2d = data3d[:, :, 8]
    # edges = canny(data2d, sigma=3.0)
    # contours = measure.find_contours(edges, 1, fully_connected='low')

    # mask = morphology.closing(edges, morphology.square(3), out=None)

    # k-means clustering
    # from sklearn.cluster import KMeans


if __name__ == "__main__":
    init_sct()
    main()
