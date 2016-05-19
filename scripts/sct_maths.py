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

import sys

from numpy import concatenate, shape, newaxis
from msct_parser import Parser
from msct_image import Image
from sct_utils import printv


class Param:
    def __init__(self):
        self.verbose = '1'

# PARSER
# ==========================================================================================
def get_parser():
    param = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Perform mathematical operations on images. Some inputs can be either a number or a 4d image or several 3d images separated with ","')
    parser.add_option(name="-i",
                      type_value='file',
                      description="Input file. ",
                      mandatory=True,
                      example="data.nii.gz")
    parser.add_option(name="-o",
                      type_value='file_output',
                      description='Output file.',
                      mandatory=True,
                      example=['data_mean.nii.gz'])

    parser.usage.addSection('\nBasic operations:')
    parser.add_option(name="-add",
                      type_value='str',
                      description='Add following input (can be number or image(s))',
                      mandatory=False)
    parser.add_option(name="-sub",
                      type_value='str',
                      description='Substract following input (can be number or image)',
                      mandatory=False)
    parser.add_option(name="-mul",
                      type_value='str',
                      description='Multiply following input (can be number or image(s))',
                      mandatory=False)
    parser.add_option(name="-div",
                      type_value='str',
                      description='Divide following input (can be number or image)',
                      mandatory=False)
    parser.add_option(name='-mean',
                      type_value='multiple_choice',
                      description='Average data across dimension.',
                      mandatory=False,
                      example=['x', 'y', 'z', 't'])
    parser.add_option(name='-std',
                      type_value='multiple_choice',
                      description='Compute STD across dimension.',
                      mandatory=False,
                      example=['x', 'y', 'z', 't'])
    parser.add_option(name="-bin",
                      description='Use (input image>0) to binarise.',
                      mandatory=False)

    parser.usage.addSection("\nThresholding methods:")
    parser.add_option(name='-otsu',
                      type_value='int',
                      description='Threshold image using Otsu algorithm.\nnbins: number of bins. Example: 256',
                      mandatory=False,
                      example="")
    parser.add_option(name="-otsu_adap",
                      type_value=[[','], 'int'],
                      description="Threshold image using Adaptive Otsu algorithm.\nblock_size:\noffset:",
                      mandatory=False,
                      example="")
    parser.add_option(name="-otsu_median",
                      type_value=[[','], 'int'],
                      description='Threshold image using Median Otsu algorithm. Separate with ","\n- Size of the median filter. Example: 2\n- Number of iterations. Example: 3\n',
                      mandatory=False,
                      example='2,3')
    parser.add_option(name='-percent',
                      type_value='int',
                      description="Threshold image using percentile of its histogram.",
                      mandatory=False)
    parser.add_option(name="-thr",
                      type_value='float',
                      description='Use following number to threshold image (zero below number).',
                      mandatory=False,
                      example="")

    parser.usage.addSection("\nMathematical morphology")
    parser.add_option(name='-dilate',
                      type_value='int',
                      description='Dilate binary image using specified ball radius.',
                      mandatory=False,
                      example="")
    parser.add_option(name='-erode',
                      type_value='int',
                      description='Erode binary image using specified ball radius.',
                      mandatory=False,
                      example="")

    parser.usage.addSection("\nFiltering methods:")
    parser.add_option(name="-smooth",
                      type_value=[[','], 'float'],
                      description='Gaussian smoothing filter with specified standard deviations in mm for each axis (e.g.: 2,2,1) or single value for all axis (e.g.: 2).',
                      mandatory=False,
                      example='0.5')
    parser.add_option(name='-laplacian',
                      type_value='float',
                      description='Laplacian filtering with specified standard deviations in mm for all axes (e.g.: 2).',
                      mandatory=False,
                      example='1')
    parser.add_option(name='-denoise',
                      type_value=[[','], 'str'],
                      description='Non-local means adaptative denoising from P. Coupe et al. Separate with ",". Example: v=3,f=1,h=0.05.\n'
                        'v:  similar patches in the non-local means are searched for locally, inside a cube of side 2*v+1 centered at each voxel of interest. Default: v=3\n'
                        'f:  the size of the block to be used (2*f+1)x(2*f+1)x(2*f+1) in the blockwise non-local means implementation. Default: f=1\n'
                        'h:  the standard deviation of rician noise in the input image, expressed as a ratio of the maximum intensity in the image. The higher, the more aggressive the denoising. Default: h=0.01',
                      mandatory=False,
                      example="")
    parser.usage.addSection("\nMisc")
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value=param.verbose,
                      example=['0', '1', '2'])
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    dim_list = ['x', 'y', 'z', 't']

    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    fname_in = arguments["-i"]
    fname_out = arguments["-o"]
    verbose = int(arguments['-v'])

    # Open file(s)
    im = Image(fname_in)
    data = im.data  # 3d or 4d numpy array
    dim = im.dim

    # run command
    if '-otsu' in arguments:
        param = arguments['-otsu']
        data_out = otsu(data, param)

    elif '-otsu_adap' in arguments:
        param = arguments['-otsu_adap']
        data_out = otsu_adap(data, param[0], param[1])

    elif '-otsu_median' in arguments:
        param = arguments['-otsu_median']
        data_out = otsu_median(data, param[0], param[1])

    elif '-thr' in arguments:
        param = arguments['-thr']
        data_out = threshold(data, param)

    elif '-percent' in arguments:
        param = arguments['-percent']
        data_out = perc(data, param)

    elif '-bin' in arguments:
        data_out = binarise(data)

    elif '-add' in arguments:
        from numpy import sum
        data2 = get_data_or_scalar(arguments["-add"], data)
        data_concat = concatenate_along_4th_dimension(data, data2)
        data_out = sum(data_concat, axis=3)

    elif '-sub' in arguments:
        data2 = get_data_or_scalar(arguments["-sub"], data)
        data_out = data - data2

    elif "-laplacian" in arguments:
        sigmas = arguments["-laplacian"]
        # if len(sigmas) == 1:
        sigmas = [sigmas for i in range(len(data.shape))]
        # elif len(sigmas) != len(data.shape):
        #     printv(parser.usage.generate(error='ERROR: -laplacian need the same number of inputs as the number of image dimension OR only one input'))
        # adjust sigma based on voxel size
        [sigmas[i] / dim[i+4] for i in range(3)]
        # smooth data
        data_out = laplacian(data, sigmas)

    elif '-mul' in arguments:
        from numpy import prod
        data2 = get_data_or_scalar(arguments["-mul"], data)
        data_concat = concatenate_along_4th_dimension(data, data2)
        data_out = prod(data_concat, axis=3)

    elif '-div' in arguments:
        from numpy import divide
        data2 = get_data_or_scalar(arguments["-div"], data)
        data_out = divide(data, data2)

    elif '-mean' in arguments:
        from numpy import mean
        dim = dim_list.index(arguments['-mean'])
        if dim+1 > len(shape(data)):  # in case input volume is 3d and dim=t
            data = data[..., newaxis]
        data_out = mean(data, dim)

    elif '-std' in arguments:
        from numpy import std
        dim = dim_list.index(arguments['-std'])
        if dim+1 > len(shape(data)):  # in case input volume is 3d and dim=t
            data = data[..., newaxis]
        data_out = std(data, dim)

    elif "-smooth" in arguments:
        sigmas = arguments["-smooth"]
        if len(sigmas) == 1:
            sigmas = [sigmas[0] for i in range(len(data.shape))]
        elif len(sigmas) != len(data.shape):
            printv(parser.usage.generate(error='ERROR: -smooth need the same number of inputs as the number of image dimension OR only one input'))
        # adjust sigma based on voxel size
        [sigmas[i] / dim[i+4] for i in range(3)]
        # smooth data
        data_out = smooth(data, sigmas)

    elif '-dilate' in arguments:
        data_out = dilate(data, arguments['-dilate'])

    elif '-erode' in arguments:
        data_out = erode(data, arguments['-erode'])

    elif '-denoise' in arguments:
        # parse denoising arguments
        v, f, h = 3, 1, 0.01  # default arguments
        list_denoise = arguments['-denoise']
        for i in list_denoise:
            if 'v' in i:
                v = int(i.split('=')[1])
            if 'f' in i:
                f = int(i.split('=')[1])
            if 'h' in i:
                h = float(i.split('=')[1])
        data_out = denoise_ornlm(data, v, f, h)
    # if no flag is set
    else:
        data_out = None
        printv(parser.usage.generate(error='ERROR: you need to specify an operation to do on the input image'))

    # Write output
    nii_out = Image(fname_in)  # use header of input file
    nii_out.data = data_out
    nii_out.setFileName(fname_out)
    nii_out.save()
    # TODO: case of multiple outputs
    # assert len(data_out) == n_out
    # if n_in == n_out:
    #     for im_in, d_out, fn_out in zip(nii, data_out, fname_out):
    #         im_in.data = d_out
    #         im_in.setFileName(fn_out)
    #         if "-w" in arguments:
    #             im_in.hdr.set_intent('vector', (), '')
    #         im_in.save()
    # elif n_out == 1:
    #     nii[0].data = data_out[0]
    #     nii[0].setFileName(fname_out[0])
    #     if "-w" in arguments:
    #             nii[0].hdr.set_intent('vector', (), '')
    #     nii[0].save()
    # elif n_out > n_in:
    #     for dat_out, name_out in zip(data_out, fname_out):
    #         im_out = nii[0].copy()
    #         im_out.data = dat_out
    #         im_out.setFileName(name_out)
    #         if "-w" in arguments:
    #             im_out.hdr.set_intent('vector', (), '')
    #         im_out.save()
    # else:
    #     printv(parser.usage.generate(error='ERROR: not the correct numbers of inputs and outputs'))

    # display message
    printv('\nDone! To view results, type:', verbose)
    printv('fslview '+fname_out+' &\n', verbose, 'info')

def otsu(data, nbins):
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(data, nbins)
    return data > thresh


def otsu_adap(data, block_size, offset):
    from skimage.filters import threshold_adaptive

    mask = data
    for iz in range(data.shape[2]):
        mask[:, :, iz] = threshold_adaptive(data[:, :, iz], block_size, offset)
        # mask[:, :, iz] = threshold_otsu(data[:, :, iz], 5)
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


def binarise(data):
    return data > 0


def dilate(data, radius):
    """
    Dilate data using ball structuring element
    :param data: 2d or 3d array
    :param radius: radius of structuring element
    :return: data dilated
    """
    from skimage.morphology import dilation, ball
    selem = ball(radius)
    return dilation(data, selem=selem, out=None)


def erode(data, radius):
    """
    Erode data using ball structuring element
    :param data: 2d or 3d array
    :param radius: radius of structuring element
    :return: data eroded
    """
    from skimage.morphology import erosion, ball
    selem = ball(radius)
    return erosion(data, selem=selem, out=None)


def get_data(list_fname):
    """
    Get data from file names separated by ","
    :param list_fname:
    :return: 3D or 4D numpy array.
    """
    nii = [Image(f_in) for f_in in list_fname]
    data = nii[0].data
    # check that every images have same shape
    for i in range(1, len(nii)):
        if not shape(nii[i].data) == shape(data):
            printv('ERROR: all input images must have same dimensions.', 1, 'error')
        else:
            concatenate_along_4th_dimension(data, nii[i].data)
    return data

def get_data_or_scalar(argument, data_in):
    """
    Get data from list of file names (scenario 1) or scalar (scenario 2)
    :param argument: list of file names of scalar
    :param data_in: if argument is scalar, use data to get shape
    :return: 3d or 4d numpy array
    """
    if argument.replace('.', '').isdigit():  # so that it recognize float as digits too
        # build data2 with same shape as data
        data_out = data_in[:, :, :] * 0 + float(argument)
    else:
        # parse file name and check integrity
        parser2 = Parser(__file__)
        parser2.add_option(name='-i', type_value=[[','], 'file'])
        list_fname = parser2.parse(['-i', argument]).get('-i')
        data_out = get_data(list_fname)
    return data_out


def concatenate_along_4th_dimension(data1, data2):
    """
    Concatenate two data along 4th dimension.
    :param data1: 3d or 4d array
    :param data2: 3d or 4d array
    :return data_concat: concate(data1, data2)
    """
    if len(shape(data1)) == 3:
        data1 = data1[..., newaxis]
    if len(shape(data2)) == 3:
        data2 = data2[..., newaxis]
    return concatenate((data1, data2), axis=3)


def denoise_ornlm(data_in, v=3, f=1, h=0.05):
    from commands import getstatusoutput
    from sys import path
    # append python path for importing module
    # N.B. PYTHONPATH variable should take care of it, but this is only used for Travis.
    status, path_sct = getstatusoutput('echo $SCT_DIR')
    path.append(path_sct + '/external/denoise/ornlm')
    from ornlm import ornlm
    from numpy import array, max, float64
    dat = data_in.astype(float64)
    denoised = array(ornlm(dat, v, f, max(dat)*h))
    return denoised


def smooth(data, sigmas):
    """
    Smooth data by convolving Gaussian kernel
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



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()
