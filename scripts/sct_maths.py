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

from msct_parser import Parser
from msct_image import Image
from sct_utils import extract_fname, printv


class Param:
    def __init__(self):
        self.verbose = '1'

# PARSER
# ==========================================================================================
def get_parser():
    # parser initialisation
    parser = Parser(__file__)

    # # initialize parameters
    # param = Param()
    # param_default = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Perform mathematical operations on images. Only one operation at a time can be done (except for intensity scaling: flag -scale).')
    parser.add_option(name="-i",
                      type_value=[[','], 'file'],
                      description="Input file(s). If several inputs: separate them by a coma without white space.\n"
                                  "If several inputs: the sme operation is applied to all the inputs (except for the operations -add, -sub and -scale)",
                      mandatory=True,
                      example="data.nii.gz")
    parser.add_option(name="-o",
                      type_value=[[','], 'file_output'],
                      description='Output file(s). If several outputs: separate them by a coma without white space.',
                      mandatory=True,
                      example=['data_mean.nii.gz'])
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
    parser.usage.addSection("\nThresholding methods")
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
    parser.usage.addSection("\nBasic operations")
    parser.add_option(name="-add",
                      description='Add input images (need more than one input).',
                      mandatory=False)
    parser.add_option(name="-sub",
                      description='Substract two input images: output = input1 - input2',
                      mandatory=False)
    parser.add_option(name="-mul",
                      description='Multiply input images (need more than one input).',
                      mandatory=False)
    parser.add_option(name="-div",
                      description='Divide two input images: output = input1 / input2.',
                      mandatory=False)
    parser.add_option(name="-scale",
                      type_value=[[','], 'float'],
                      description='Scaling factors applied to all the inputs intensity.\n'
                                  'The intensity scaling can be used in combination to another operation and will be applied first.\n'
                                  'The number of scaling factors must be the same as the number of inputs.',
                      mandatory=False,
                      example='0.5')
    parser.add_option(name="-smooth",
                      type_value=[[','], 'float'],
                      description='Gaussian smoothing filter with specified standard deviations in mm for each axis (e.g.: 2,2,1) or single value for all axis (e.g.: 2).',
                      mandatory=False,
                      example='0.5')
    parser.add_option(name="-bin",
                      description='Use (input image>0) to binarise.',
                      mandatory=False)
    parser.add_option(name="-pad",
                      type_value="str",
                      description="Padding dimensions in voxels for the x, y, and z dimensions, separated with \"x\".",
                      mandatory=False,
                      example='0x0x1')
    parser.usage.addSection("\nDimensionality reduction operations")
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
    parser.usage.addSection("\nMulti-component operations")
    parser.add_option(name='-mcs',
                      description='Multi-component split. Outputs the components separately.\n'
                                  '(If less outputs names than components in the image, outputs as many components as the number of outputs name specifed.)\n'
                                  'Only one input',
                      mandatory=False)
    parser.add_option(name='-omc',
                      description='Multi-component output. Merge inputted images into one multi-component image.\n'
                                  'Only one output',
                      mandatory=False)
    parser.usage.addSection("\nMisc")
    parser.add_option(name="-w",
                      description="Output is a warping field",
                      mandatory=False)
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value=param.verbose,
                      example=['0', '1', '2'])
    return parser


# MAIN
# ==========================================================================================
def main(args = None):

    dim_list = ['x', 'y', 'z', 't']

    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_in = arguments["-i"]
    n_in = len(fname_in)
    fname_out = arguments["-o"]
    n_out = len(fname_out)

    verbose = int(arguments['-v'])

    # Open file(s)
    nii = [Image(f_in) for f_in in fname_in]
    data = [im.data for im in nii]

    # run command
    if "-scale" in arguments:
        factors = arguments["-scale"]
        if len(factors) != n_in:
            printv(parser.usage.generate(error='ERROR: -scale must have the same number of factors as the number of inputs'))
        data_out = scale_intensity(data, factors)
        for im_in, d_out in zip(nii, data_out):
            im_in.data = d_out
    if '-otsu' in arguments:
        param = arguments['-otsu']
        data_out = [otsu(d, param) for d in data]
    elif '-otsu_adap' in arguments:
        param = arguments['-otsu_adap']
        data_out = [otsu_adap(d, param[0], param[1]) for d in data]
    elif '-otsu_median' in arguments:
        param = arguments['-otsu_median']
        data_out = [otsu_median(d, param[0], param[1]) for d in data]
    elif '-thr' in arguments:
        param = arguments['-thr']
        data_out = [threshold(d, param) for d in data]
    elif '-percent' in arguments:
        param = arguments['-percent']
        data_out = [perc(d, param) for d in data]
    elif '-bin' in arguments:
        data_out = [binarise(d) for d in data]
    elif '-add' in arguments:
        if n_in == 1:
            printv(parser.usage.generate(error='ERROR: -add needs more than one input'))
        check_shape(data)
        data_out = add(data)
        data_out = [data_out]
    elif '-sub' in arguments:
        if n_in != 2:
            printv(parser.usage.generate(error='ERROR: -sub needs only 2 inputs'))
        check_shape(data)
        data_out = substract(data)
        data_out = [data_out]
    elif '-mul' in arguments:
        if n_in == 1:
            printv(parser.usage.generate(error='ERROR: -mul needs more than one input'))
        check_shape(data)
        data_out = mul(data)
        data_out = [data_out]
    elif '-div' in arguments:
        if n_in == 1:
            printv(parser.usage.generate(error='ERROR: -mul needs more than one input'))
        check_shape(data)
        data_out = mul(data)
        data_out = [data_out]
    elif '-mean' in arguments:
        dim = dim_list.index(arguments['-mean'])
        data_out = [compute_mean(d, dim) for d in data]
    elif '-std' in arguments:
        dim = dim_list.index(arguments['-std'])
        data_out = [compute_std(d, dim) for d in data]
    elif "-pad" in arguments:
        padx, pady, padz = arguments["-pad"].split('x')
        padx, pady, padz = int(padx), int(pady), int(padz)
        data_out = [pad_image(im, padding_x=padx, padding_y=pady, padding_z=padz) for im in nii]
    elif "-smooth" in arguments:
        sigmas = arguments["-smooth"]
        data_out = []
        for d in data:
            if len(sigmas) == 1:
                sigmas = [sigmas[0] for i in range(len(d.shape))]
            elif len(sigmas) != len(d.shape):
                printv(parser.usage.generate(error='ERROR: -smooth need the same number of inputs as the number of image dimension OR only one input'))
            data_out.append(smooth(d, sigmas))
    elif '-dilate' in arguments:
        data_out = [dilate(d, arguments['-dilate']) for d in data]
    elif '-erode' in arguments:
        data_out = [erode(d, arguments['-erode']) for d in data]
    elif '-mcs' in arguments:
        if n_in != 1:
            printv(parser.usage.generate(error='ERROR: -mcs need only one input'))
        if len(data[0].shape) != 5:
            printv(parser.usage.generate(error='ERROR: -mcs input need to be a multi-component image'))
        data_out = multicomponent_split(data[0])
        if len(data_out) > n_out:
            data_out = data_out[:n_out]
    elif '-omc' in arguments:
        if n_out != 1:
            printv(parser.usage.generate(error='ERROR: -omc need only one output'))
        for dat in data:
            if dat.shape != data[0].shape:
                printv(parser.usage.generate(error='ERROR: -omc inputs need to have all the same shapes'))
        data_out = multicomponent_merge(data)
    elif "-scale" not in arguments:
        printv('No process applied.', 1, 'warning')
        return

    # Write output
    assert len(data_out) == n_out
    if n_in == n_out:
        for im_in, d_out, fn_out in zip(nii, data_out, fname_out):
            im_in.data = d_out
            im_in.setFileName(fn_out)
            if "-w" in arguments:
                im_in.hdr.set_intent('vector', (), '')
            im_in.save()
    elif n_out == 1:
        nii[0].data = data_out[0]
        nii[0].setFileName(fname_out[0])
        if "-w" in arguments:
                nii[0].hdr.set_intent('vector', (), '')
        nii[0].save()
    elif n_out > n_in:
        for dat_out, name_out in zip(data_out, fname_out):
            im_out = nii[0].copy()
            im_out.data = dat_out
            im_out.setFileName(name_out)
            if "-w" in arguments:
                im_out.hdr.set_intent('vector', (), '')
            im_out.save()
    else:
        printv(parser.usage.generate(error='ERROR: not the correct numbers of inputs and outputs'))

    # display message
    printv('Created file(s):\n--> '+str(fname_out)+'\n', verbose, 'info')


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


def compute_mean(data, dim):
    from numpy import mean
    return mean(data, dim)


def compute_std(data, dim):
    from numpy import std
    return std(data, dim)


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


def pad_image(im, padding_x=0, padding_y=0, padding_z=0):
    from numpy import zeros
    nx, ny, nz, nt, px, py, pz, pt = im.dim
    padding_x, padding_y, padding_z = int(padding_x), int(padding_y), int(padding_z)
    padded_data = zeros((nx+2*padding_x, ny+2*padding_y, nz+2*padding_z))

    if padding_x == 0:
        padxi = None
        padxf = None
    else:
        padxi=padding_x
        padxf=-padding_x

    if padding_y == 0:
        padyi = None
        padyf = None
    else:
        padyi = padding_y
        padyf = -padding_y

    if padding_z == 0:
        padzi = None
        padzf = None
    else:
        padzi = padding_z
        padzf = -padding_z

    padded_data[padxi:padxf, padyi:padyf, padzi:padzf] = im.data
    im.data = padded_data  # done after the call of the function

    # adapt the origin in the sform and qform matrix
    def get_sign_offsets(orientation):
        default = 'LPI'
        offset_sign = [0, 0, 0]

        for i in range(3):
            if default[i] in orientation:
                offset_sign[i] = -1
            else:
                offset_sign[i] = 1
        return offset_sign

    offset_signs = get_sign_offsets(im.orientation)
    im.hdr.structarr['qoffset_x'] += offset_signs[0]*padding_x*px
    im.hdr.structarr['qoffset_y'] += offset_signs[1]*padding_y*py
    im.hdr.structarr['qoffset_z'] += offset_signs[2]*padding_z*pz
    im.hdr.structarr['srow_x'][-1] += offset_signs[0]*padding_x*px
    im.hdr.structarr['srow_y'][-1] += offset_signs[1]*padding_y*py
    im.hdr.structarr['srow_z'][-1] += offset_signs[2]*padding_z*pz

    return padded_data


def add(data_list):
    """
    Sum a bunch of numpy arrays
    """
    from numpy import sum
    return sum(data_list, axis=0)


def substract(data_list):
    """
    Substract two numpy arrays
    """
    from numpy import reshape
    data0, data1 = data_list
    return data0 - data1


def mul(data_list):
    """
    Multiply a bunch of numpy arrays
    """
    from numpy import prod
    return prod(data_list, axis=0)


def div(data_list):
    """
    Divide two numpy arrays
    """
    from numpy import divide
    data0, data1 = data_list
    return divide(data0, data1)


def scale_intensity(data_list, factors):
    data_out = []
    assert len(data_list) == len(factors)
    for dat, f in zip(data_list, factors):
        data_out.append(dat*f)
    return data_out


def smooth(data, sigmas):
    assert len(data.shape) == len(sigmas)
    from scipy.ndimage.filters import gaussian_filter
    return gaussian_filter(data, sigmas)


def multicomponent_split(data):
    from numpy import reshape
    assert len(data.shape) == 5
    data_out = []
    for i in range(data.shape[-1]):
        dat_out = data[:, :, :, :, i]
        if dat_out.shape[-1] == 1:
            dat_out = reshape(dat_out, dat_out.shape[:-1])
            if dat_out.shape[-1] == 1:
                dat_out = reshape(dat_out, dat_out.shape[:-1])
        data_out.append(dat_out.astype('float32'))

    return data_out


def multicomponent_merge(data_list):
    from numpy import zeros, reshape
    # WARNING: output multicomponent is not optimal yet, some issues may be related to the use of this function
    new_shape = list(data_list[0].shape)
    if len(new_shape) == 3:
        new_shape.append(1)
    new_shape.append(len(data_list))
    new_shape = tuple(new_shape)

    data_out = zeros(new_shape)
    for i, dat in enumerate(data_list):
        if len(dat.shape) < 4:
            new_shape = list(dat.shape)
            while len(new_shape) < 4:
                new_shape.append(1)
            dat = reshape(dat, tuple(new_shape))
        data_out[:, :, :, :, i] = dat.astype('float32')
    return [data_out.astype('float32')]

def check_shape(data_list):
    """
    Make sure all elements of the list (given by first axis) have same shape
    """
    from numpy import shape, array_equal
    for i in range(1, shape(data_list)[0]):
        if not array_equal(data_list[0], data_list[i]):
            printv('ERROR: all input images must have same dimensions.', 1, 'error')
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
    # # initialize parameters
    param = Param()
    # call main function
    main()
