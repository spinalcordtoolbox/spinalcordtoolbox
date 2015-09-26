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
from sct_utils import extract_fname, printv


class Param:
    def __init__(self):
        self.verbose = '1'

# PARSER
# ==========================================================================================
def get_parser():

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Perform mathematical operations on images. Some inputs can be either a number or a 4d image or several 3d images separated with ","')
    parser.add_option(name="-i",
                      type_value=[[','], 'file'],
                      description="Input file(s). If several inputs: separate them by a coma without white space.\n"
                                  "If several inputs: the sme operation is applied to all the inputs (except for the operations -add, -sub and -scale)",
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
    parser.add_option(name="-pad",
                      type_value="str",
                      description='Pad 3d image. Specify padding as: "x,y,z" (in voxel)',
                      mandatory=False,
                      example='0,0,1')

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

    parser.usage.addSection("\nMulti-component operations:")
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
    verbose = int(arguments['-v'])

    # Open file(s)
    data = get_data(fname_in)  # 3d or 4d numpy array

    # run command
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
        from numpy import sum
        data2 = get_data_or_scalar(arguments["-add"], data)
        data_concat = concatenate_along_4th_dimension(data, data2)
        data_out = sum(data_concat, axis=3)

    elif '-sub' in arguments:
        data2 = get_data_or_scalar(arguments["-sub"], data)
        data_out = data - data2

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

    elif "-pad" in arguments:
        # TODO: check input is 3d
        padx, pady, padz = arguments["-pad"].split(',')
        padx, pady, padz = int(padx), int(pady), int(padz)
        nii = Image(fname_in[0])
        nii_out = pad_image(nii, padding_x=padx, padding_y=pady, padding_z=padz)
        # data_out = pad_image(nii, padding_x=padx, padding_y=pady, padding_z=padz)

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

    # Write output
    if not "-pad" in arguments:
        nii_out = Image(fname_in[0])  # use header of first file (if multiple input files)
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

    return im


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
    if argument.isdigit():
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
    # # initialize parameters
    param = Param()
    # call main function
    main()
