#!/usr/bin/env python
#########################################################################################
#
# Perform mathematical operations on images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
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
    parser.usage.set_description('Perform mathematical operations on images. Only one operation at a time can be done.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Input file.",
                      mandatory=True,
                      example="data.nii.gz")
    parser.add_option(name="-o",
                      type_value="file_output",
                      description='Output file.',
                      mandatory=True,
                      example=['data_mean.nii.gz'])
    parser.add_option(name='-otsu',
                      type_value='int',
                      description='Threshold image using Otsu algorithm.\nnbins: number of bins. Example: 256',
                      mandatory=False,
                      example="")
    parser.add_option(name="-otsu_adap",
                      type_value=[[','], 'int'],
                      description="Threshold image using Adaptive Otsu algorithm.\nblock_size:\noffset:\n",
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
                      type_value='int',
                      description='Use following number to threshold image (zero below number).',
                      mandatory=False,
                      example="")
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
    fname_out = arguments["-o"]
    verbose = int(arguments['-v'])

    # Build fname_out
    if fname_out == '':
        path_in, file_in, ext_in = extract_fname(fname_in)
        fname_out = path_in+file_in+'_mean'+ext_in

    # Open file.
    nii = Image(fname_in)
    data = nii.data

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
    elif '-mean' in arguments:
        dim = dim_list.index(arguments['-mean'])
        data_out = compute_mean(data, dim)
    elif '-std' in arguments:
        dim = dim_list.index(arguments['-std'])
        data_out = compute_std(data, dim)
    else:
        printv('No process applied.', 1, 'warning')
        return

    # Write output
    nii.data = data_out
    nii.setFileName(fname_out)
    nii.save()

    # display message
    printv('Created file:\n--> '+fname_out+'\n', verbose, 'info')


def otsu(data, nbins):
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(data, nbins)
    return data > thresh


def otsu_adap(data, block_size, offset):
    from skimage.filters import threshold_adaptive, threshold_otsu
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
    return data > thr_value


def perc(data, perc_value):
    from numpy import percentile
    perc = percentile(data, perc_value)
    return data > perc


def compute_mean(data, dim):
    from numpy import mean
    return mean(data, dim)


def compute_std(data, dim):
    from numpy import std
    return std(data, dim)


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
