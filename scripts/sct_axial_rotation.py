#!/usr/bin/env python
#########################################################################################
#  This code detects the axial rotation (xy plane) of the spinal cord
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2018 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Nicolas Pinon
#
# License: see the LICENSE.TXT
#########################################################################################


from __future__ import division, absolute_import

import sys, os, shutil
from math import asin, cos, sin, acos, pi, atan2, atan, floor
import numpy as np

from scipy.ndimage import convolve
from scipy.io import loadmat
from scipy.signal import medfilt, find_peaks_cwt, argrelextrema
from skimage.filters import sobel_h, sobel_v
from skimage.draw import line
from nibabel import load, Nifti1Image, save
from skimage import feature as ft
from scipy.ndimage.filters import gaussian_filter

from spinalcordtoolbox.image import Image
import sct_utils as sct
from msct_parser import Parser

import matplotlib.pyplot as plt




def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Blablablabla')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image source.",
                      mandatory=True,
                      example="src.nii.gz")
    # parser.add_option(name="-d",
    #                   type_value="file",
    #                   description="Image destination.",
    #                   mandatory=True,
    #                   example="dest.nii.gz")
    parser.add_option(name="-iseg",
                      type_value="file",
                      description="Segmentation source.",
                      mandatory=True,
                      example="src_seg.nii.gz")
    # parser.add_option(name="-dseg",
    #                   type_value="file",
    #                   description="Segmentation destination.",
    #                   mandatory=True,
    #                   example="dest_seg.nii.gz")
    parser.add_option(name="-onumber",
                      type_value="str",
                      description="outputnumber",
                      mandatory=False,
                      example="56")
    parser.add_option(name="-ofolder",
                      type_value="str",
                      description="output folder",
                      mandatory=False,
                      example="path/to/output/folder")
    # parser.add_option(name="-r",
    #                   type_value="multiple_choice",
    #                   description="""Remove temporary files.""",
    #                   mandatory=False,
    #                   default_value='1',
    #                   example=['0', '1'])
    # parser.add_option(name="-v",
    #                   type_value="multiple_choice",
    #                   description="""Verbose.""",
    #                   mandatory=False,
    #                   default_value='1',
    #                   example=['0', '1', '2'])

    return parser


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    arguments = parser.parse(args)
    # get arguments
    fname_src = arguments['-i']
    # fname_dest = arguments['-d']
    fname_src_seg = arguments['-iseg']
    # fname_dest_seg = arguments['-dseg']
    if '-onumber' in arguments:
        output_number = arguments['-onumber']
    else:
        name_output = "output.nii.gz"  # TODO: arrange this
    if '-ofolder' in arguments:
        path_output = arguments['-ofolder']
    else:
        path_output = os.getcwd()
    # remove_temp_files = int(arguments['-r'])
    # verbose = int(arguments['-v'])

    # SCT Image object
    im_src = Image(fname_src).change_orientation("RPI")
    # im_dest = Image(fname_dest)
    im_src_seg = Image(fname_src_seg).change_orientation("RPI")
    # im_dest_seg = Image(fname_dest_seg)

    # extracting data
    data_src = im_src.data
    # data_dest = im_dest.data
    data_src_seg = im_src_seg.data
    # data_dest_seg = im_dest_seg.data

    # Get image dimensions
    sct.printv('\nGet image dimensions of tutut image...', verbose=1)
    if im_src.dim != im_src_seg.dim:
        sct.printv("Dimensions of seg and image are not the same, don't know how to deal with this problem")
        # TODO: deal with this
        return

    nx, ny, nz, nt, px, py, pz, pt = im_src.dim
    sct.printv('  matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose=1)
    sct.printv('  voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', verbose=1)

    # Initialisation
    centermass_dest = np.zeros([nx, ny, nz])
    data_dest_wline = np.copy(data_src)
    angle_src = np.full(nz, -1, dtype=float)

    # Number of bins for orientation histogram
    nb_bin = 360  # TODO : relate to the resolution

    kfilter_param = 1

    nb_max = 5  #search for 5 maximums

    angle_max = 15

    filter = 'median'

    conv_plot = np.zeros((nz, nb_bin))

    # portion = 5
    # portion2 = 2
    # tut = int(floor(ny / portion))
    #
    # data_crop = data_src[:, tut:portion2 * tut, :]
    # save_nifti_like(data=data_crop, fname="t2_crop.nii.gz", fname_like=fname_src,
    #                 ofolder=path_output)

    for iz in range(0, nz):


        # slice_data = data_src[:, tut:portion2*tut, iz]
        slice_data = data_src[:, :, iz]

        coord_src_seg = np.array(data_src_seg[:, :, iz].nonzero())  # non zero coordinate of the slice

        if coord_src_seg.size != 0:  # If array is not empty
            centermass = coord_src_seg.mean(1).round().astype(int)  # Average of the coordinate (center of mass) than
            # round and convert to int
            # just to see the output of centerofmass :
            centermass_dest[centermass[0], centermass[1], iz] = 1  # put the on center of mass voxel to one

            # acquire histogram of gradient orientation
            hog_ancest = hog_ancestor(slice_data, nb_bin=nb_bin)
            # smooth it with median filter
            hog_ancest_smooth = circular_filter_1d(hog_ancest, kfilter_param, filter=filter)
            # fft than square than ifft to calculate convolution
            hog_fft2 = np.fft.fft(hog_ancest_smooth) ** 2
            hog_conv = np.real(np.fft.ifft(hog_fft2))
            conv_plot[iz, :] = hog_conv
            # search for maximum to find angle of rotation
            argmaxs = argrelextrema(hog_conv, np.greater, mode='wrap', order=kfilter_param)[0]  # get local maxima
            argmaxs_sorted = [tutut for _, tutut in sorted(zip(hog_conv[argmaxs], argmaxs))]  # sort maxima based on value

            found = False
            for angle in range(0, nb_max):
                if len(argmaxs_sorted)>=angle+1:
                    if (-angle_max < argmaxs_sorted[angle]-135 < angle_max):
                        angle_src[iz] = argmaxs_sorted[angle] -135
                        found = True
            # TODO URGENT MODIFIER CES HISTOIRES DE 135 DEGRES


            if found == False:
                angle_src[iz] = None
                data_dest_wline[:, :, iz] = data_src[:, :, iz]
            else:
                data_dest_wline[:, :, iz] = generate_2Dimage_line(data_src[:, :, iz], centermass[0], centermass[1],
                                                                  angle_src[iz])
            # generate image to visualise angle of orientation
            # data_dest_wline[:, :, iz] = generate_2Dimage_line(data_src[:, :, iz], centermass[0], centermass[1], angle_src[iz])
            # data_dest_wline[:, :, iz] = generate_2Dimage_line(generate_2Dimage_line(data_src[:, :, iz], centermass[0], centermass[1], argmaxs_sorted[0]), centermass[0], centermass[1],
            #                                                   argmaxs_sorted[1])



    save_nifti_like(data=centermass_dest, fname="centermass.nii", fname_like=fname_src, ofolder=path_output)
    save_nifti_like(data=data_dest_wline, fname="t2_wline_" + output_number + ".nii.gz", fname_like=fname_src, ofolder=path_output)

    plt.ioff()

    plt.figure()
    plt.plot(np.arange(0, nz), angle_src)
    plt.title("angle at a given slice")
    plt.xlabel("slice number")
    plt.ylabel("angle (degrees)")

    if '-f' in arguments:
        os.chdir(path_output)

    plt.savefig("angle_z_" + output_number + ".png")
    plt.close()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for z in range(0, nz):
        if np.amax(conv_plot[z, :]) > 0 :
            ax.plot(np.arange(0, 360, 360.0 / nb_bin), conv_plot[z, :], zs=z)
    plt.show()

    sct.printv("done !")



def save_nifti_like(data, fname, fname_like, ofolder=None):
    """ This functions creates a nifti image with data provided and the same header as the file provided
    inputs :
        - data : ND numpy array of data that we want to save as nifti
        - fname : name wanted for the data, an str
        - fname_like : name of the file that the header will be copied from to form the image, an str
    outputs :
        - the output image is saved under the name fname, contains the data data and the header of the fname_like file
        """

    img_like = load(fname_like)
    header = img_like.header.copy()
    img = Nifti1Image(data, None, header=header)
    if ofolder is not None:
        cwd = os.getcwd()
        os.chdir(ofolder)
    save(img, fname)
    if ofolder is not None:
        os.chdir(cwd)

def circular_filter_1d(signal, param_filt, filter='gaussian'):

    """ This function filters circularly the signal inputted with a median filter of inputted size, in this context
    circularly means that the signal is wrapped around and then filtered
    inputs :
        - signal : 1D numpy array
        - window_size : size of the median filter, an int
    outputs :
        - signal_smoothed : 1D numpy array"""

    signal_extended = np.concatenate((signal, signal, signal))  # replicate signal at both ends
    if filter == 'gaussian':
        signal_extended_smooth = gaussian_filter(signal_extended, param_filt)  # gaussian
    elif filter == 'median':
        signal_extended_smooth = medfilt(signal_extended, param_filt)  # median filtering
    else:
        sct.printv("unknow type of filter")
        raise

    length = len(signal)
    signal_smoothed = signal_extended_smooth[length:2*length]  # truncate back the signal

    return signal_smoothed

def hog_ancestor(image, nb_bin, grad_ksize=123456789): # TODO implement selection of gradient's kernel size

    """ This function takes an image as an input and return its orientation histogram
    inputs :
        - image : the image to compute the orientation histogram from, a 2D numpy array
        - nb_bin : the number of bins of the histogram, an int
        - grad_ksize : kernel size of gradient (work in progress)
    outputs :
        - hog_ancest : the histogram of the orientations of the image, a 1D numpy array of length nb_bin"""

    h_kernel = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]]) / 4.0
    v_kernel = h_kernel.T

    # x and y gradients
    gradx = convolve(image, v_kernel)
    grady = convolve(image, h_kernel)
    # orientation gradient
    orient = np.arctan2(grady, gradx)*180/pi
    # changing results from [-180,180] to [0,360] (more convenient to visualise) :
    # negatives = orient < 0
    # orient[negatives] = orient[negatives] + 360  #TODO !!!

    # weight by gradient magnitude : TODO : this step seems dumb, it alters the angles
    # actually it can be smart but by doing a weighted histogram, not weight the image

    grad_mag = (np.abs(gradx.astype(object))**2+np.abs(grady.astype(object))**2)**0.5
    grad_weight = (grad_mag > 0).astype(int)

    # compute histogram :
    hog_ancest = np.histogram(np.concatenate(orient), bins=nb_bin, range=(-nb_bin/2, nb_bin/2), weights=np.concatenate(grad_weight))
    # hog_ancest = np.histogram(np.concatenate(orient), bins=nb_bin)

    return hog_ancest[0].astype(float)  # return only the values of the bins, not the bins (we know them)

def generate_2Dimage_line(image, x0, y0, angle):

    """ This function takes an image and a line (defined by a point and an angle) as inputs and outputs the same image
    but with the line drawn on it
    inputs :
        - image : image to draw the line on, 2D numpy array
        - x0 and y0 : coordinates of one point the line passes through, two ints
        - angle : angle the lines makes with  x axis
    outputs :
        - image_wline : base image with the line drawn on it, 2D numpy array"""

    angle = (angle - 90)*pi/180  # TODO : justify this

    # coordinates of image's borders :
    x_max, y_max = image.shape
    x_max, y_max = x_max - 1, y_max - 1  # because indexing starts at 0
    x_min, y_min = 0, 0

    # we want to generate the line across the image, to do so we must provide two points that are on the edge of the
    # image (to draw a full, beautiful line) so we search for the two points (x1,y1) and (x2,y2) that are
    # on the line that passes through the point (x0,y0) with angle = angle and that are on the edges of the image
    # we will first find (x1,y1) and then (x2,y2)

    first_point_found = False  # (x1,y1) not found yet

    # Justification of the later : basic geometry

    x = round( (y_min - y0)/(np.tan(angle) + 0.00001) + x0 ) # not elegant at all, must change TODO : change this
    if  x >= 0 and x<= x_max:
        x1 = x
        y1 = y_min
        first_point_found = True
    x = round( (y_max - y0)/(np.tan(angle) + 0.00001) + x0 )
    if x >= 0 and x <= x_max:
        if first_point_found is False:  # this condition means the first point has not been found yet
            x1 = x
            y1 = y_max
            first_point_found = True
        else:
            x2 = x
            y2 = y_max
    y = round( (x_min - x0)*np.tan(angle) + y0 )
    if y >= 0 and y <= y_max:
        if first_point_found is False:
            x1 = x_min
            y1 = y
            first_point_found = True
        else:
            x2 = x_min
            y2 = y
    y = round( (x_max - x0) * np.tan(angle) + y0 )
    if y >= 0 and y <= y_max:
        if first_point_found is False:
            sct.printv("Error, this is not supposed to happen")  # impossible not to have found the first point at
            # the latest step because we must find 2 points
        else:
            x2 = x_max
            y2 = y

    coord_linex, coord_liney = line(int(floor(x1)), int(floor(y1)), int(floor(x2)), int(floor(y2)))
    # use the line function from scikit image to acquire pixel coordinates of the line

    image_wline = image
    image_wline[coord_linex, coord_liney] = np.amax(image)  # put the line at full intensity (not really elegant)
    # actually the "copy" is not useful, just used to clarify, because python does not make an actual copy when you do
    # this

    return image_wline

def symmetry_angle(image_data, nb_bin=360, kmedian_size=5, nb_axes=1):

    "This function outputs the symetry angle, put -1 in nb_axes to get all the axes found" #  TODO: detail this

    # acquire histogram of gradient orientation
    hog_ancest = hog_ancestor(image_data, nb_bin=nb_bin)
    # smooth it with median filter
    hog_ancest_smooth = circular_filter_1d(hog_ancest, kmedian_size, filter='median')
    # fft than square than ifft to calculate convolution
    hog_fft2 = np.fft.fft(hog_ancest_smooth) ** 2
    hog_conv = np.real(np.fft.ifft(hog_fft2))

    # TODO FFT CHECK SAMPLING
    # hog_conv = np.convolve(hog_ancest_smooth, hog_ancest_smooth, mode='same')

    # search for maximum to find angle of rotation
    # TODO : works only if nb_bin = 360
    argmaxs = argrelextrema(hog_conv, np.greater, mode='wrap', order=kmedian_size)[0]  # get local maxima
    argmaxs_sorted = [tutut for _, tutut in sorted(zip(hog_conv[argmaxs], argmaxs), reverse=True)]  # sort maxima based on
    if nb_axes == -1:
        angles = argmaxs_sorted
    elif nb_axes > len(argmaxs_sorted):
        sct.printv(str(nb_axes) + " were asked for, only found " + str(len(argmaxs_sorted)))
        angles = argmaxs_sorted
    else:
        angles = argmaxs_sorted[0:nb_axes]

    return angles






if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
