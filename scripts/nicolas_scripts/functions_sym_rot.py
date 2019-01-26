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
        raise Exception("unknow type of filter")

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

    grad_mag = ((np.abs(gradx.astype(object)) ** 2 + np.abs(grady.astype(object)) ** 2) ** 0.5)
    # TODO: weird data type manipulation, to explain
    # grad_weight = (grad_mag > 0).astype(int)

    # compute histogram :
    hog_ancest = np.histogram(np.concatenate(orient), bins=nb_bin, range=(-nb_bin/2, nb_bin/2), weights=np.concatenate(grad_mag))
    # hog_ancest = np.histogram(np.concatenate(orient), bins=nb_bin)
    grad_mag = (grad_mag * 255/np.max(grad_mag)).astype(float).round()  # just for debbuguing purpose

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

    angle = angle *pi/180  # converting to radians

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

    x = round((y_min - y0)/(np.tan(angle) + 0.00001) + x0)  # not elegant at all, must change TODO : change this
    if  x >= 0 and x<= x_max:
        x1 = x
        y1 = y_min
        first_point_found = True
    x = round((y_max - y0)/(np.tan(angle) + 0.00001) + x0)
    if x >= 0 and x <= x_max:
        if first_point_found is False:  # this condition means the first point has not been found yet
            x1 = x
            y1 = y_max
            first_point_found = True
        else:
            x2 = x
            y2 = y_max
    y = round((x_min - x0)*np.tan(angle) + y0)
    if y >= 0 and y <= y_max:
        if first_point_found is False:
            x1 = x_min
            y1 = y
            first_point_found = True
        else:
            x2 = x_min
            y2 = y
    y = round((x_max - x0) * np.tan(angle) + y0)
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
