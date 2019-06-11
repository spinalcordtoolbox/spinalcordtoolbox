from __future__ import division, absolute_import, print_function

import sys, os, shutil
from math import asin, cos, sin, acos, pi, atan2, atan, floor
import numpy as np

from scipy.ndimage import convolve
from scipy.io import loadmat
from scipy.signal import medfilt, find_peaks_cwt, argrelmax
from skimage.filters import sobel_h, sobel_v
from skimage.draw import line
from nibabel import load, Nifti1Image, save
from skimage import feature as ft
from scipy.ndimage.filters import gaussian_filter

from spinalcordtoolbox.image import Image
import sct_utils as sct
from msct_parser import Parser

import matplotlib.pyplot as plt
import matplotlib

import skimage.morphology as morph



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

def create_proba_map(segmentation, pixdim):
    """segmentation is a binary numpy array, pixdim is a mean voxel dimension"""

    constant = 0.5

    proba_map = segmentation
    coeff = 1
    # TODO : maybe do the dil slice by slice
    # TODO : does not take into account x/y pixdim difference

    while True:
        coeff = coeff * np.exp(-constant/pixdim)
        if coeff <= 0.01:
            break
        proba_map = proba_map + coeff*(morph.binary_dilation(segmentation) - segmentation)
        segmentation = morph.dilation(segmentation)

    return proba_map

def find_angle(image, segmentation, px, py, method, angle_range=None, return_centermass=False, save_figure_path=None):

    from msct_register import compute_pca
    _, pca, centermass = compute_pca(segmentation)

    if method is "pca":

        pca_eigenratio_th = 1.6

        eigenv = pca.components_.T[0][0], pca.components_.T[1][0]  # pca_src.components_.T[0]
        # # Make sure first element is always positive (to prevent sign flipping)
        # if eigenv[0] <= 0:
        #     eigenv = tuple([i * (-1) for i in eigenv])
        arccos_angle = np.dot(eigenv, [1, 0]) / np.linalg.norm(eigenv)
        arccos_angle = 1.0 if arccos_angle > 1.0 else arccos_angle
        arccos_angle = -1.0 if arccos_angle < -1.0 else arccos_angle
        sign_angle = np.sign(np.cross(eigenv, [1, 0]))
        angle_found = sign_angle * acos(arccos_angle)
        # check if ratio between the two eigenvectors is high enough to prevent poor robustness
        if pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1] < pca_eigenratio_th:
            angle_found = 0
        if angle_range is not None:
            if angle_found < -angle_range * pi/180 or angle_found > angle_range * pi/180:
                angle_found = 0

        conf_score = round(pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1], 2)

    elif method is "hog":

        sigma = 1
        sigmax = sigma / px
        sigmay = sigma / py
        nb_bin = 360
        if nb_bin % 2 != 0:
            nb_bin = nb_bin - 1
        kmedian_size = 5
        angle_range = angle_range
        conf_treshold = 1

        nx, ny = image.shape

        xx, yy = np.mgrid[:nx, :ny]
        seg_weighted_mask = np.exp(
            -(((xx - centermass[0]) ** 2) / (2 * (sigmax ** 2)) + ((yy - centermass[1]) ** 2) / (2 * (sigmay ** 2))))

        grad_orient_histo, proba_map, orient_image = gradient_orientation_histogram(image, nb_bin=nb_bin, seg_weighted_mask=seg_weighted_mask, return_image=True, return_orient=True)

        edges_hist = np.linspace(-(pi - pi / nb_bin), (pi - pi / nb_bin), nb_bin)
        repr_hist = np.linspace(-(pi - 2 * pi / nb_bin), (pi - 2 * pi / nb_bin), nb_bin - 1)

        grad_orient_histo_smooth = circular_filter_1d(grad_orient_histo, kmedian_size, filter='median')  # fft than square than ifft to calculate convolution

        # hog_fft2 = np.fft.rfft(grad_orient_histo_smooth) ** 2
        # grad_orient_histo_conv = np.real(np.fft.irfft(hog_fft2))  # TODO implemente circular conv to be sure it does the same thing
        grad_orient_histo_conv = np.convolve(grad_orient_histo_smooth, grad_orient_histo_smooth, mode='same')

        # grad_orient_histo_conv_reordered = np.zeros(nb_bin)  # wtf is the point of this => hog conv with angle [0 pi] instead of [-pi/2 pi/2] like we want
        # grad_orient_histo_conv_reordered[0:nb_bin//2] = grad_orient_histo_conv[nb_bin//2:nb_bin]
        # grad_orient_histo_conv_reordered[nb_bin//2:nb_bin] = grad_orient_histo_conv[0:nb_bin//2]
        grad_orient_histo_conv_reordered = grad_orient_histo_conv

        index_restrain = int(np.ceil(np.true_divide(angle_range, 180) * nb_bin))
        center = (nb_bin - 1) // 2

        grad_orient_histo_conv_restrained = grad_orient_histo_conv_reordered[center - index_restrain + 1:center + index_restrain + 1]

        index_angle_found = np.argmax(grad_orient_histo_conv_restrained) + (nb_bin // 2 - index_restrain)
        angle_found = repr_hist[index_angle_found] / 2
        angle_found_score = np.amax(grad_orient_histo_conv_restrained)

        conf_score = angle_found_score / np.mean(grad_orient_histo_conv_reordered)
        # sct.printv("confidence score is : " + str(confidence_score) + "\n angle associated is : " + str(angles_sorted[0] * 180/pi))
        # sct.printv("confidence score is : " + str(confidence_score) + "\n angle associated is : " + str(angle_found *180/pi))

        # Plotting stuff :

        if save_figure_path is not None:

            # matplotlib.use("Agg")
            plt.figure(figsize=(6.4*2, 4.8*2))
            plt.suptitle("angle found is : " + str(np.round(angle_found * 180/pi, 1)) + " with conf score = " + str(conf_score))
            plt.subplot(241)
            plt.imshow(np.max(image) - image, cmap="Greys")
            plt.title("image")
            plt.subplot(242)
            plt.imshow(proba_map)
            plt.title("weighting map")
            plt.colorbar()
            plt.subplot(243)
            plt.imshow(np.multiply(orient_image, proba_map > 0) * 180/pi, cmap="hsv")
            plt.colorbar()
            plt.title("Orientation map")
            plt.subplot(244)
            plt.bar(repr_hist * 180/pi, grad_orient_histo, width=0.8 * 360/nb_bin)
            plt.xlabel("angle")
            plt.title("Orientation of weighted gradient histogram")
            plt.subplot(245)
            plt.bar(repr_hist * 180/pi, grad_orient_histo_smooth, width=0.8 * 360/nb_bin)
            plt.xlabel("angle")
            plt.title("Orientation of weighted gradient \n histogram smoothed")
            plt.subplot(246)
            plt.plot(repr_hist * 90/pi, grad_orient_histo_conv_reordered)
            plt.xlabel("angle")
            plt.title("Convolution of the histogram")
            plt.subplot(247)
            plt.plot(repr_hist[center - index_restrain+1:center + index_restrain + 1] * 90/pi, grad_orient_histo_conv_restrained)
            plt.xlabel("angle")
            plt.title("Convolution of the histogram restrained to angle range")
            plt.subplot(248)
            plt.imshow(np.amax(image) - generate_2Dimage_line(image, centermass[0], centermass[1], angle_found), cmap="Greys")
            plt.title("Image with axis of rotation superposed")
            plt.show()
            plt.savefig(save_figure_path.split(".")[0] + ".png", format="png")
            plt.close()

        if conf_score < conf_treshold:
            sct.printv("confidence score bellow threshold")
            angle_found = 0

    else:
        raise Exception("method " + method + " not implemented")

    if return_centermass:
        return angle_found, conf_score, centermass
    else:
        return angle_found, conf_score



def gradient_orientation_histogram(image, nb_bin, grad_ksize=123456789, seg_weighted_mask=None, return_image=False, return_orient=False):  # TODO implement selection of gradient's kernel size, sure that is pertinent ? check wikip image gradient
                                                                                                    # sun et si original say that by increasing the kernel size we reduce the (0, 45, 90) effect

    """ This function takes an image as an input and return its orientation histogram
    inputs :
        - image : the image to compute the orientation histogram from, a 2D numpy array
        - nb_bin : the number of bins of the histogram, an int
        - grad_ksize : kernel size of gradient (work in progress)
    outputs :
        - hog_ancest : the histogram of the orientations of the image, a 1D numpy array of length nb_bin"""

    h_kernel = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]]) / 4.0  # TODO this is sobel operator, test prewits ? others ?
                                                    # this filter is actually separable, maybe could speed up the computation
                                                        # TODO laplacian filter, detect zero? less sensitive to noise
    v_kernel = h_kernel.T

    # x and y gradients
    gradx = convolve(image, v_kernel)
    grady = convolve(image, h_kernel)
    # orientation gradient
    orient = np.arctan2(grady, gradx)  # results are in the range -pi pi

    # weight by gradient magnitude :  this step seems dumb, it alters the angles
    # actually it can be smart but by doing a weighted histogram, not weight the image

    grad_mag = ((np.abs(gradx.astype(object)) ** 2 + np.abs(grady.astype(object)) ** 2) ** 0.5)
    if np.max(grad_mag) != 0:
        grad_mag = grad_mag / np.max(grad_mag)  # to have map between 0 and 1
    # TODO: weird data type manipulation, to explain

    if seg_weighted_mask is not None:
        weighting_map = np.multiply(seg_weighted_mask, grad_mag)  # include weightning by segmentation
    else:
        weighting_map = grad_mag

    # uncomment following line to have vanilla Sun et al. method
    weighting_map = np.ones(grad_mag.shape)
    # compute histogram :
    grad_orient_histo = np.histogram(np.concatenate(orient), bins=nb_bin-1, range=(-(pi-pi/nb_bin), (pi-pi/nb_bin)), weights=np.concatenate(weighting_map))  # check param density that permits outputting a distribution that has integral of 1
    grad_mag = (grad_mag * 255).astype(float).round()  # just for debbuguing purpose (visualisation)
    if seg_weighted_mask is not None:
        seg_weighted_mask = (seg_weighted_mask * 255).astype(float).round()
    weighting_map = (weighting_map * 255).astype(float).round()

    if return_image:
        if return_orient:
            return grad_orient_histo[0].astype(float), weighting_map, orient
        else:
            return grad_orient_histo[0].astype(float), weighting_map
    else:
        return grad_orient_histo[0].astype(float)  # return only the values of the bins, not the bins (we know them)


def compute_similarity_metric(array1, array2, metric="Dice"):

    if array1.shape != array2.shape:
        raise Exception("The 2 image do not gave the same dimension")

    if (array1 > 0).sum() + (array2 > 0).sum() == 0:
        dice_coeff = 1
    else:
        dice_coeff = np.true_divide(2*(np.logical_and(array1, array2) > 0).sum(), (array1 > 0).sum() + (array2 > 0).sum())

    return dice_coeff

def generate_2Dimage_line(image, x0, y0, angle, value=0):

    """ This function takes an image and a line (defined by a point and an angle) as inputs and outputs the same image
    but with the line drawn on it
    inputs :
        - image : image to draw the line on, 2D numpy array
        - x0 and y0 : coordinates of one point the line passes through, two ints
        - angle : angle the lines makes with  x axis
    outputs :
        - image_wline : base image with the line drawn on it, 2D numpy array"""

    # angle = angle *pi/180  # converting to radians

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

    if value == 0:
        image[coord_linex, coord_liney] = np.amax(image)  # put the line at full intensity (not really elegant)
    else:
        image[coord_linex, coord_liney] = value
    # actually the "copy" is not useful, just used to clarify, because python does not make an actual copy when you do
    # this

    return image


def visu3d(array3d, axis=2):

    class IndexTracker(object):
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('use scroll wheel to navigate images')

            self.X = X
            rows, cols, self.slices = X.shape
            self.ind = self.slices // 2

            if axis == 2:
                self.im = ax.imshow(self.X[:, :, self.ind], cmap='Greys')
            elif axis == 1:
                self.im = ax.imshow(self.X[:, self.ind, :], cmap='Greys')
            else:
                self.im = ax.imshow(self.X[self.ind, :, :], cmap='Greys')
            self.update()

        def onscroll(self, event):
            # print("%s %s" % (event.button, event.step))
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.update()

        def update(self):
            if axis == 2:
                self.im.set_data(self.X[:, :, self.ind])
            elif axis == 1:
                self.im.set_data(self.X[:, self.ind, :])
            else:
                self.im.set_data(self.X[self.ind, :, :])
            self.ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()

    array3d_np = np.array(array3d)
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, array3d_np)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

    return fig, ax, tracker
