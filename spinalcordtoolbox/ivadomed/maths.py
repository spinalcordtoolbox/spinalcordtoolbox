import numpy as np
import os
import scipy.signal


def rescale_values_array(arr, minv=0.0, maxv=1.0, dtype=np.float32):
    """Rescale the values of numpy array `arr` to be from `minv` to `maxv`.

    Args:
        arr (ndarry): Array whose values will be rescaled.
        minv (float): Minimum value of the output array.
        maxv (float): Maximum value of the output array.
        dtype (type): Cast array to this type before performing the rescaling.
    """
    if dtype is not None:
        arr = arr.astype(dtype)

    mina = np.min(arr)
    maxa = np.max(arr)

    if mina == maxa:
        return arr * minv

    norm = (arr - mina) / (maxa - mina)  # normalize the array first
    return (norm * (maxv - minv)) + minv  # rescale by minv and maxv, which is the normalized array by default


def gaussian_kernel(kernlen=10):
    """
    Create a 2D gaussian kernel with user-defined size.

    Args:
        kernlen (int): size of kernel

    Returns:
        ndarray: a 2D array of size (kernlen,kernlen)
    """

    x = np.linspace(-1, 1, kernlen + 1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return rescale_values_array(kern2d / kern2d.sum())


def heatmap_generation(image, kernel_size):
    """
    Generate heatmap from image containing sing voxel label using
    convolution with gaussian kernel
    Args:
        image (ndarray): 2D array containing single voxel label
        kernel_size (int): size of gaussian kernel

    Returns:
        ndarray: 2D array heatmap matching the label.

    """
    kernel = gaussian_kernel(kernel_size)
    map = scipy.signal.convolve(image, kernel, mode='same')
    return rescale_values_array(map)

