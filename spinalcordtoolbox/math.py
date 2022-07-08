# Functions that perform mathematical operations on an image.


import logging

import numpy as np
from skimage.morphology import erosion, dilation, disk, ball, square, cube
from skimage.filters import threshold_local, threshold_otsu
from scipy.ndimage.filters import gaussian_filter, gaussian_laplace
from scipy.stats import pearsonr, spearmanr
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.segment.mask import median_otsu
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.patch2self import patch2self
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score

from spinalcordtoolbox.image import Image

logger = logging.getLogger(__name__)

ALMOST_ZERO = 0.000000001


def _get_selem(shape, size, dim):
    """
    Create structuring element of desired shape and radius

    :param shape: str: Shape of the structuring element. See available options below in the code
    :param size: int: size of the element.
    :param dim: {0, 1, 2}: Dimension of the array which 2D structural element will be orthogonal to. For example, if
    you wish to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.
    :return: numpy array: structuring element
    """
    # TODO: enable custom selem
    if shape == 'square':
        selem = square(size)
    elif shape == 'cube':
        selem = cube(size)
    elif shape == 'disk':
        selem = disk(size)
    elif shape == 'ball':
        selem = ball(size)
    else:
        ValueError("This shape is not a valid entry: {}".format(shape))

    if not (len(selem.shape) in [2, 3] and selem.shape[0] == selem.shape[1]):
        raise ValueError("Invalid shape")

    # If 2d kernel, replicate it along the specified dimension
    if len(selem.shape) == 2:
        selem3d = np.zeros([selem.shape[0]] * 3)
        imid = np.floor(selem.shape[0] / 2).astype(int)
        if dim == 0:
            selem3d[imid, :, :] = selem
        elif dim == 1:
            selem3d[:, imid, :] = selem
        elif dim == 2:
            selem3d[:, :, imid] = selem
        else:
            raise ValueError("dim can only take values: {0, 1, 2}")
        selem = selem3d
    return selem


def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.

    :param im1 : array-like, bool\
        Any array of arbitrary size. If not boolean, will be converted.
    :param im2 : array-like, bool\
        Any other array of identical size. If not boolean, will be converted.
    :return dice : float\
        Dice coefficient as a float on range [0,1].\
        Maximum similarity = 1\
        No similarity = 0

    .. note::
        The order of inputs for `dice` is irrelevant. The result will be
        identical if `im1` and `im2` are switched.

    Source: https://gist.github.com/JDWarner/6730747
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def dilate(data, size, shape, dim=None):
    """
    Dilate data using ball structuring element

    :param data: Image or numpy array: 2d or 3d array
    :param size: int: If shape={'square', 'cube'}: Corresponds to the length of an edge (size=1 has no effect).\
        If shape={'disk', 'ball'}: Corresponds to the radius, not including the center element (size=0 has no effect).
    :param shape: {'square', 'cube', 'disk', 'ball'}
    :param dim: {0, 1, 2}: Dimension of the array which 2D structural element will be orthogonal to. For example, if\
    you wish to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.
    :return: numpy array: data dilated
    """
    if isinstance(data, Image):
        im_out = data.copy()
        im_out.data = dilate(data.data, size, shape, dim)
        return im_out
    else:
        return dilation(data, selem=_get_selem(shape, size, dim), out=None)


def erode(data, size, shape, dim=None):
    """
    Dilate data using ball structuring element

    :param data: Image or numpy array: 2d or 3d array
    :param size: int: If shape={'square', 'cube'}: Corresponds to the length of an edge (size=1 has no effect).\
    If shape={'disk', 'ball'}: Corresponds to the radius, not including the center element (size=0 has no effect).
    :param shape: {'square', 'cube', 'disk', 'ball'}
    :param dim: {0, 1, 2}: Dimension of the array which 2D structural element will be orthogonal to. For example, if\
    you wish to apply a 2D disk kernel in the X-Y plane, leaving Z unaffected, parameters will be: shape=disk, dim=2.
    :return: numpy array: data dilated
    """
    if isinstance(data, Image):
        im_out = data.copy()
        im_out.data = erode(data.data, size, shape, dim)
        return im_out
    else:
        return erosion(data, selem=_get_selem(shape, size, dim), out=None)


def mutual_information(x, y, nbins=32, normalized=False):
    """
    Compute mutual information

    :param x: 1D numpy.array : flatten data from an image
    :param y: 1D numpy.array : flatten data from an image
    :param nbins: number of bins to compute the contingency matrix (only used if normalized=False)
    :return: float non negative value : mutual information
    """
    if normalized:
        mi = normalized_mutual_info_score(x, y)
    else:
        c_xy = np.histogram2d(x, y, nbins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
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

    if type == 'pearson':
        corr = pearsonr(x, y)[0]
    if type == 'spearman':
        corr = spearmanr(x, y)[0]

    return corr


def smooth(data, sigmas):
    """
    Smooth data by convolving Gaussian kernel
    :param data: input 3D numpy array
    :param sigmas: Kernel SD in voxel
    :return:
    """
    assert len(data.shape) == len(sigmas)
    return gaussian_filter(data.astype(float), sigmas, order=0, truncate=4.0)


def laplacian(data, sigmas):
    """
    Apply Laplacian filter
    """
    assert len(data.shape) == len(sigmas)
    return gaussian_laplace(data.astype(float), sigmas)


def compute_similarity(data1, data2, metric):
    '''
    Compute a similarity metric between two images data

    :param data1: numpy.array 3D data
    :param data2: numpy.array 3D data
    :param fname_out: file name of the output file. Output file should be either a text file ('.txt') or a pickle file ('.pkl', '.pklz' or '.pickle')
    :param metric: 'mi' for mutual information or 'corr' for pearson correlation coefficient
    :return: tuple with computetd results of similarity, data1 flattened array, data2 flattened array
    '''
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
    if metric == 'minorm':
        res = mutual_information(data1_1d, data2_1d, normalized=True)
    if metric == 'corr':
        res = correlation(data1_1d, data2_1d)
    else:
        raise ValueError(f"Don't know what metric to use! Got unsupported: {metric}")

    return res, data1_1d, data2_1d


def otsu(data, nbins):
    thresh = threshold_otsu(data, nbins)
    return data > thresh


def adap(data, block_size, offset):
    mask = data
    for iz in range(data.shape[2]):
        adaptive_thresh = threshold_local(data[:, :, iz], block_size,
                                          method='gaussian', offset=offset)
        mask[:, :, iz] = mask[:, :, iz] > adaptive_thresh
    return mask


def otsu_median(data, size, n_iter):
    data, mask = median_otsu(data, size, n_iter)
    return mask


def threshold(data, lthr=None, uthr=None):
    if lthr is not None:
        data[data < lthr] = 0
    if uthr is not None:
        data[data > uthr] = 0
    return data


def perc(data, perc_value):
    perc = np.percentile(data, perc_value)
    return data > perc


def binarize(data, bin_thr=0):
    return data > bin_thr


def concatenate_along_last_dimension(data):
    """
    Concatenate multiple data arrays, while ensuring that the last axis of the
    array ("N") is safe to use for operations involving "axis=-1" (e.g. `np.sum(axis=-1)`).

      * 3D (X,Y,Z)   -> 4D (X,Y,Z,N)
      * 4D (X,Y,Z,T) -> 5D (X,Y,Z,T,N)
      * 3D + 4D      -> 4D (X,Y,Z,N)

    :param data: List of ndarrays.
    :return data_concat: concatenate([data])
    """
    ndims = set([arr.ndim for arr in data])

    # Case 1: All images have the same ndim, so add a new axis to every image
    if ndims == {3} or ndims == {4}:
        data = [arr[..., np.newaxis] for arr in data]

    # Case 2: Mix of 3D and 4D images --> No longer supported
    elif ndims == {3, 4}:
        raise ValueError(f"Can only process images with the same number of dimensions, but got mix: {ndims}")

    # Case 3: 2D/5D/etc. images --> Not supported
    else:
        raise ValueError(f"Can only process 3D/4D images, but received images with ndim = {ndims - {3,4}}")

    return np.concatenate(data, axis=-1)


def denoise_nlmeans(data_in, patch_radius=1, block_radius=5):
    """
    :param data_in: nd_array to denoise

    .. note::
        for more info about patch_radius and block radius, please refer to the dipy website: http://dipy.org/dipy/reference/dipy.denoise.html#dipy.denoise.nlmeans.nlmeans
    """

    data_in = np.asarray(data_in)

    block_radius_max = min(data_in.shape) - 1
    block_radius = block_radius_max if block_radius > block_radius_max else block_radius

    sigma = estimate_sigma(data_in)
    denoised = nlmeans(data_in, sigma, patch_radius=patch_radius, block_radius=block_radius)

    return denoised


def symmetrize(data, dim):
    """
    Symmetrize data along specified dimension.
    :param data: numpy.array 3D data.
    :param dim: dimension of array to symmetrize along.

    :return data_out: symmetrized data
    """
    data_out = (data + np.flip(data, axis=dim)) / 2.0
    return data_out


def denoise_patch2self(data_in, bvals_in, patch_radius=0, model='ols'):
    """
    :param data_in: 4d array to denoise
    :param bvals_in: b-values associated with the 4D DWI data
    :param patch_radius: radius of the p-neighbourhoods defined in the Patch2Self algorithm
    :param model: regression model required to learn the mapping within Patch2Self

    .. note::
        for more info about patch_radius and model, please refer to the dipy website: https://dipy.org/documentation/1.4.1./examples_built/denoise_patch2self/#example-denoise-patch2self
    """
    denoised = patch2self(data_in, bvals_in, patch_radius=patch_radius,
                          model=model)

    return denoised
