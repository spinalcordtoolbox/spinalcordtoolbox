# pytest unit tests for spinalcordtoolbox.register.algorithms::find_angle_hog

import numpy as np
from skimage.transform import rotate

from spinalcordtoolbox.registration.algorithms import normalized_sobel, weighted_orientation_histogram, circular_autoconvolution


def test_normalized_sobel():
    """Test that normalized_sobel computations are correctly scaled."""
    # 2-dimensional tests
    # generate a rectangular, 6x8 image with a constant rate of change of
    # 2 units/voxel in the first axis, and 3 units/voxel in the second axis
    xx, yy = np.mgrid[0:6, 0:8]
    image = 2*xx + 3*yy
    # use an anisotropic resolution of (7mm, 8mm) per voxel
    assert np.allclose(normalized_sobel(image, axis=0, p=7.0), 2/7)
    assert np.allclose(normalized_sobel(image, axis=1, p=8.0), 3/8)

    # 3-dimensional tests
    # now use a 6x8x10 image, again with a known constant rate of change along
    # each axis, and a resolution of (7mm, 8mm, 9mm) per voxel
    xx, yy, zz = np.mgrid[0:6, 0:8, 0:10]
    image = 2*xx + 3*yy + 4*zz
    assert np.allclose(normalized_sobel(image, axis=0, p=7.0), 2/7)
    assert np.allclose(normalized_sobel(image, axis=1, p=8.0), 3/8)
    assert np.allclose(normalized_sobel(image, axis=2, p=9.0), 4/9)


def test_weighted_orientation_histogram():
    """
    Test weighted_orientation_histogram on an artificial image with a clear
    dividing line, so that all nonzero gradients point in roughly the same
    direction.
    """
    # a 2-dimensional image in RP orientation, with zeros in the P half, and
    # ones in the A half, so that the gradient vectors along the dividing half
    # point in the A direction
    image = np.zeros((30, 30), dtype=float)
    image[:, 15:] = 1.0
    px = py = 1.0

    # the center of the image, which is between the 0.0 half and the 1.0 half
    centermass = (14.5, 14.5)

    # collect the absolute errors between expected and computed angles
    abs_degree_diffs = []
    for degrees in range(360):
        # the skimage angle convention is the opposite from us:
        # we want +90 degrees to point in the L direction
        rotated_image = rotate(image, -degrees, mode='edge')
        hist = weighted_orientation_histogram(rotated_image, px, py, centermass)
        abs_degree_diffs.append(min(
            (degrees - hist.argmax()) % 360,
            (hist.argmax() - degrees) % 360,
        ))
    abs_degree_diffs = np.array(abs_degree_diffs)

    assert abs_degree_diffs.max() <= 3
    assert abs_degree_diffs.mean() <= 1


def test_circular_autoconvolution():
    """
    Test circular_autoconvolution on actually symmetric inputs to check that
    the maximum value of the result is at the right position.
    """
    # axis of symmetry in the middle of a bin
    # even and odd number of bins
    for size in [10, 11]:
        for axis in range(size):
            array = np.zeros(size)
            array[axis] = 1
            assert np.argmax(circular_autoconvolution(array)) == (2*axis) % size

    # axis of symmetry between two bins
    # even and odd number of bins
    for size in [10, 11]:
        for axis in range(size):
            array = np.zeros(size)
            array[axis] = 1
            array[(axis+1) % size] = 1
            assert np.argmax(circular_autoconvolution(array)) == (2*axis + 1) % size
