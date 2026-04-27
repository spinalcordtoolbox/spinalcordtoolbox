# pytest unit tests for spinalcordtoolbox.register.algorithms::find_angle_hog

from math import degrees, radians

import numpy as np
from scipy.ndimage import center_of_mass
from skimage.transform import rotate

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.registration.algorithms import normalized_sobel, weighted_orientation_histogram, circular_autoconvolution, find_angle_hog_2
from spinalcordtoolbox.utils.sys import sct_test_path


def abs_degree_diff(x, y):
    """
    Return the absolute difference between x degrees and y degrees,
    taking into account wrap-around behaviour at 360 degrees.
    """
    return min(
        (x - y) % 360,
        (y - x) % 360,
    )


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
    diffs = []
    for expected_degrees in range(360):
        # the skimage angle convention is the opposite from us:
        # we want +90 degrees to point in the L direction
        rotated_image = rotate(image, -expected_degrees, mode='edge')
        hist = weighted_orientation_histogram(rotated_image, px, py, centermass)
        diffs.append(abs_degree_diff(expected_degrees, hist.argmax()))
    diffs = np.array(diffs)

    assert diffs.max() <= 3
    assert diffs.mean() <= 1


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


def test_find_angle_hog_2():
    """
    Test find_angle_hog on a real image before and after rotating it.
    """
    # load the test image and its segmentation, to get each slice's centermass
    img = Image(sct_test_path('t2/t2.nii.gz')).change_orientation('RPI')
    seg = Image(sct_test_path('t2/t2_seg-manual.nii.gz')).change_orientation('RPI')
    _, _, nz, _, px, py, _, _ = img.dim
    assert px == py, "expecting an isotropic image"
    assert img.dim == seg.dim, "expecting a segmentation which matches the image"

    for iz in range(nz):
        # rotate() only works correctly on floating point data
        original_slice = img.data[:, :, iz].astype(float)
        cx, cy = center_of_mass(seg.data[:, :, iz])

        # we use a wider angle_range because the slice may already be somewhat
        # rotated, and we want to rotate it some more
        kwargs = dict(px=px, py=py, centermass=(cx, cy), angle_range=radians(20))
        original_degrees = degrees(find_angle_hog_2(original_slice, **kwargs))

        diffs = []
        for expected_degrees in range(-10, 11):
            rotated_slice = rotate(
                original_slice,
                -expected_degrees,  # skimage convention is the opposite of ours
                center=(cy, cx),  # skimage convention is the opposite of ours
                mode='edge',
            )
            computed_degrees = degrees(find_angle_hog_2(rotated_slice, **kwargs))
            diffs.append(abs_degree_diff(
                original_degrees + expected_degrees,
                computed_degrees,
            ))
        diffs = np.array(diffs)

        assert diffs.max() <= 16
        assert diffs.mean() <= 8
