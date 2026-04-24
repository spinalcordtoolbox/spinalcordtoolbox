# pytest unit tests for spinalcordtoolbox.register.algorithms::find_angle_hog

import numpy as np

from spinalcordtoolbox.registration.algorithms import normalized_sobel


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
