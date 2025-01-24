# pytest unit tests for spinalcordtoolbox.deepseg_.postprocessing

import numpy as np

from spinalcordtoolbox.deepseg_.postprocessing import keep_largest_object


def test_keep_largest_object_2d():
    """
    Test the keep_largest_object function for 2D arrays.
    """
    # Define test cases
    single_object_2d = np.array([
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    two_objects_2d = np.array([
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0]
    ])
    three_objects_2d = np.array([
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1]
    ])

    # Expected result for all cases
    expected_result = single_object_2d

    # Test single object
    result_single = keep_largest_object(single_object_2d, None, None)
    assert np.array_equal(result_single, expected_result)
    # Test two objects
    result_two = keep_largest_object(two_objects_2d, None, None)
    assert np.array_equal(result_two, expected_result)
    # Test three objects
    result_three = keep_largest_object(three_objects_2d, None, None)
    assert np.array_equal(result_three, expected_result)


def test_keep_largest_object_3d():
    """
    Test the keep_largest_object function for 3D arrays.
    """
    # Define test cases
    single_object_3d = np.array([
        [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    ])

    two_objects_3d = np.array([
        [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
        [[0, 1, 1], [0, 0, 0], [0, 0, 0]]
    ])

    three_objects_3d = np.array([
        [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
        [[0, 1, 1], [0, 0, 0], [0, 1, 1]]
    ])

    # Expected result for all cases
    expected_result = single_object_3d

    # Test single object
    result_single = keep_largest_object(single_object_3d, None, None)
    assert np.array_equal(result_single, expected_result)
    # Test two objects
    result_two = keep_largest_object(two_objects_3d, None, None)
    assert np.array_equal(result_two, expected_result)
    # # Test three objects
    result_three = keep_largest_object(three_objects_3d, None, None)
    assert np.array_equal(result_three, expected_result)
