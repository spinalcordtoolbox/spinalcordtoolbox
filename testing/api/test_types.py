# pytest unit tests for spinalcordtoolbox.types

from spinalcordtoolbox.types import Coordinate


def test_coordinate_does_not_mutate_input_list():
    """
    Regression test: Coordinate() must not modify the caller's list when
    appending the default value to a 3-element coordinate.
    """
    coord = [1, 2, 3]
    c = Coordinate(coord)

    assert coord == [1, 2, 3]
    assert (c.x, c.y, c.z, c.value) == (1, 2, 3, 0)
