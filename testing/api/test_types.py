# pytest unit tests for spinalcordtoolbox.types

import pytest

from spinalcordtoolbox.types import Coordinate


def test_coordinate_from_list_of_three():
    coord = Coordinate([1, 2, 3])
    assert (coord.x, coord.y, coord.z, coord.value) == (1, 2, 3, 0)


def test_coordinate_from_list_of_four():
    coord = Coordinate([1, 2, 3, 7])
    assert (coord.x, coord.y, coord.z, coord.value) == (1, 2, 3, 7)


def test_coordinate_does_not_mutate_input_list():
    # A 3-element list passed to Coordinate() must not be mutated in place
    # (issue #5220): building a Coordinate should not append a value to the
    # caller's list.
    src = [1, 2, 3]
    Coordinate(src)
    assert src == [1, 2, 3]


def test_coordinate_from_string():
    coord = Coordinate("1,2,3")
    assert (coord.x, coord.y, coord.z, coord.value) == (1, 2, 3, 0)


@pytest.mark.parametrize("bad", [[1, 2], [1, 2, 3, 4, 5]])
def test_coordinate_rejects_wrong_length(bad):
    with pytest.raises(ValueError):
        Coordinate(bad)
