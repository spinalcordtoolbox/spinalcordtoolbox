import pytest
import spinalcordtoolbox.types.centerline as centerline


@pytest.fixture()
def random_array(request):
    return [2, 4, 6]


@pytest.mark.parametrize(('x', 'y', 'z', 'dx', 'dy', 'dz'),
                         [(random_array, random_array, random_array,
                           random_array, random_array, random_array),
                         ])
def test_centerline(x, y, z, dx, dy, dz):
    centerline.Centerline(x, y, z, dx, dy, dz)
