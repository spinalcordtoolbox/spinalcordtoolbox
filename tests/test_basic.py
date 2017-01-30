import pytest
import spinalcordtoolbox.types.centerline as centerline


@pytest.mark.parametrize(('x', 'y', 'z', 'dx', 'dy', 'dz'),
                         [(3, 5, 7, .4, .5, .8),
                          (5, 52, 77, .6, .4, .6)])
def test_centerline(x, y, z, dx, dy, dz):
    centerline.Centerline(x, y, z, dx, dy, dz)
