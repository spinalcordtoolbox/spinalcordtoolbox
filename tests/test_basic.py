# -*- coding: utf-8 -*-
import spinalcordtoolbox.types.centerline as centerline


def test_centerline():
    x = [0, 0, 0, 0, 0, 0]
    y = [0, 0, 0, 0, 0, 0]
    z = [0, 1, 2, 3, 4, 5]
    dx = [0, 0, 0, 0, 0, 0]
    dy = [0, 0, 0, 0, 0, 0]
    dz = [1, 1, 1, 1, 1, 1]
    cl = centerline.Centerline(x, y, z, dx, dy, dz)
    cl.find_nearest_index([1, 1, 1])
    cl.get_plan_parameters(2)
    cl.compute_coordinate_system(3)
    cl.find_nearest_indexes([2, 2, 3])
