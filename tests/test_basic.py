# -*- coding: utf-8 -*-

import pytest
import spinalcordtoolbox.types.centerline as centerline


def test_centerline():
    x = [0, 0, 0, 0, 0, 0]
    y = [0, 0, 0, 0, 0, 0]
    z = [0, 1, 2, 3, 4, 5]
    dx = [0, 0, 0, 0, 0, 0]
    dy = [0, 0, 0, 0, 0, 0]
    dz = [1, 1, 1, 1, 1, 1]
    centerline.Centerline(x, y, z, dx, dy, dz)
