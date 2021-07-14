import logging
import os
import unittest

import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.gui import base, centerline, sagittal
from spinalcordtoolbox.gui.base import InvalidActionWarning
from spinalcordtoolbox.utils import sct_test_path


logging.basicConfig(level=logging.DEBUG)


class CenterlineTestCase(unittest.TestCase):
    def setUp(self):
        self.image = msct_image.Image(sct_test_path('t2', 't2.nii.gz'))
        self.overlay = msct_image.Image(self.image)
        self.params = base.AnatomicalParams()

    def _init_auto(self):
        controller = centerline.CenterlineController(self.image, self.params)
        controller.reformat_image()
        controller.mode = 'AUTO'
        return controller

    def _init_custom(self):
        controller = centerline.CenterlineController(self.image, self.params)
        controller.reformat_image()
        controller.mode = 'CUSTOM'
        return controller

    def test_auto_init(self):
        controller = self._init_auto()
        assert controller.position == (0, 30, 26)
        assert len(controller.points) == 0

    def test_custom_init(self):
        controller = self._init_custom()
        assert controller.position == (0, 30, 26)
        assert len(controller.points) == 0

    def test_select_auto_points(self):
        self.params.num_points = 3
        controller = self._init_auto()
        expected = [(0, 45, 33, 1), (25, 51, 35, 1), (30, 59, 31, 1)]
        points = [(x + self.params.interval_in_mm, y, z) for x, y, z, _ in expected]

        with self.assertRaises(InvalidActionWarning):
            controller.select_slice(*points[0])

        # First point
        x, y, z, _ = expected[0]
        controller.select_point(x, y, z)
        assert controller.position == points[0]
        assert controller.points == expected[0:1]

        # Second point
        x, y, z, _ = expected[1]
        controller.select_point(x, y, z)
        assert controller.position == points[1]
        assert controller.points == expected[:-1]

        # Skip slice
        controller.skip_slice()
        x, y, z, _ = expected[2]
        controller.select_point(x, y, z)
        assert controller.position == points[2]
        assert controller.points == expected

        # Undo
        controller.undo()
        x, y, z, _ = expected[-1]
        assert controller.position == (x, y, z)
        assert controller.points == expected[:-1]


class SagittalTestCase(unittest.TestCase):
    def setUp(self):
        self.image = msct_image.Image(sct_test_path('t2', 't2.nii.gz'))
        self.overlay = msct_image.Image(self.image)
        self.params = base.AnatomicalParams()

    def test_init_sagittal(self):
        controller = sagittal.SagittalController(self.image, self.params)
        controller.reformat_image()

        assert len(controller.points) == 0
        assert controller.position == (27, 30, 26)

    def test_select_L1_label(self):
        controller = sagittal.SagittalController(self.image, self.params)
        controller.reformat_image()
        expected = input_point = (25, 50, 21, 21)
        input_point = (25, 50, 21, 21)

        controller.label = 21
        controller.select_point(*input_point)

        assert controller.points == [expected]
