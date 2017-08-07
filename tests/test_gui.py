import logging
import unittest

from scripts import msct_image
from spinalcordtoolbox.gui import propseg, base, labelvertebrae
from spinalcordtoolbox.gui.base import InvalidActionWarning

logging.basicConfig(level=logging.DEBUG)


class PropsegTestCase(unittest.TestCase):
    def setUp(self):
        self.image = msct_image.Image('/Users/geper_admin/sct_example_data/t2/t2.nii.gz')
        self.overlay = msct_image.Image('/Users/geper_admin/manual_propseg.nii.gz')
        self.params = base.AnatomicalParams()

    def _init_auto(self):
        controller = propseg.PropSegController(self.image, self.params, max_points=3)
        controller.align_image()
        controller.reset_position()
        controller.mode = 'AUTO'
        return controller

    def _init_custom(self):
        controller = propseg.PropSegController(self.image, self.params, max_points=3)
        controller.align_image()
        controller.mode = 'CUSTOM'
        return controller

    def test_auto_init(self):
        controller = self._init_auto()
        assert controller.position == (160, 160, 32)
        assert len(controller.points) == 0

    def test_custom_init(self):
        controller = self._init_custom()
        assert controller.position == (160, 160, 32)
        assert len(controller.points) == 0

    def test_select_auto_points(self):
        controller = propseg.PropSegController(self.image, self.params, max_points=3)
        controller.align_image()
        controller.mode = 'AUTO'
        x = controller._slice
        y = 45
        z = 33

        try:
            controller.select_slice(x + 100, y + 55, z + 34)
            assert False, "Can not selected a slice in AUTO mode"
        except InvalidActionWarning:
            pass
        controller.select_point(5, y, z)

        # First point
        assert controller.position == (x, y, z)
        assert controller.points == [(x, y, z, 1)]

        second_point = (20, 51, 11)
        expected = (controller._slice, 51, 11)
        controller.select_point(*second_point)

        # Second point
        assert controller.position == expected
        assert controller.points == [(x, y, z, 1),
                                     (expected[0], expected[1], expected[2], 1)]

        controller.save()
        assert controller.as_string()

    def test_select_custom_points(self):
        controller = self._init_custom()
        x = 200
        y = 146
        z = 44

        try:
            controller.select_point(33, 44, 55)
            assert False, "Expected to fail without a slice selected"
        except ValueError:
            pass

        controller.select_slice(100, 110, 40)
        controller.select_point(x, y + 22, z - 14)
        controller.select_point(x + 44, y, z)

        assert len(controller.points) == 1
        assert controller.points == [(100, y + 22, z - 14, 1)]

        controller.save()
        assert controller.as_string()

    def test_save_labelvertebrae(self):
        controller = labelvertebrae.LabelVertebraeController(self.image, self.params)
        controller.align_image()
        controller.select_slice(2, 45, 45)
        controller.select_point(6, 33, 88)

        # assert len(controller.points) == 1
