import logging
import unittest

from scripts import msct_image
from spinalcordtoolbox.gui import propseg, base, labelvertebrae, groundtruth
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
        controller.mode = 'AUTO'
        return controller

    def _init_custom(self):
        controller = propseg.PropSegController(self.image, self.params, max_points=3)
        controller.align_image()
        controller.mode = 'CUSTOM'
        return controller

    def test_auto_init(self):
        controller = self._init_auto()
        assert controller.position == (15, 160, 32)
        assert len(controller.points) == 0

    def test_custom_init(self):
        controller = self._init_custom()
        assert controller.position == (160, 160, 32)
        assert len(controller.points) == 0

    def test_select_auto_points(self):
        controller = propseg.PropSegController(self.image, self.params, max_points=3)
        controller.align_image()
        controller.mode = 'AUTO'
        expected = [(15, 45, 33, 1), (30, 51, 35, 1), (60, 71, 31, 1)]
        points = [(x + controller.INTERVAL, y, z) for x, y, z, _ in expected]

        with self.assertRaises(InvalidActionWarning):
            controller.select_slice(*points[0])

        # First point
        x, y, z, _ = expected[0]
        controller.select_point(5, y, z)
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
        controller.select_point(x - 32, y, z)
        assert controller.position == points[2]
        assert controller.points == expected

        # Undo
        controller.undo()
        x, y, z, _ = expected[-1]
        assert controller.position == (x, y, z)
        assert controller.points == expected[:-1]

        controller.save()
        assert controller.as_string()

    def test_select_custom_points(self):
        controller = self._init_custom()
        input_slice = (100, 110, 40)
        input_point = (200, 146, 44)
        input_point_2 = (155, 71, 33)
        expected = (100, 146, 44, 1)

        with self.assertRaises(InvalidActionWarning):
            controller.select_point(33, 44, 55)

        controller.select_slice(*input_slice)
        controller.select_point(*input_point)

        with self.assertRaises(InvalidActionWarning):
            controller.select_point(*input_point_2)

        assert controller.points == [expected]

        controller.save()
        assert controller.as_string()


class LabelVertebraeTestCase(unittest.TestCase):
    def setUp(self):
        self.image = msct_image.Image('/Users/geper_admin/sct_example_data/t2/t2.nii.gz')
        self.overlay = msct_image.Image('/Users/geper_admin/manual_propseg.nii.gz')
        self.params = base.AnatomicalParams()

    def test_init_labelvertebrae(self):
        controller = labelvertebrae.LabelVertebraeController(self.image, self.params)
        controller.align_image()

        assert len(controller.points) == 0
        assert controller.position == (160, 160, 32)

    def test_select_t2_vertebrae(self):
        controller = labelvertebrae.LabelVertebraeController(self.image, self.params, self.overlay)
        controller.align_image()
        expected = (160, 123, 36, 10)
        input_point = (160, 123, 36, 10)

        controller.label = 10
        controller.select_point(*input_point)

        assert controller.points == [expected]


class GroundtruthTestCase(unittest.TestCase):
    def setUp(self):
        self.image = msct_image.Image('/Users/geper_admin/sct_example_data/t2/t2.nii.gz')
        self.overlay = msct_image.Image('/Users/geper_admin/manual_propseg.nii.gz')
        self.params = base.AnatomicalParams()

    def test_init_groundtruth(self):
        controller = groundtruth.GroundTruthController(self.image, self.params)
        controller.align_image()

        assert len(controller.points) == 0
        assert controller.position == (160, 160, 32)

    def test_select_L1_label(self):
        controller = groundtruth.GroundTruthController(self.image, self.params)
        controller.align_image()
        expected = (99, 89, 21, 21)
        input_point = (99, 89, 21)

        controller.label = 21
        controller.select_point(*input_point)

        assert controller.points == [expected]
