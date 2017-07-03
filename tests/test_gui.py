import sys
import unittest

from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest

from scripts import msct_image
from spinalcordtoolbox.gui import propseg, base


app = QApplication(sys.argv)


class PropsegTestCase(unittest.TestCase):
    def setUp(self):
        image = msct_image.Image('/Users/geper_admin/sct_example_data/t2/t2.nii.gz')
        overlay = msct_image.Image('/Users/geper_admin/manual_propseg.nii.gz')
        params = base.AnatomicalParams()
        self.controller = propseg.PropSegController(image, params, overlay, max_points=3)
        self.controller.align_image()
        self.dialog = propseg.PropSeg(self.controller)

    def test_startup(self):
        QTest.mouseClick(self.dialog.btn_ok, Qt.LeftButton)
        assert self.dialog._controller.as_string() == self.controller.as_string()

    def test_overload_points(self):
        self.controller.select_slice(1, 2, 4)
        self.controller.select_slice(2, 4, 6)
        self.controller.select_slice(11, 22, 33)
        self.controller.select_slice(12, 32, 55)
