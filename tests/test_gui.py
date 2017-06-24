import sys
import unittest

from PyQt4.QtCore import Qt, QPoint
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest

from scripts import msct_image
from spinalcordtoolbox.gui import propseg, base2


app = QApplication(sys.argv)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        image = msct_image.Image('/Users/geper_admin/sct_example_data/t2/t2.nii.gz')
        params = base2.AnatomicalParams()
        self.dialog = propseg.PropSeg(params, image)

    def test_startup(self):
        QTest.mouseClick(self.dialog.btn_ok, Qt.LeftButton)
        assert self.dialog.controller.points == []

    def test_click_point(self):
        # QTest.mouseClick(self.dialog.main_canvas, Qt.LeftButton, pos=QPoint(50, 50))
        QTest.mousePress(self.dialog.main_canvas, Qt.LeftButton)
        QTest.mouseRelease(self.dialog.main_canvas, Qt.LeftButton, pos=QPoint(50, 50))
        QTest.mouseClick(self.dialog.btn_ok, Qt.LeftButton)
        assert self.dialog.controller.points == [(5, 5, 29, 'toto')]
