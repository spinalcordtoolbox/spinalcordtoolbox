from __future__ import absolute_import
from __future__ import division

from PyQt4 import QtGui

import base2 as base


class GroundTruth(base.BaseDialog):

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        sag = base.SagittalCanvas(self)
        cor = base.AxialCanvas(self)
        layout.addWidget(sag)
        layout.addWidget(cor)
        parent.addLayout(layout)

    def _init_controls(self, parent):
        pass

    def _init_toolbar(self, parent):
        pass

    def _init_footer(self, parent):
        ctrl_layout = super(GroundTruth, self)._init_footer(parent)
        skip = QtGui.QPushButton('Skip')
        ctrl_layout.addWidget(skip, -1)


if __name__ == '__main__':
    import sys
    import logging
    from scripts.msct_image import Image

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)

    params = base.AnatomicalParams()
    img = Image('/Users/geper_admin/sct_example_data/t2/t2.nii.gz')
    img.change_orientation('SAL')
    base_win = GroundTruth(params, img)
    base_win.show()
    base_win.activateWindow()
    base_win.raise_()
    app.exec_()
