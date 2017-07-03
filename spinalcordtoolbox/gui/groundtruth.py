from __future__ import absolute_import
from __future__ import division

import logging

from PyQt4 import QtGui, QtCore

from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui import widgets


logger = logging.getLogger(__name__)


class GroundTruthController(base.BaseController):

    def __init__(self, image, params, init_values=None):
        super(GroundTruthController, self).__init__(image, params, init_values)

    def initialize_dialog(self):
        self._dialog.update_status('1. Select a label -> 2. Select a axial slice -> 3. Select a point in the corrinal plane')

    def select_point(self, x, y, z):
        label = self._dialog.selected_label
        logger.debug('Point Selected {}'.format(self._print_point((x, y, z))))
        self._dialog.update_status('Point Selected {}'.format(self._print_point((x, y, z))))
        if self.valid_point(x, y, z) and label:
            self.points.append((x, y, z, label))
            self._dialog.set_slice(x, y, z)

    def as_image(self):
        pass


class GroundTruth(base.BaseDialog):
    def __init__(self, *args, **kwargs):
        super(GroundTruth, self).__init__(*args, **kwargs)
        self.corrinal_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        self.sagittal_canvas = widgets.SagittalCanvas(self)
        self.sagittal_canvas.plot_hslices()

        self.labels_checkboxes = widgets.VertebraeWidget(self)

        self.corrinal_canvas = widgets.CorrinalCanvas(self)
        self.corrinal_canvas.plot_points()

        layout.addWidget(self.labels_checkboxes)
        layout.addWidget(self.sagittal_canvas)
        layout.addWidget(self.corrinal_canvas)

        self.sagittal_canvas.point_selected_signal.connect(self._controller.select_point)
        parent.addLayout(layout)

    def _init_controls(self, parent):
        pass

    # def _init_footer(self, parent):
    #     ctrl_layout = super(GroundTruth, self)._init_footer(parent)
    #     skip = QtGui.QPushButton('Skip')
    #     ctrl_layout.insertWidget(2, skip)
    #
    #     skip.clicked.connect(self._controller.skip_slice)

    def set_slice(self, x=0, y=0, z=0):
        self.sagittal_canvas.on_refresh_slice(x, y, z)
        self.labels_checkboxes.on_refresh()

    @property
    def selected_label(self):
        return self.labels_checkboxes._selected_label.label


if __name__ == '__main__':
    import sys
    from scripts.msct_image import Image

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)

    params = base.AnatomicalParams()
    img = Image('/Users/geper_admin/sct_example_data/t2/t2.nii.gz')
    img.change_orientation('SAL')
    controller = GroundTruthController(img, params)
    controller.align_image()
    base_win = GroundTruth(controller)
    base_win.show()
    app.exec_()
    print(base_win._controller.as_string())
    base_win._controller.as_image()
    base_win._controller.as_niftii()
