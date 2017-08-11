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

    def select_point(self, x, y, z):
        logger.debug('Point Selected {}'.format((x, y, z, self._label)))
        if self.valid_point(x, y, z) and self._label:
            self.points.append((x, y, z, self._label))

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value


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

    def set_slice(self, x=0, y=0, z=0):
        self.sagittal_canvas.on_refresh_slice(x, y, z)
        self.labels_checkboxes.refresh()

    @property
    def selected_label(self):
        return self.labels_checkboxes._active_label.label


if __name__ == '__main__':
    import os
    import sys
    from scripts.msct_image import Image

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)

    try:
        file_name = sys.argv[1]
        overlay_name = sys.argv[2]
    except Exception:
        file_name = '/Users/geper_admin/sct_example_data/t2/t2.nii.gz'
        overlay_name = '/Users/geper_admin/manual_propseg.nii.gz'

    params = base.AnatomicalParams()
    params.init_message = '1. Select a label -> 2. Select a axial slice -> 3. Select a point in the corrinal plane'
    img = Image(file_name)
    if os.path.exists:
        overlay = Image(overlay_name)
    controller = GroundTruthController(img, params)
    controller.align_image()
    base_win = GroundTruth(controller)
    base_win.show()
    app.exec_()
    print(base_win._controller.as_string())
    base_win._controller.as_niftii()
