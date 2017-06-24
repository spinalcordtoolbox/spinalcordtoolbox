from __future__ import absolute_import
import logging

from spinalcordtoolbox.gui import base2 as base

from PyQt4 import QtCore
from PyQt4 import QtGui
from copy import copy


logger = logging.getLogger(__name__)


class PropSegController(object):
    points = []
    slices = []
    MODES = {'AUTO': 'Auto Mode',
             'CUSTOM': 'Custom Mode'}

    def __init__(self, ui_dialog, init_values=None):
        self.dialog = ui_dialog
        if isinstance(init_values, list):
            self.points.extend(init_values)

        elif init_values:
            self.points.append(init_values)

        self.init_points = copy(self.points)

    @staticmethod
    def valid_point(x, y, z):
        if x > 0 and y > 0 and z > 0:
            return True

    def skip_slice(self):
        self.points = self.points

    def undo_point(self):
        logger.debug('Undo pressed')
        if self.points:
            self.points.pop()
        else:
            self.dialog.update_warning('There is no points selected to undo')

    def reset(self):
        self.points = copy(self.init_points)

    def select_point(self, x, y, z):
        logger.debug('Point Selected %d, %d, %d', x, y, z)
        self.dialog.update_status('Point Selected {}, {}, {}'.format(x, y, z))
        if self.valid_point(x, y, z):
            self.points.append((x, y, z, 1))
            # self.dialog.refresh_dialog()

    def select_slice(self, x, y, z):
        if self.mode == 'CUSTOM':
            logger.debug('Slice Selected %d, %d, %d', x, y, z)
            self.dialog.update_status('Slice Selected {}, {}, {}'.format(x, y, z))


class PropSeg(base.BaseDialog):
    def __init__(self, *args, **kwargs):
        self.controller = PropSegController(self)
        super(PropSeg, self).__init__(*args, **kwargs)
        self.main_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        self.second_canvas = base.SagittalCanvas(self)
        self.second_canvas.plot_hslices(self.controller.points)
        layout.addWidget(self.second_canvas)

        self.main_canvas = base.AxialCanvas(self, interactive=True)
        self.main_canvas.plot_points(self.controller.points)
        layout.addWidget(self.main_canvas)

        self.main_canvas.point_selected_signal.connect(self.controller.select_point)
        self.second_canvas.point_selected_signal.connect(self.controller.select_slice)
        self.update_canvas_signal.connect(self.main_canvas.on_refresh_slice)

        parent.addLayout(layout)

        self.main_canvas.point_selected_signal.connect(self.add_point)

    def _init_controls(self, parent):
        group = QtGui.QGroupBox()
        group.setFlat(True)

        layout = QtGui.QHBoxLayout()
        auto_mode = QtGui.QRadioButton('Mode Auto')
        auto_mode.mode = 'AUTO'
        custom_mode = QtGui.QRadioButton('Mode Custom')
        custom_mode.mode = 'CUSTOM'

        auto_mode.toggled.connect(self._toggle_mode)
        custom_mode.toggled.connect(self._toggle_mode)

        layout.addWidget(auto_mode)
        layout.addWidget(custom_mode)
        group.setLayout(layout)
        parent.addWidget(group)
        auto_mode.setChecked(True)

    def _init_footer(self, parent):
        ctrl_layout = super(PropSeg, self)._init_footer(parent)
        skip = QtGui.QPushButton('Skip')
        ctrl_layout.insertWidget(2, skip)

        skip.clicked.connect(self.controller.skip_slice)
        self.btn_undo.clicked.connect(self.controller.undo_point)

    @QtCore.Slot(int, int, int)
    def add_point(self, x, y, z):
        self.update_canvas_signal.emit(x, y, z)

    def _toggle_mode(self):
        widget = self.sender()
        if widget.mode in self.controller.MODES.keys() and widget.isChecked():
            self.controller.mode = widget.mode


if __name__ == '__main__':
    import sys
    import logging
    from scripts.msct_image import Image

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)

    params = base.AnatomicalParams()
    img = Image('/Users/geper_admin/sct_example_data/t2/t2.nii.gz')
    img.change_orientation('SAL')
    base_win = PropSeg(params, img)
    base_win.show()
    app.exec_()
