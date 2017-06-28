from __future__ import absolute_import
from __future__ import division

import logging
import numpy as np
from PyQt4 import QtCore, QtGui

from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui import widgets


logger = logging.getLogger(__name__)


class PropSegController(base.BaseController):
    mode = 'AUTO'
    _slice = 0
    _interval = 15
    _dialog = None
    MODES = ['AUTO', 'CUSTOM']
    _overlay_image = None

    def __init__(self, image, params, init_values=None):
        super(PropSegController, self).__init__(image, params, init_values)

    def align_image(self):
        super(PropSegController, self).align_image()
        self.init_x = self._slice
        if self.image.dim[0] < self._interval:
            self._interval = 1
        self._max_points = self.image.dim[0] / self._interval

    def initialize_dialog(self):
        """Set the dialog with default data"""
        self._dialog.update_status('Hello World')

    def valid_point(self, x, y, z):
        dim = self.image.dim
        if 0 <= x < dim[0] and 0 <= y < dim[1] and 0 <= z < dim[2]:
            return True
        return False

    def skip_slice(self):
        if self.mode == 'AUTO':
            logger.debug('Advance slice from {} to {}'.format(self._slice, self._interval + self._slice))
            self._slice += self._interval
            if self._slice >= self.image.dim[0]:
                self._dialog.update_warning('Reached the maximum superior / inferior axis length')
            else:
                self._dialog.set_slice(self._slice, self.init_y, self.init_z)

    def on_undo(self):
        """Remove the last point selected and refresh the UI"""
        if self.points:
            point = self.points[-1]
            self.points = self.points[:-1]
            self._slice = point[0]
            if self.valid_point(point[0], point[1], point[2]):
                self._dialog.set_slice(point[0], point[1], point[2])
            logger.debug('Point removed {}'.format(point))
        else:
            self._dialog.update_warning('There is no points selected to undo')

    def _print_point(self, point):
        max_x = self.image.dim[0]
        return '{} {} {}'.format(max_x - point[0], point[1], point[2])

    def select_point(self, x, y, z):
        logger.debug('Point Selected {}'.format(self._print_point((x, y, z))))
        self._dialog.update_status('Point Selected {}'.format(self._print_point((x, y, z))))
        if self.valid_point(x, y, z):
            self.points.append((x, y, z, 1))
            if self.mode == 'AUTO':
                self._slice += self._interval
                if self.valid_point(self._slice, y, z):
                    self._dialog.set_slice(self._slice, self.init_y, self.init_z)
                else:
                    self._dialog.update_warning('Reached the maximum superior / inferior axis length')

            else:
                self._dialog.set_slice(x, y, z)

    def select_slice(self, x, y, z):
        if self.mode == 'CUSTOM':
            logger.info('Slice Selected {}'.format(self._print_point((x, y, z))))
            self._dialog.update_status('Slice Selected {}'.format(self._print_point((x, y, z))))
            self._dialog.set_slice(x, y, z)

    def save_quit(self):
        self._overlay_image = self.image.copy()
        self._overlay_image.data *= 0
        for point in self.points:
            self._overlay_image.data[point[0], point[1], point[2]] = 1

        self._overlay_image.change_orientation(self.orientation)
        self._dialog.close()

    def as_string(self):
        if not self._overlay_image:
            logger.warning('There is no information to save')
            return ''
        output = []
        xs, ys, zs = np.where(self._overlay_image.data)
        for x, y, z in zip(xs, ys, zs):
            output.append('{},{},{},{}'.format(x, y, z, 1))
        return ':'.join(output)

    def as_niftii(self, file_name=None):
        if not self._overlay_image:
            logger.warning('There is no information to save')
            raise IOError('There is no information to save')
        if not file_name:
            file_name = 'manual_propseg.nii.gz'
        print(np.where(self._overlay_image.data))
        self._overlay_image.setFileName(file_name)
        self._overlay_image.save()


class PropSeg(base.BaseDialog):
    def __init__(self, *args, **kwargs):
        super(PropSeg, self).__init__(*args, **kwargs)
        self.main_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        self.second_canvas = widgets.SagittalCanvas(self)
        self.second_canvas.plot_hslices()
        layout.addWidget(self.second_canvas)

        self.main_canvas = widgets.AxialCanvas(self, interactive=True)
        self.main_canvas.plot_points(self._controller.points)
        layout.addWidget(self.main_canvas)

        self.main_canvas.point_selected_signal.connect(self._controller.select_point)
        self.second_canvas.point_selected_signal.connect(self._controller.select_slice)

        parent.addLayout(layout)

    def _init_controls(self, parent):
        group = QtGui.QGroupBox()
        group.setFlat(True)

        layout = QtGui.QHBoxLayout()
        auto_mode = QtGui.QRadioButton('Mode Auto')
        auto_mode.setToolTip('Automatically move down the axis slice on the sagittal plane')
        auto_mode.mode = 'AUTO'
        custom_mode = QtGui.QRadioButton('Mode Custom')
        custom_mode.setToolTip('Manually select the axis slice on sagittal plane')
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

        skip.clicked.connect(self._controller.skip_slice)
        self.btn_undo.clicked.connect(self._controller.on_undo)
        self.btn_ok.clicked.connect(self._controller.save_quit)

    def set_slice(self, x=0, y=0, z=0):
        self.main_canvas.on_refresh_slice(x, y, z)
        self.second_canvas.on_refresh_slice(x, y, z)

    def _toggle_mode(self):
        widget = self.sender()
        if widget.mode in self._controller.MODES and widget.isChecked():
            self._controller.mode = widget.mode


if __name__ == '__main__':
    import sys
    from scripts.msct_image import Image

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)

    try:
        file_name = sys.argv[1]
    except Exception:
        file_name = '/Users/geper_admin/sct_example_data/t2/t2.nii.gz'

    params = base.AnatomicalParams()
    img = Image(file_name)
    controller = PropSegController(img, params)
    controller.align_image()
    base_win = PropSeg(controller)
    base_win.show()
    app.exec_()
    print(base_win._controller.as_string())
    base_win._controller.as_niftii()
