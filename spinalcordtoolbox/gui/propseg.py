from __future__ import absolute_import
from __future__ import division

import logging

from PyQt4 import QtCore, QtGui

from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui import widgets


logger = logging.getLogger(__name__)


class PropSegController(base.BaseController):
    mode = 'AUTO'
    _slice = 0
    _interval = 15
    MODES = ['AUTO', 'CUSTOM']

    def __init__(self, image, params, init_values=None, max_points=0):
        super(PropSegController, self).__init__(image, params, init_values, max_points)

    def align_image(self):
        super(PropSegController, self).align_image()
        self.init_x = self._slice
        if self.image.dim[0] < self._interval:
            self._interval = 1
        self._max_points = self.image.dim[0] / self._interval

    def initialize_dialog(self):
        """Set the dialog with default data"""
        self._dialog.update_status('Propergation segmentation manually ')

    def skip_slice(self):
        if self.mode == 'AUTO':
            logger.debug('Advance slice from {} to {}'.format(self._slice, self._interval + self._slice))
            self._slice += self._interval
            if self._slice >= self.image.dim[0]:
                self._dialog.update_warning('Reached the maximum superior / inferior axis length')
            else:
                self._dialog.set_slice(self._slice, self.init_y, self.init_z)

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
        self.main_canvas.plot_points()
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
        overlay_name = sys.argv[2]
    except Exception:
        file_name = '/Users/geper_admin/sct_example_data/t2/t2.nii.gz'
        overlay_name = '/Users/geper_admin/manual_propseg.nii.gz'

    params = base.AnatomicalParams()
    img = Image(file_name)
    overlay = Image(overlay_name)
    controller = PropSegController(img, params, overlay)
    controller.align_image()
    base_win = PropSeg(controller)
    base_win.show()
    app.exec_()
    print(base_win._controller.as_string())
    base_win._controller.as_niftii()
