from __future__ import absolute_import
from __future__ import division

import logging

from PyQt4 import QtCore, QtGui

from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui import widgets
from spinalcordtoolbox.gui.base import TooManyPointsWarning, InvalidActionWarning

logger = logging.getLogger(__name__)


class PropSegController(base.BaseController):
    _mode = 'AUTO'
    INTERVAL = 15
    MODES = ['AUTO', 'CUSTOM']

    def __init__(self, image, params, init_values=None, max_points=0):
        super(PropSegController, self).__init__(image, params, init_values, max_points)
        self._slice = self.INTERVAL

    def align_image(self):
        super(PropSegController, self).align_image()
        self.init_x = self._slice
        if self.image.dim[0] < self.INTERVAL:
            self.INTERVAL = 1
        if not self._max_points:
            self._max_points = self.image.dim[0] / self.INTERVAL

    def reset_position(self):
        super(PropSegController, self).reset_position()
        if self.mode == 'AUTO':
            self.position = (self._slice, self.position[1], self.position[2])

    def initialize_dialog(self):
        """Set the dialog with default data"""
        self._dialog.update_status('1. Select saggital slice -> 2. Select the Axial center of the spinalcord')

    def skip_slice(self):
        if self.mode == 'AUTO':
            logger.debug('Advance slice from {} to {}'.format(self._slice,
                                                              self.INTERVAL + self._slice))
            self._slice += self.INTERVAL
            if self._slice >= self.image.dim[0]:
                self._slice -= self.INTERVAL
                raise TooManyPointsWarning()

    def select_point(self, x, y, z):
        x = self._slice
        if not self.valid_point(x, y, z):
            raise ValueError('Invalid point selected {}'.format((x, y, z)))

        logger.debug('Point Selected {}'.format((x, y, z)))

        existing_points = [i for i, j in enumerate(self.points) if j[0] == x]
        if existing_points:
            self.points[existing_points[0]] = (x, y, z, 1)
        else:
            if len(self.points) >= self._max_points:
                raise TooManyPointsWarning()
            self.points.append((x, y, z, 1))
        self.position = (x, y, z)
        self._next_slice()

    def _next_slice(self):
        if self.mode == 'AUTO':
            self._slice += self.INTERVAL

    def select_slice(self, x, y, z):
        if self.mode != 'CUSTOM':
            raise InvalidActionWarning('Can only select a slice in CUSTOM mode')

        if not self.valid_point(x, y, z):
            raise ValueError('Invalid slice selected {}'.format((x, y, z)))

        logger.debug('Slice Selected {}'.format((x, y, z)))
        self.position = (x, self.position[1], self.position[2])
        self._slice = x

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in self.MODES:
            raise ValueError('Invalid mode %', value)

        if value != self._mode:
            self._mode = value
            self.points = []
            self.reset_position()


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

        self.main_canvas.point_selected_signal.connect(self.select_point)
        self.second_canvas.point_selected_signal.connect(self.on_select_slice)

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

        auto_mode.toggled.connect(self.on_toggle_mode)
        custom_mode.toggled.connect(self.on_toggle_mode)

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

    def on_toggle_mode(self):
        widget = self.sender()
        if widget.mode in self._controller.MODES and widget.isChecked() and widget.mode != self._controller.mode:
            self._controller.mode = widget.mode
            self.main_canvas.refresh()
            self.second_canvas.refresh()
            self.update_status('Reset manual segmentation: Now in mode {}'.format(widget.mode))

    def on_select_slice(self, x, y, z):
        try:
            logger.debug('Slice clicked {}'.format((x, y, z)))
            self._controller.select_slice(x, y, z)
            self.main_canvas.refresh()
            self.second_canvas.refresh()
            self.update_status('Sagittal slice seleted: {}'.format(self._controller._slice))
        except (TooManyPointsWarning, InvalidActionWarning) as err:
            self.update_warning(err.message)

    def select_point(self, x, y, z):
        try:
            logger.debug('Point clicked {}'.format((x, y, z)))
            self._controller.select_point(x, y, z)
            self.main_canvas.refresh()
            self.second_canvas.refresh()
            self.update_status('Point {} selected: {}'.format(len(self._controller.points), self._controller.points))
        except (TooManyPointsWarning, InvalidActionWarning) as warn:
            self.update_warning(warn)


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
    img = Image(file_name)
    if os.path.exists(overlay_name):
        overlay = Image(overlay_name)
    controller = PropSegController(img, params, None, 12)
    controller.align_image()
    base_win = PropSeg(controller)
    base_win.show()
    app.exec_()
    print(base_win._controller.as_string())
    base_win._controller.as_niftii()
