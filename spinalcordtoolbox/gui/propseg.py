#!/usr/bin/env python
#  Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT

""" Qt dialog for manually segmenting a spinalcord image """

from __future__ import absolute_import
from __future__ import division

import logging

from PyQt4 import QtCore, QtGui

from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui import widgets
from spinalcordtoolbox.gui.base import TooManyPointsWarning, InvalidActionWarning

logger = logging.getLogger(__name__)


class PropSegController(base.BaseController):
    _mode = ''
    INTERVAL = 15
    MODES = ['AUTO', 'CUSTOM']

    def __init__(self, image, params, init_values=None):
        super(PropSegController, self).__init__(image, params, init_values)
        self._slice = self.INTERVAL

    def align_image(self):
        super(PropSegController, self).align_image()
        if self.image.dim[0] < self.INTERVAL:
            self.INTERVAL = 1
        if not self.params.num_points:
            self.params.num_points = self.image.dim[0] / self.INTERVAL

    def reset_position(self):
        super(PropSegController, self).reset_position()
        if self.mode == 'AUTO':
            self.position = (self._slice, self.position[1], self.position[2])

    def skip_slice(self):
        if self.mode == 'AUTO':
            logger.debug('Advance slice from {} to {}'.format(self._slice,
                                                              self.INTERVAL + self._slice))
            self._slice += self.INTERVAL
            if self._slice >= self.image.dim[0]:
                self._slice -= self.INTERVAL
                raise InvalidActionWarning('The slice can not exceed the height of the anatomy')
            self.position = (self._slice, self.position[1], self.position[2])

    def select_point(self, x, y, z):
        if self.mode == 'CUSTOM' and not self._slice:
            raise InvalidActionWarning('Select a saggital slice before selecting a point.')
        x = self._slice
        if not self.valid_point(x, y, z):
            raise ValueError('Invalid point selected {}'.format((x, y, z)))

        logger.debug('Point Selected {}'.format((x, y, z)))

        existing_points = [i for i, j in enumerate(self.points) if j[0] == x]
        if existing_points:
            self.points[existing_points[0]] = (x, y, z, 1)
        else:
            if len(self.points) >= self.params.num_points:
                raise TooManyPointsWarning()
            self.points.append((x, y, z, 1))
        self.position = (x, y, z)
        self._next_slice()

    def _next_slice(self):
        if self.mode == 'AUTO':
            self._slice += self.INTERVAL
            self.position = (self.position[0] + self.INTERVAL, self.position[1], self.position[2])
        else:
            self._slice = 0

    def select_slice(self, x, y, z):
        if self.mode != 'CUSTOM':
            raise InvalidActionWarning('Can only select a slice in CUSTOM mode')

        if not self.valid_point(x, y, z):
            raise ValueError('Invalid slice selected {}'.format((x, y, z)))

        _, y, z = self.position
        logger.debug('Slice Selected {}'.format((x, y, z)))
        self.position = (x, y, z)
        self._slice = x

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in self.MODES:
            raise ValueError('Invalid mode %', value)

        if value != self._mode:
            self._slice = 0 if value == 'CUSTOM' else self.INTERVAL
            self._mode = value
            self.points = []
            self.reset_position()


class PropSeg(base.BaseDialog):
    def __init__(self, *args, **kwargs):
        super(PropSeg, self).__init__(*args, **kwargs)
        self.axial_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        self.sagittal_canvas = widgets.SagittalCanvas(self, plot_points=True, plot_position=True)
        layout.addWidget(self.sagittal_canvas)

        self.axial_canvas = widgets.AxialCanvas(self, crosshair=True)
        self.axial_canvas.plot_points()
        layout.addWidget(self.axial_canvas)

        self.axial_canvas.point_selected_signal.connect(self.select_point)
        self.sagittal_canvas.point_selected_signal.connect(self.on_select_slice)

        parent.addLayout(layout)

    def _init_controls(self, parent):
        group = QtGui.QGroupBox()
        group.setFlat(True)
        layout = QtGui.QHBoxLayout()

        custom_mode = QtGui.QRadioButton('Mode Custom')
        custom_mode.setToolTip('Manually select the axis slice on sagittal plane')
        custom_mode.toggled.connect(self.on_toggle_mode)
        custom_mode.mode = 'CUSTOM'
        layout.addWidget(custom_mode)

        auto_mode = QtGui.QRadioButton('Mode Auto')
        auto_mode.setToolTip('Automatically move down the axis slice on the sagittal plane')
        auto_mode.toggled.connect(self.on_toggle_mode)
        auto_mode.mode = 'AUTO'
        layout.addWidget(auto_mode)

        group.setLayout(layout)
        parent.addWidget(group)
        auto_mode.click()

    def _init_footer(self, parent):
        ctrl_layout = super(PropSeg, self)._init_footer(parent)
        skip = QtGui.QPushButton('Skip')
        ctrl_layout.insertWidget(2, skip)

        skip.clicked.connect(self.on_skip_slice)

    def on_skip_slice(self):
        try:
            logger.debug('Skipping slice')
            self._controller.skip_slice()
            self.sagittal_canvas.refresh()
        except InvalidActionWarning as warn:
            self.update_warning(warn.message)

    def on_toggle_mode(self):
        widget = self.sender()
        if widget.mode in self._controller.MODES and widget.isChecked() and widget.mode != self._controller.mode:
            self._controller.mode = widget.mode
            self.axial_canvas.refresh()
            self.sagittal_canvas.refresh()
            self.update_status('Reset manual segmentation: Now in mode {}'.format(widget.mode))

    def on_select_slice(self, x, y, z):
        try:
            logger.debug('Slice clicked {}'.format((x, y, z)))
            self._controller.select_slice(x, y, z)
            self.axial_canvas.refresh()
            self.sagittal_canvas.refresh()
            self.update_status('Sagittal slice seleted: {}'.format(self._controller._slice))
        except (TooManyPointsWarning, InvalidActionWarning) as warn:
            self.update_warning(warn.message)

    def select_point(self, x, y, z):
        try:
            logger.debug('Point clicked {}'.format((x, y, z)))
            self._controller.select_point(x, y, z)
            self.axial_canvas.refresh()
            self.sagittal_canvas.refresh()
            self.update_status('Point {} selected: {}'.format(len(self._controller.points), self._controller.points))
        except (TooManyPointsWarning, InvalidActionWarning) as warn:
            self.update_warning(warn.message)

    def on_undo(self):
        super(PropSeg, self).on_undo()
        self.axial_canvas.refresh()
        self.sagittal_canvas.refresh()


if __name__ == '__main__':
    import os
    import sys
    from scripts.msct_image import Image

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    try:
        file_name = sys.argv[1]
        overlay_name = sys.argv[2]
    except Exception:
        file_name = '/Users/geper_admin/sct_example_data/t2/t2.nii.gz'
        overlay_name = '/Users/geper_admin/manual_propseg.nii.gz'

    params = base.AnatomicalParams()
    params.init_message = '1. Select saggital slice -> 2. Select the Axial center of the spinalcord'
    img = Image(file_name)
    if os.path.exists(overlay_name):
        overlay = Image(overlay_name)
    else:
        overlay = Image(img)
        overlay.file_name = overlay_name
    controller = PropSegController(img, params, overlay)
    controller.align_image()
    base.launch_dialog(controller, PropSeg)
    print(controller.as_string())
    controller.as_niftii(overlay_name)
    sys.exit()
