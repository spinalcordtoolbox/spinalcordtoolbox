#!/usr/bin/env python
#  Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT

""" Qt dialog for manual labeling of an image """

from __future__ import absolute_import
from __future__ import division

import sct_utils as sct
import logging
from PyQt4 import QtCore, QtGui

from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui import widgets
from spinalcordtoolbox.gui.base import TooManyPointsWarning, InvalidActionWarning

# TODO: remove this useless logger (because no handler is found) by sct.log
logger = logging.getLogger(__name__)


class CenterlineController(base.BaseController):
    _mode = ''
#    INTERVAL = 15
    MODES = ['AUTO', 'CUSTOM']

    def __init__(self, image, params, init_values=None):
        super(CenterlineController, self).__init__(image, params, init_values)

    def reformat_image(self):
        super(CenterlineController, self).reformat_image()
        max_x, max_z = self.image.dim[:3:2]
        self.params.num_points = self.params.num_points or 11

        # # if the starting slice is of invalid value then use default value
        # if self.params.starting_slice > max_x or self.params.starting_slice < 0:
        #     self.params.starting_slice = self.default_position[0]
        #     sct.log.warning('Starting slice value is out of range')
        # # if the starting slice is a fraction, recalculate the starting slice as a ratio.
        # elif 0 < self.params.starting_slice < 1:
        #     self.params.starting_slice = max_z // self.params.starting_slice

        self.reset_position()


    def reset_position(self):
        super(CenterlineController, self).reset_position()
        # set first slice location (see definitions in base.py)
        if self.params.starting_slice == 'fov':
            x_start = 0
        elif self.params.starting_slice == 'midfovminusinterval':
            x_start = round(self.image.dim[0] / 2 - self.params.interval)
        self.position = (self.params.starting_slice, self.position[1], self.position[2])

    def skip_slice(self):
        if self.mode == 'AUTO':
            self.increment_slice()

    def select_point(self, x, y, z):
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
        if self.mode == 'AUTO':
            self.increment_slice()

    def increment_slice(self):
        interval = 1
        if self.mode == 'AUTO':
            interval = self.params.interval

        x, y, z = self.position
        new_x = x + interval
        max_x = self.image.dim[0]

        if new_x >= max_x:
            new_x = max_x - 1

        if self.valid_point(new_x, y, z):
            self.position = (new_x, y, z)

    def decrement_slice(self):
        interval = 1
        if self.mode == 'AUTO':
            interval = self.params.interval

        x, y, z = self.position
        new_x = x - interval

        if new_x < 0:
            new_x = 0

        if self.valid_point(new_x, y, z):
            self.position = (new_x, y, z)

    def select_slice(self, x, y, z):
        if self.mode != 'CUSTOM':
            raise InvalidActionWarning('Can only select a slice in CUSTOM mode')

        if not self.valid_point(x, y, z):
            raise ValueError('Invalid slice selected {}'.format((x, y, z)))

        _, y, z = self.position
        logger.debug('Slice Selected {}'.format((x, y, z)))
        self.position = (x, y, z)

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


class Centerline(base.BaseDialog):
    def __init__(self, *args, **kwargs):
        super(Centerline, self).__init__(*args, **kwargs)
        self.axial_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        parent.addLayout(layout)
        self.sagittal_canvas = widgets.SagittalCanvas(self, plot_points=True, horizontal_nav=True)
        self.sagittal_canvas.title(self.params.subtitle)
        layout.addWidget(self.sagittal_canvas)

        self.axial_canvas = widgets.AxialCanvas(self, plot_points=True, crosshair=True)
        self.axial_canvas.plot_points()
        layout.addWidget(self.axial_canvas)

        self.axial_canvas.point_selected_signal.connect(self.on_select_point)
        self.sagittal_canvas.point_selected_signal.connect(self.on_select_slice)

    def _init_controls(self, parent):
        group = QtGui.QGroupBox()
        group.setFlat(True)
        layout = QtGui.QHBoxLayout()

        custom_mode = QtGui.QRadioButton('Mode Custom')
        custom_mode.setToolTip('Manually select the axis slice on sagittal plane')
        custom_mode.toggled.connect(self.on_toggle_mode)
        custom_mode.mode = 'CUSTOM'
        custom_mode.sagittal_title = 'Select an axial slice.\n{}'.format(self.params.subtitle)
        custom_mode.axial_title = 'Select the center of the spinal cord'
        layout.addWidget(custom_mode)

        auto_mode = QtGui.QRadioButton('Mode Auto')
        auto_mode.setToolTip('Automatically move down the axis slice on the sagittal plane')
        auto_mode.toggled.connect(self.on_toggle_mode)
        auto_mode.mode = 'AUTO'
        auto_mode.sagittal_title = 'The axial slice is automatically selected\n{}'.format(self.params.subtitle)
        auto_mode.axial_title = 'Click in the center of the spinal cord'
        layout.addWidget(auto_mode)

        group.setLayout(layout)
        parent.addWidget(group)
        auto_mode.click()

    def _init_footer(self, parent):
        ctrl_layout = super(Centerline, self)._init_footer(parent)
        skip = QtGui.QPushButton('Skip')
        ctrl_layout.insertWidget(2, skip)

        skip.clicked.connect(self.on_skip_slice)

    def on_skip_slice(self):
        try:
            logger.debug('Skipping slice')
            self._controller.skip_slice()
            self.sagittal_canvas.refresh()
            self.axial_canvas.refresh()
        except InvalidActionWarning as warn:
            self.update_warning(warn.message)

    def on_toggle_mode(self):
        widget = self.sender()
        if widget.mode in self._controller.MODES and widget.isChecked() and widget.mode != self._controller.mode:
            self._controller.mode = widget.mode
            self.update_status('Reset manual labels: Now in mode {}'.format(widget.mode))

            self.sagittal_canvas.title(widget.sagittal_title)
            self.sagittal_canvas.refresh()
            self.axial_canvas.title(widget.axial_title)
            self.axial_canvas.refresh()

    def on_select_slice(self, x, y, z):
        try:
            logger.debug('Slice clicked {}'.format((x, y, z)))
            self._controller.select_slice(x, y, z)
            self.axial_canvas.refresh()
            self.sagittal_canvas.refresh()
            self.update_status('Sagittal position at {:8.2f}'.format(self._controller.position[0]))
        except (TooManyPointsWarning, InvalidActionWarning) as warn:
            self.update_warning(warn.message)

    def increment_horizontal_nav(self):
        self._controller.increment_slice()
        self.axial_canvas.refresh()
        self.sagittal_canvas.refresh()

    def decrement_horizontal_nav(self):
        self._controller.decrement_slice()
        self.axial_canvas.refresh()
        self.sagittal_canvas.refresh()

    def on_select_point(self, x, y, z):
        try:
            logger.debug('Point clicked {}'.format((x, y, z)))
            self._controller.select_point(x, y, z)
            self.axial_canvas.refresh()
            self.sagittal_canvas.refresh()
            self.update_status('{} point(s) selected'.format(len(self._controller.points)))
        except (TooManyPointsWarning, InvalidActionWarning) as warn:
            self.update_warning(warn.message)

    def on_undo(self):
        super(Centerline, self).on_undo()
        self.axial_canvas.refresh()
        self.sagittal_canvas.refresh()


def launch_centerline_dialog(im_input, im_output, params):
    params.input_file_name = im_input.absolutepath
    params.subtitle = u'Use the Up/Down arrows to navigate the superior-inferior direction'
    controller = CenterlineController(im_input, params, im_output)
    controller.reformat_image()

    app = QtGui.QApplication([])
    dialog_ = Centerline(controller)
    dialog_.show()
    app.exec_()
    return controller
