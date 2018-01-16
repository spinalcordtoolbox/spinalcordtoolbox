#!/usr/bin/env python
#  Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT

""" Qt dialog for manually segmenting a spinalcord image """

from __future__ import absolute_import
from __future__ import division

import logging

from PyQt4 import QtGui

from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui import widgets
from spinalcordtoolbox.gui.base import TooManyPointsWarning, MissingLabelWarning

logger = logging.getLogger(__name__)


class SagittalController(base.BaseController):
    def __init__(self, image, params, init_values=None):
        super(SagittalController, self).__init__(image, params, init_values)

    def select_point(self, x, y, z, label):
        if not self.valid_point(x, y, z):
            raise ValueError('Invalid coordinates {}'.format((x, y, z)))

        existing_point = [i for i, j in enumerate(self.points) if j[3] == label]

        if existing_point:
            self.points[existing_point[0]] = (x, y, z, label)
        else:
            if self.params.num_points and len(self.points) >= self.params.num_points:
                raise TooManyPointsWarning()
            self.points.append((x, y, z, label))

        self.position = (x, y, z)


class SagittalDialog(base.BaseDialog):
    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        parent.addLayout(layout)

        self.labels = widgets.VertebraeWidget(self, self.params.vertebraes)
        self.labels.label = self.params.start_vertebrae
        layout.addWidget(self.labels)

        self.sagittal = widgets.SagittalCanvas(self, plot_points=True, annotate=True)
        self.sagittal.title(self.params.subtitle)
        self.sagittal.point_selected_signal.connect(self.on_select_point)
        layout.addWidget(self.sagittal)

    def _init_controls(self, parent):
        pass

    def on_select_point(self, x, y, z):
        try:
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            label = self.labels.label
            self._controller.select_point(x, y, z, label)
            self.labels.refresh()
            self.sagittal.refresh()

            index = self.params.vertebraes.index(label)
            if index + 1 < len(self.params.vertebraes):
                self.labels.label = self.params.vertebraes[index + 1]
        except (TooManyPointsWarning, MissingLabelWarning) as warn:
            self.update_warning(warn.message)

    def on_undo(self):
        super(SagittalDialog, self).on_undo()
        self.sagittal.refresh()
        self.labels.refresh()
        self.labels.label = self._controller.label


def launch_sagittal_dialog(input_file, output_file, params):
    if not params.vertebraes:
        params.vertebraes = [3, 5]
    params.input_file_name = input_file.absolutepath
    controller = SagittalController(input_file, params, output_file)
    controller.reformat_image()

    app = QtGui.QApplication([])
    dialog = SagittalDialog(controller)
    dialog.show()
    app.exec_()
    return controller
