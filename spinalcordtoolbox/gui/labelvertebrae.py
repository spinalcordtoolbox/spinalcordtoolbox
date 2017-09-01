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


class LabelVertebraeController(base.BaseController):
    def __init__(self, image, params, init_values=None):
        super(LabelVertebraeController, self).__init__(image, params, init_values)
        self._label = 0

    def select_point(self, x, y, z, label):
        if not self.valid_point(x, y, z):
            raise ValueError('Invalid point selected {}'.format((x, y, z)))

        logger.debug('Point Selected {}'.format((x, y, z, label)))
        existing_point = [i for i, j in enumerate(self.points) if j[3] == label]

        if existing_point:
            self.points[existing_point[0]] = (x, y, z, label)
        else:
            if self.params.num_points and len(self.points) >= self.params.num_points:
                raise TooManyPointsWarning()
            self.points.append((x, y, z, label))
        self.position = (x, y, z)
        self._label = label

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value


class LabelVertebrae(base.BaseDialog):
    def __init__(self, *args, **kwargs):
        super(LabelVertebrae, self).__init__(*args, **kwargs)

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        parent.addLayout(layout)

        self.labels = widgets.VertebraeWidget(self)
        self.labels.label = self.params.start_label
        layout.addWidget(self.labels)

        self.sag = widgets.SagittalCanvas(self, plot_points=True, annotate=True)
        self.sag.point_selected_signal.connect(self.on_select_point)
        layout.addWidget(self.sag)

    def _init_controls(self, parent):
        main_ctrl = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        main_ctrl.setLayout(layout)
        parent.addWidget(main_ctrl)

    def _init_toolbar(self, parent):
        pass

    def on_select_point(self, x, y, z):
        try:
            label = self.labels.label
            message = 'Label {} selected {}'.format(label, (x, y, z))
            logger.debug(message)
            self._controller.select_point(x, y, z, label)
            self.labels.selected_label(label)
            self.sag.refresh()
            self.update_status(message)
        except (TooManyPointsWarning, MissingLabelWarning) as warn:
            self.update_warning(warn.message)

    def on_undo(self):
        super(LabelVertebrae, self).on_undo()
        self.labels.refresh()
        self.sag.refresh()


if __name__ == '__main__':
    import sys
    import os
    from scripts.msct_image import Image

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    try:
        file_name = sys.argv[1]
        overlay_name = sys.argv[2]
    except Exception:
        file_name = '/Users/geper_admin/sct_example_data/t2/t2.nii.gz'
        overlay_name = '/Users/geper_admin/manual_propseg.nii.gz'

    params = base.AnatomicalParams()
    params.init_message = '1. Select a label -> 2. Select a point in the sagittal plane'
    params.num_points = 1
    params.start_vertebrae = 3
    params.end_vertebrae = 4
    img = Image(file_name)
    if os.path.exists(overlay_name):
        overlay = Image(overlay_name)
    else:
        overlay = Image(file_name)
        overlay.file_name = overlay_name
    controller = LabelVertebraeController(img, params, overlay)
    controller.align_image()
    ctrl = base.launch_dialog(controller, LabelVertebrae)
    print(controller.as_string())
    controller.as_niftii(overlay_name)
    sys.exit()
