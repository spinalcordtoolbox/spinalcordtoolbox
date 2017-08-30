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
from spinalcordtoolbox.gui.base import TooManyPointsWarning, InvalidActionWarning, MissingLabelWarning


logger = logging.getLogger(__name__)


class GroundTruthController(base.BaseController):

    def __init__(self, image, params, init_values=None):
        super(GroundTruthController, self).__init__(image, params, init_values)

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

    def select_slice(self, x, y, z):
        if not self.valid_point(x, y, z):
            raise ValueError('Invalid slice selected {}'.format((x, y, z)))

        _, _, z = self.position
        logger.debug('Slice selected {}'. format((x, y, z)))
        self.position = (x, y, z)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value


class GroundTruth(base.BaseDialog):
    def __init__(self, *args, **kwargs):
        super(GroundTruth, self).__init__(*args, **kwargs)

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()

        self.labels = widgets.VertebraeWidget(self)
        layout.addWidget(self.labels)

        self.sagittal_canvas = widgets.SagittalCanvas(self, plot_points=True, plot_position=True, annotate=True)
        self.sagittal_canvas.point_selected_signal.connect(self.on_select_slice)
        layout.addWidget(self.sagittal_canvas)

        self.main_canvas = widgets.AxialCanvas(self, crosshair=True)
        self.main_canvas.plot_points()
        self.main_canvas.point_selected_signal.connect(self.on_select_point)
        layout.addWidget(self.main_canvas)

        parent.addLayout(layout)

    def _init_controls(self, parent):
        pass

    def on_select_point(self, x, y, z):
        try:
            self._controller.select_point(x, y, z, self.labels.label)
            self.labels.refresh()
            self.main_canvas.refresh()
            self.sagittal_canvas.refresh()
        except (TooManyPointsWarning, InvalidActionWarning, MissingLabelWarning) as warn:
            self.update_warning(warn.message)

    def on_select_slice(self, x, y, z):
        try:
            logger.debug('Select slice {}'.format((x, y, z)))
            self._controller.select_slice(x, y, z)
            self.main_canvas.refresh()
        except (TooManyPointsWarning, InvalidActionWarning) as warn:
            self.update_warning(warn.message)


if __name__ == '__main__':
    import os
    import sys
    from scripts.msct_image import Image

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    try:
        file_name = sys.argv[1]
        overlay_name = sys.argv[2]
    except IndexError:
        file_name = '/Users/geper_admin/sct_example_data/t2/t2.nii.gz'
        overlay_name = '/Users/geper_admin/manual_propseg.nii.gz'

    params = base.AnatomicalParams()
    params.init_message = '1. Select a label -> 2. Select a axial slice -> 3. Select a point in the corrinal plane'
    img = Image(file_name)
    if os.path.exists:
        overlay = Image(overlay_name)
    else:
        overlay = Image(img)
        overlay.file_name = overlay_name
    controller = GroundTruthController(img, params, overlay)
    controller.align_image()
    base.launch_dialog(controller, GroundTruth)
    print(controller.as_string())
    controller.as_niftii(overlay_name)
    sys.exit()
