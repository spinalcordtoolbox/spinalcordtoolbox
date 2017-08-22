#!/usr/bin/env python
#  Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT

""" Qt dialog for manually segmenting a spinalcord image """


from __future__ import absolute_import
from __future__ import division

import logging

from PyQt4 import QtGui, QtCore

from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui import widgets
from spinalcordtoolbox.gui.base import TooManyPointsWarning, InvalidActionWarning, MissingLabelWarning


logger = logging.getLogger(__name__)


class GroundTruthController(base.BaseController):

    def __init__(self, image, params, init_values=None):
        super(GroundTruthController, self).__init__(image, params, init_values)

    def select_point(self, x, y, z, label):
        logger.debug('Point Selected {}'.format((x, y, z, label)))
        if self.valid_point(x, y, z) and label:
            self.points.append((x, y, z, label))

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
        self.corrinal_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        self.sagittal_canvas = widgets.SagittalCanvas(self)

        self.labels = widgets.VertebraeWidget(self)

        self.corrinal_canvas = widgets.CorrinalCanvas(self)
        self.corrinal_canvas.plot_points()

        layout.addWidget(self.labels)
        layout.addWidget(self.corrinal_canvas)
        layout.addWidget(self.sagittal_canvas)

        self.corrinal_canvas.point_selected_signal.connect(self.on_select_point)
        self.sagittal_canvas.point_selected_signal.connect(self.on_select_slice)
        parent.addLayout(layout)

    def _init_toolbar(self, parent):
        pass

    def _init_controls(self, parent):
        pass

    def on_select_point(self, x, y, z):
        try:
            self._controller.select_point(x, y, z, self.labels.label)
            self.labels.refresh()
            self.corrinal_canvas.refresh()
            self.sagittal_canvas.refresh()
        except (TooManyPointsWarning, InvalidActionWarning, MissingLabelWarning) as warn:
            self.update_warning(warn.message)

    def on_select_slice(self, x, y, z):
        try:
            logger.debug('Select slice {}'.format((x, y, z)))
            self._controller.select_slice(x, y, z)
            self.corrinal_canvas.refresh()
        except (TooManyPointsWarning, InvalidActionWarning) as warn:
            self.update_warning(warn.message)


if __name__ == '__main__':
    import os
    import sys
    from scripts.msct_image import Image

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)

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
    base_win = GroundTruth(controller)
    base_win.show()
    app.exec_()
    print(base_win._controller.as_string())
    base_win._controller.as_niftii(overlay_name)
