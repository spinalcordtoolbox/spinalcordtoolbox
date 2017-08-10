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

    def initialize_dialog(self):
        self._dialog.update_status('1. Select a label -> 2. Select a axial slice -> 3. Select a point in the corrinal plane')

    def select_point(self, x, y, z, label):
        if not self.valid_point(x, y, z):
            raise ValueError('Invalid point selected {}'.format((x, y, z)))

        logger.debug('Point Selected {}'.format((x, y, z, label)))
        self._label = label

        self.points.append((x, y, z, self._label))
        self.position = (x, y, z)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value


class LabelVertebrae(base.BaseDialog):
    def __init__(self, *args, **kwargs):
        super(LabelVertebrae, self).__init__(*args, **kwargs)
        # self.labels = controller.labels

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        self.sag = widgets.SagittalCanvas(self, interactive=True)
        self.labels = widgets.VertebraeWidget(self)
        layout.addWidget(self.labels)
        layout.addWidget(self.sag)
        parent.addLayout(layout)

        self.sag.point_selected_signal.connect(self.select_point)

    def _init_controls(self, parent):
        self._label_controler(parent)

    def _init_toolbar(self, parent):
        pass

    def _label_controler(self, parent):
        main_ctrl = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        main_ctrl.setLayout(layout)

        parent.addWidget(main_ctrl)

    def _init_footer(self, parent):
        ctrl_layout = super(LabelVertebrae, self)._init_footer(parent)
        skip = QtGui.QPushButton('Skip')
        ctrl_layout.addWidget(skip, -1)

    def select_point(self, x, y, z):
        try:
            label = self.labels.label
            logger.debug('Point clicked {}'.format((x, y, z)))
            self._controller.select_point(x, y, z, label)
            self.labels.selected_label(label)
            message = 'Label {} selected {}'.format(label, (x, y, z))
            self.update_status(message)
        except (TooManyPointsWarning, MissingLabelWarning) as warn:
            self.update_warning(warn.message)


if __name__ == '__main__':
    import sys
    import os
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
    controller = LabelVertebraeController(img, params)
    controller.align_image()
    base_win = LabelVertebrae(controller)
    base_win.show()
    app.exec_()
    print(controller.as_string())
    controller.as_niftii()
