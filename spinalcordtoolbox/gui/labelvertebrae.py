from __future__ import absolute_import

from PyQt4 import QtGui

import base2 as base


class LabelVertebrae(base.BaseDialog):
    LABELS = {'1': 'anterior base of pontomedullary junction (label=50)',
              '2': 'pontomedullary groove (label=49)',

              '3': 'top of C1 vertebrae (label=1)',
              '4': 'posterior edge of C2/C3 intervertebral disk (label=3)',
              '5': 'posterior edge of C3/C4 intervertebral disk (label=4)',
              '6': 'posterior edge of C4/C5 intervertebral disk (label=5)',
              '7': 'posterior edge of C5/C6 intervertebral disk (label=6)',
              '8': 'posterior edge of C6/C7 intervertebral disk (label=7)',
              '9': 'posterior edge of C7/T1 intervertebral disk (label=8)',

              '10': 'posterior edge of T1/T2 intervertebral disk (label=9)',
              '11': 'posterior edge of T2/T3 intervertebral disk (label=10)',
              '12': 'posterior edge of T3/T4 intervertebral disk (label=11)',
              '13': 'posterior edge of T4/T5 intervertebral disk (label=12)',
              '14': 'posterior edge of T5/T6 intervertebral disk (label=13)',
              '15': 'posterior edge of T6/T7 intervertebral disk (label=14)',
              '16': 'posterior edge of T7/T8 intervertebral disk (label=15)',
              '17': 'posterior edge of T8/T9 intervertebral disk (label=16)',
              '18': 'posterior edge of T9/T10 intervertebral disk (label=17)',
              '19': 'posterior edge of T10/T11 intervertebral disk (label=18)',
              '20': 'posterior edge of T11/T12 intervertebral disk (label=19)',
              '21': 'posterior edge of T12/L1 intervertebral disk (label=20)',

              '22': 'posterior edge of L1/L2 intervertebral disk (label=21)',
              '23': 'posterior edge of L2/L3 intervertebral disk (label=22)',
              '24': 'posterior edge of L3/L4 intervertebral disk (label=23)',
              '25': 'posterior edge of L4/S1 intervertebral disk (label=24)',

              '26': 'posterior edge of S1/S2 intervertebral disk (label=25)',
              '27': 'posterior edge of S2/S3 intervertebral disk (label=26)'}

    def _init_canvas(self, parent):
        layout = QtGui.QHBoxLayout()
        sag = base.SagittalCanvas(self)
        cor = base.AxialCanvas(self)
        layout.addWidget(sag)
        layout.addWidget(cor)
        parent.addLayout(layout)

    def _init_controls(self, parent):
        self._label_controler(parent)

    def _init_toolbar(self, parent):
        pass

    def _label_controler(self, parent):
        main_ctrl = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        main_ctrl.setLayout(layout)
        font = QtGui.QFont()
        font.setPointSize(8)

        for key, label in self.LABELS.items():
            rdo = QtGui.QRadioButton(label)
            rdo.setFont(font)
            layout.addWidget(rdo)

        parent.addWidget(main_ctrl)

    def _init_footer(self, parent):
        ctrl_layout = super(LabelVertebrae, self)._init_footer(parent)
        skip = QtGui.QPushButton('Skip')
        ctrl_layout.addWidget(skip, -1)


if __name__ == '__main__':
    import sys
    import logging
    from scripts.msct_image import Image

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    app = QtGui.QApplication(sys.argv)

    params = base.AnatomicalParams()
    img = Image('/Users/geper_admin/sct_example_data/t2/t2.nii.gz')
    img.change_orientation('SAL')
    base_win = LabelVertebrae(params, img)
    base_win.show()
    app.exec_()
