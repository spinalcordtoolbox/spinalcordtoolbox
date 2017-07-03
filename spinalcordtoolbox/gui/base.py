from __future__ import absolute_import
from __future__ import division

import logging
import webbrowser
from copy import copy

import matplotlib as mpl


mpl.use('Qt4Agg')

import numpy as np
from PyQt4 import QtCore, QtGui

logger = logging.getLogger(__name__)


"""Base classes for creating GUI objects to create manually selected points.
"""


class AnatomicalParams(object):
    """The base parameter object for GUI configuration"""

    def __init__(self, cmap='gray', aspect=1.0, interp='nearest', vmin=5., vmax=95.,
                 vmode='percentile', alpha=1.0):
        """

        Parameters
        ----------
        cmap : str
        aspect : float
        interp : str
        vmin : int
        vmax : int
        vmode : str
        alpha : float
        """
        self.cmap = cmap
        self.aspect = aspect
        self.interp = interp
        self.min = vmin
        self.max = vmax
        self.vmode = vmode
        self.alpha = alpha


class BaseDialog(QtGui.QDialog):
    """Abstract base class to a Anatomical GUI.

    Attributes
    ----------
    update_canvas_signal : QtCore.Signal
        Signal emits when dialog has a point to add to the

    """
    _help_web_address = 'https://sourceforge.net/p/spinalcordtoolbox/wiki/correction_PropSeg/attachment/propseg_viewer.png'

    def __init__(self, controller):
        """Initialize the UI parameters

        Parameters
        ----------
        controller : BaseController
            The logical object that controls the state of the UI
        """
        super(BaseDialog, self).__init__()
        self.params = controller.params
        self._controller = controller
        self.image = controller.image
        self._controller._dialog = self
        self._init_ui()
        self._controller.initialize_dialog()

    def _init_ui(self):
        self.resize(800, 600)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        layout = QtGui.QVBoxLayout(self)

        self._init_header(layout)
        self._init_canvas(layout)
        self._init_controls(layout)
        self._init_footer(layout)

    def _init_canvas(self, parent):
        """

        Parameters
        ----------
        parent : QtGui.QWidget
            The widget / dialog that will host the canvas layout
        """
        raise NotImplementedError('Include _init_canvas in your class declaration')

    def _init_controls(self, parent):
        """

        Parameters
        ----------
        parent : QtGui.QWidget
            The widget / dialog that will host the control layout

        """
        raise NotImplementedError('Include _init_controls in your class declaration')

    def _init_header(self, parent):
        self.lb_status = QtGui.QLabel('Label Status')
        self.lb_status.setStyleSheet("color:black")
        self.lb_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_warning = QtGui.QLabel()
        self.lb_warning.setStyleSheet('color:red')
        self.lb_warning.setAlignment(QtCore.Qt.AlignCenter)

        parent.addWidget(self.lb_status)
        parent.addWidget(self.lb_warning)
        parent.addItem(
            QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum,
                              QtGui.QSizePolicy.Expanding))

    def _init_footer(self, parent):
        """

        Parameters
        ----------
        parent : QtGui.QWidget
            The widget / dialog that will host the footer layout

        Returns
        -------
            The footer layout created
        """
        ctrl_layout = QtGui.QHBoxLayout()

        self.btn_ok = QtGui.QPushButton('Save and Quit')
        self.btn_undo = QtGui.QPushButton('Undo')
        self.btn_help = QtGui.QPushButton('Help')

        ctrl_layout.addItem(
            QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding,
                              QtGui.QSizePolicy.Minimum))
        ctrl_layout.addWidget(self.btn_help)
        ctrl_layout.addWidget(self.btn_undo)
        ctrl_layout.addWidget(self.btn_ok)

        self.btn_help.clicked.connect(self.press_help)

        parent.addLayout(ctrl_layout)
        return ctrl_layout

    def show(self):
        """Override the base class show to fix a bug found in MAC"""
        super(BaseDialog, self).show()
        self.activateWindow()
        self.raise_()

    def update_status(self, msg):
        """

        Parameters
        ----------
        msg : str
            The message to display in the header of dialog
        """
        self.lb_status.setText(msg)
        self.lb_warning.setText('')

    def update_warning(self, msg):
        """

        Parameters
        ----------
        msg : str
            The message to display in the header of dialog
        """
        self.lb_warning.setText(msg)
        self.lb_status.setText('')

    def press_help(self):
        webbrowser.open(self._help_web_address, new=0, autoraise=True)


class BaseController(object):
    points = []
    orientation = None
    _overlay_image = None
    _dialog = None

    def __init__(self, image, params, init_values=None):
        self.image = image
        self.params = params

        if isinstance(init_values, list):
            self.points.extend(init_values)

        elif init_values:
            self.points.append(init_values)

        self.init_points = copy(self.points)

    def align_image(self):
        self.orientation = self.image.orientation
        self.image.change_orientation('SAL')
        x, y, z, t, dx, dy, dz, dt = self.image.dim
        self.params.aspect = dx / dy
        self.params.offset = x * dx
        clip = np.percentile(self.image.data, (self.params.min,
                                               self.params.max))
        self.params.vmin, self.params.vmax = clip
        self.init_x = x // 2
        self.init_y = y // 2
        self.init_z = z // 2

    def _print_point(self, point):
        max_x = self.image.dim[0]
        return '{} {} {}'.format(max_x - point[0], point[1], point[2])

    def valid_point(self, x, y, z):
        dim = self.image.dim
        if 0 <= x < dim[0] and 0 <= y < dim[1] and 0 <= z < dim[2]:
            return True
        return False

    def save_quit(self):
        self._overlay_image = self.image.copy()
        self._overlay_image.data *= 0
        for point in self.points:
            self._overlay_image.data[point[0], point[1], point[2]] = 1

        self._overlay_image.change_orientation(self.orientation)
        self._dialog.close()

    def as_string(self):
        if not self._overlay_image:
            logger.warning('There is no information to save')
            return ''
        output = []
        xs, ys, zs = np.where(self._overlay_image.data)
        for x, y, z in zip(xs, ys, zs):
            output.append('{},{},{},{}'.format(x, y, z, 1))
        return ':'.join(output)

    def as_niftii(self, file_name=None):
        if not self._overlay_image:
            logger.warning('There is no information to save')
            raise IOError('There is no information to save')
        if not file_name:
            file_name = 'manual_propseg.nii.gz'
        print(np.where(self._overlay_image.data))
        self._overlay_image.setFileName(file_name)
        self._overlay_image.save()
