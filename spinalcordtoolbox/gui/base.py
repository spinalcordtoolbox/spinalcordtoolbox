"""Base classes for creating GUI objects to create manually selected points."""

from __future__ import absolute_import
from __future__ import division

import logging
import webbrowser

import matplotlib as mpl
import numpy as np

mpl.use('Qt4Agg')

from PyQt4 import QtCore, QtGui

logger = logging.getLogger(__name__)


class AnatomicalParams(object):
    """The base parameter object for GUI configuration"""
    CODES = [50, 49, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

    def __init__(self,
                 cmap='gray',
                 aspect=1.0,
                 interp='nearest',
                 vmin=5.,
                 vmax=95.,
                 vmode='percentile',
                 alpha=1.0):
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
        self.start_vertebrae = 50
        self.end_vertebrae = -1
        self.num_points = 0
        self._title = ''
        self._vertebraes = []

    @property
    def dialog_title(self):
        if not self._title:
            self._title = '{}: maunally label'.format(self.input_file_name)
        return self._title

    @property
    def vertebraes(self):
        return self._vertebraes

    @vertebraes.setter
    def vertebraes(self, values):
        if not values:
            return

        if len(set(values) - set(self.CODES)) != 0:
            raise ValueError('The values are not all valid vertebraes labels')

        self._vertebraes = values
        self.start_vertebrae = values[0]
        self.end_vertebrae = values[-1]


class BaseDialog(QtGui.QDialog):
    """Abstract base class to a Anatomical GUI.

    Attributes
    ----------
    update_canvas_signal : QtCore.Signal
        Signal emits when dialog has a point to add to the

    """
    _help_web_address = 'https://sourceforge.net/p/spinalcordtoolbox/wiki/correction_PropSeg/attachment/propseg_viewer.png'
    lb_status = None
    lb_warning = None
    btn_ok = None
    btn_undo = None
    btn_help = None

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

    def _init_ui(self):
        self.resize(1200, 800)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        layout = QtGui.QVBoxLayout(self)

        self._init_header(layout)
        self._init_canvas(layout)
        self._init_controls(layout)
        self._init_footer(layout)

        self.setWindowTitle(self.params.dialog_title)

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
        parent.addStretch()
        message = getattr(self.params, 'init_message', '')
        self.update_status(message)

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

        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_help)
        ctrl_layout.addWidget(self.btn_undo)
        ctrl_layout.addWidget(self.btn_ok)

        self.btn_help.clicked.connect(self.on_help)
        self.btn_undo.clicked.connect(self.on_undo)
        self.btn_ok.clicked.connect(self.on_save_quit)

        parent.addLayout(ctrl_layout)
        return ctrl_layout

    def on_save_quit(self):
        self._controller.save()
        self.close()

    def on_undo(self):
        try:
            self._controller.undo()
        except InvalidActionWarning as err:
            self.update_warning(err.message)

    def on_help(self):
        webbrowser.open(self._help_web_address, new=0, autoraise=True)

    def show(self):
        """Override the base class show to fix a bug found in MAC"""
        super(BaseDialog, self).show()
        self.activateWindow()
        self.raise_()

    def update_status(self, msg):
        """Print the message into the dialog's status widget and clear the warning widget

        Parameters
        ----------
        msg : str  The message to display in the header of dialog
        """
        self.lb_status.setText(msg)
        self.lb_warning.setText('')

    def update_warning(self, msg):
        """Print the message into the dialog's warning widget and clear the status widget

        Parameters
        ----------
        msg : str  The message to display in the header of dialog
        """
        self.lb_warning.setText(msg)
        self.lb_status.setText('')


class BaseController(object):
    orientation = None
    _overlay_image = None
    _dialog = None
    position = None
    saved = False

    def __init__(self, image, params, init_values=None):
        self.image = image
        self.params = params
        self.points = []

        if init_values:
            self._overlay_image = init_values

    def reformat_image(self):
        """Set the camera position and increase contrast.

        The image orientation is set to SAL. And set the default contrast, and
        axes position for all canvases. Need to run before displaying the GUI
        with the image.

        """
        logger.debug('Image orientation {}'.format(self.image.orientation))
        self.orientation = self.image.orientation
        self.image.change_orientation('SAL')

        if self._overlay_image:
            self._overlay_image.change_orientation('SAL')

        x, y, z, t, dx, dy, dz, dt = self.image.dim
        self.params.aspect = dx / dy
        self.params.offset = x * dx

        clip = np.percentile(self.image.data, (self.params.min,
                                               self.params.max))
        self.params.vmin, self.params.vmax = clip
        self.reset_position()

    def reset_position(self):
        """Set the canvas position to the center of the image"""
        x, y, z, _, _, _, _, _ = self.image.dim
        self.position = (x // 2, y // 2, z // 2)

    def valid_point(self, x, y, z):
        dim = self.image.dim
        if 0 <= x < dim[0] and 0 <= y < dim[1] and 0 <= z < dim[2]:
            return True
        return False

    def save(self):
        logger.debug('Overlay shape {}'.format(self._overlay_image.data.shape))

        for point in self.points:
            x, y, z, label = point
            self._overlay_image.data[x, y, z] = label

        if self.orientation != self._overlay_image.orientation:
            self._overlay_image.change_orientation(self.orientation)

        self.saved = True

    def undo(self):
        """Remove the last point selected and refresh the UI"""
        if self.points:
            x, y, z, _ = self.points[-1]
            self.position = (x, y, z)
            self.points = self.points[:-1]
            self._slice = self.position[0]
            logger.debug('Point removed {}'.format(self.position))
        else:
            raise InvalidActionWarning('There is no points selected to undo')

    def as_string(self):
        if self._overlay_image is None:
            logger.warning('There is no information to save')
            return ''
        output = []
        data = self._overlay_image.data
        xs, ys, zs = np.where(data)
        for x, y, z in zip(xs, ys, zs):
            output.append('{},{},{},{}'.format(x, y, z, data[x, y, z]))
        return ':'.join(output)

    def as_niftii(self, file_name=None):
        if not self._overlay_image:
            logger.warning('There is no information to save')
            raise IOError('There is no information to save')
        if file_name:
            self._overlay_image.setFileName(file_name)

        if self._overlay_image.absolutepath == self.image.absolutepath:
            raise IOError('Aborting: the original file and the labeled file are the same', self._overlay_image.absolutepath)

        logger.debug('Data: {}'.format(np.where(self._overlay_image.data)))
        self._overlay_image.save()


class TooManyPointsWarning(StopIteration):
    message = 'Reached the maximum superior / inferior axis length'


class InvalidActionWarning(ValueError):
    pass


class MissingLabelWarning(ValueError):
    pass


def launch_dialog(controller, dialog_class):
    app = QtGui.QApplication([])
    dialog = dialog_class(controller)
    dialog.show()
    app.exec_()
    return controller
