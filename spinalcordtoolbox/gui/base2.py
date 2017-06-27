from __future__ import division
from __future__ import absolute_import

import webbrowser
from copy import copy
import logging

import matplotlib as mpl

mpl.use('Qt4Agg')

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
import numpy as np
from PyQt4 import QtCore, QtGui

logger = logging.getLogger(__name__)


"""Base classes for creating GUI objects to create manually selected points.

    Example
    -------
    >>> params = AnatomicalParams()
    >>> dialog = BaseDialog(img_obj)
    >>> dialog.show()

"""


class AnatomicalParams(object):
    """The base parameter object for GUI configuration
    """
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


class AnatomicalCanvas(FigureCanvas):
    """Base canvas for anatomical views

    Attributes
    ----------
    point_selected_signal : QtCore.Signal
        Create a event when user clicks on the canvas

    """
    point_selected_signal = QtCore.Signal(int, int, int)

    def __init__(self, parent, width=8, height=8, dpi=100, interactive=False):
        self._parent = parent
        self.image = parent.image
        self.params = parent.params
        self.interactive = interactive

        self._x = self._parent._controller.init_x
        self._y = self._parent._controller.init_y
        self._z = self._parent._controller.init_z

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(AnatomicalCanvas, self).__init__(self.fig)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def _init_ui(self, data):
        self.axes = self.fig.add_subplot(111)
        self.axes.axis('off')
        self.axes.set_frame_on(True)
        self.fig.canvas.mpl_connect('button_release_event', self.on_update)
        self.view = self.axes.imshow(
            data,
            aspect=self.params.aspect,
            cmap=self.params.cmap,
            interpolation=self.params.interp,
            vmin=self.params.vmin,
            vmax=self.params.vmax,
            alpha=self.params.alpha)

        if self.interactive:
            self.cursor = Cursor(self.axes, useblit=True, color='red', linewidth=1)

    def __repr__(self):
        return '{}: {}, {}, {}'.format(self.__class__, self._x, self._y, self._z)

    @QtCore.Slot(int, int, int)
    def on_refresh_slice(self, x, y, z):
        logger.debug('Current slice {}'.format((x, y, z)))
        self._x, self._y, self._z = x, y, z
        self.refresh_slice()
        self.view.figure.canvas.draw()


class SagittalCanvas(AnatomicalCanvas):
    def __init__(self, *args, **kwargs):
        super(SagittalCanvas, self).__init__(*args, **kwargs)
        self._init_ui(self.image.data[:, :, self._z])
        self._slices = {}

    def refresh_slice(self):
        logging.debug(self._z)
        data = self.image.data[:, :, self._z]
        self.view.set_array(data)

        if self._hslices:
            self.plot_hslices()

    def on_update(self, event):
        if event.xdata > 0 and event.ydata > 0 and event.button == 1:
            self.point_selected_signal.emit(int(event.ydata), int(event.xdata), self._z)

    def plot_points(self):
        if self._parent._controller.points:
            points = [x for x in self._parent._controller.points if self._z == x[2]]
            cols = zip(*points)
            if not self.points:
                self.points, = self.axes.plot(cols[0], cols[1], '.r', markersize=10)
            else:
                self.points.set_xdata(cols[0])
                self.points.set_ydata(cols[1])
                self.view.figure.canvas.draw()

    def plot_hslices(self):
        self._hslices = True
        slices = [x[0] for x in self._parent._controller.points]
        for x in slices:
            if x not in self._slices:
                self._slices[x] = self.axes.axhline(x, color='w')


class CorrinalCanvas(AnatomicalCanvas):
    def __init__(self, *args, **kwargs):
        super(CorrinalCanvas, self).__init__(*args, **kwargs)
        self._init_ui(self.image.data[:, self._y, :])

    def refresh_slice(self):
        data = self.image.data[:, self._y, :]
        self.view.set_array(data)

    def on_update(self, event):
        if event.xdata > 0 and event.ydata > 0 and event.button == 1:
            self.point_selected_signal.emit(event.xdata, self._y, event.ydata)


class AxialCanvas(AnatomicalCanvas):
    def __init__(self, *args, **kwargs):
        super(AxialCanvas, self).__init__(*args, **kwargs)
        self._init_ui(self.image.data[self._x, :, :])

    def refresh_slice(self):
        data = self.image.data[self._x, :, :]
        self.view.set_array(data)

    def on_update(self, event):
        if event.xdata > 0 and event.ydata > 0 and event.button == 1:
            self.point_selected_signal.emit(self._x, event.ydata, event.xdata)

    def plot_points(self, points):
        if points:
            points = [x for x in points if self._x == x[0]]
            cols = zip(*points)
            if cols:
                self.points = self.axes.plot(cols[1], cols[2], '.r', markersize=10)


class AnatomicalToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        self.toolitems = (('Pan', 'Pan axes with left mouse, zoom with right',
                           'move', 'pan'), ('Back', 'Back to previous view',
                                            'back', 'back'),
                          ('Zoom', 'Zoom to rectangle', 'zoom_to_rect',
                           'zoom'))
        super(AnatomicalToolbar, self).__init__(canvas, parent)


class BaseDialog(QtGui.QDialog):
    update_canvas_signal = QtCore.Signal(int, int, int)
    _hovering_point = (0, 0, 0)
    _selected_points = []
    _help_web_address = 'https://sourceforge.net/p/spinalcordtoolbox/wiki/correction_PropSeg/attachment/propseg_viewer.png'

    def __init__(self, controller):
        """

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
        self.resize(800, 600)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        layout = QtGui.QVBoxLayout(self)

        self._init_header(layout)
        self._init_canvas(layout)
        self._init_controls(layout)
        self._init_footer(layout)

    def _init_canvas(self, parent):
        raise NotImplementedError('Include _init_canvas in your class declaration')

    def _init_controls(self, parent):
        raise NotImplementedError('Include _init_canvas in your class declaration')

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

    @QtCore.Slot(str)
    def update_status(self, msg):
        self.lb_status.setText(msg)
        self.lb_warning.setText('')

    @QtCore.Slot(str)
    def update_warning(self, msg):
        self.lb_warning.setText(msg)
        self.lb_status.setText('')

    @QtCore.Slot(int, int, int)
    def update_points(self, x, y, z):
        self.update_status('{}, {}, {}'.format(x, y, z))
        self._hovering_point = (x, y, z)
        self._selected_points.append((x, y, z))
        self.update_canvas_signal.emit(x, y, z)

    def press_help(self):
        webbrowser.open(self._help_web_address, new=0, autoraise=True)


class BaseController(object):
    _points = []
    points = []

    def __init__(self, image, params, init_values=None):
        self.image = image
        self.params = params

        if isinstance(init_values, list):
            self.points.extend(init_values)

        elif init_values:
            self.points.append(init_values)

        self.init_points = copy(self.points)

    def align_image(self):
        x, y, z, t, dx, dy, dz, dt = self.image.dim
        self.params.aspect = dx / dy
        self.params.offset = x * dx
        clip = np.percentile(self.image.data, (self.params.min,
                                               self.params.max))
        self.params.vmin, self.params.vmax = clip
        shape = self.image.data.shape
        dimension = [shape[0] * dx, shape[1] * dy, shape[2] * dz]
        max_size = max(dimension)
        self.x_offset = int(round(max_size - dimension[0]) / dx / 2)
        self.y_offset = int(round(max_size - dimension[1]) / dy / 2)
        self.z_offset = int(round(max_size - dimension[2]) / dz / 2)
