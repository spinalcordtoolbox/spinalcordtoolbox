#  Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT

""" Qt widgets for manually segementing spinalcord images """

import logging

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor

from PyQt4 import QtCore, QtGui

from spinalcordtoolbox.gui.base import MissingLabelWarning

logger = logging.getLogger(__name__)


class VertebraeWidget(QtGui.QWidget):
    """A group of checkboxes that list labels."""
    _unchecked = []
    _checked = []
    _active_label = None
    _check_boxes = {}
    _labels = None
    _label = None

    def __init__(self, parent, vertebraes):
        super(VertebraeWidget, self).__init__(parent)
        self.parent = parent
        self.vertebraes = vertebraes
        self._init_ui(parent.params)
        self.refresh()

    def _init_ui(self, params):
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        font = QtGui.QFont()
        font.setPointSize(10)

        for vertebrae in self.vertebraes:
            rdo = QtGui.QCheckBox('Label {}'.format(vertebrae))
            rdo.label = vertebrae
            rdo.setFont(font)
            rdo.setTristate()
            self._check_boxes[vertebrae] = rdo
            rdo.clicked.connect(self.on_select_label)
            layout.addWidget(rdo)

        layout.addStretch()

    def on_select_label(self):
        label = self.sender()
        self.label = label.label

    def refresh(self, labels=None):
        if labels:
            self._checked = labels
            self._unchecked = set(self._check_boxes.keys()) - set(labels)

        for checkbox in self._check_boxes.itervalues():
            checkbox.setCheckState(QtCore.Qt.Unchecked)

        logger.debug('refresh labels {}'.format(self.parent._controller.points))
        for point in self.parent._controller.points:
            self._check_boxes[point[3]].setCheckState(QtCore.Qt.Checked)

    @property
    def label(self):
        if self._active_label:
            return self._active_label.label
        raise MissingLabelWarning('No vertebrae was selected')

    @label.setter
    def label(self, index):
        self.refresh()
        self._active_label = self._check_boxes[index]
        self._active_label.setCheckState(QtCore.Qt.PartiallyChecked)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, values):
        self._labels = values
        for x in self._check_boxes.values():
            x.setCheckState(QtCore.Qt.Unchecked)

        for label in self._labels:
            self._check_boxes[label].setCheckState(QtCore.Qt.Checked)


class AnatomicalCanvas(FigureCanvas):
    """Base canvas for anatomical views

    Attributes
    ----------
    point_selected_signal : QtCore.Signal
        Create a event when user clicks on the canvas

    """
    point_selected_signal = QtCore.Signal(float, float, float)

    def __init__(self, parent, width=8, height=8, dpi=100, crosshair=False, plot_points=False, annotate=False, plot_position=False):
        self._parent = parent
        self._image = parent.image
        self._params = parent.params
        self._crosshair = crosshair
        self._plot_points = plot_points
        self._annotate_points = annotate
        self._plot_position = plot_position
        self.position = None

        self._x, self._y, self._z = [int(i) for i in self._parent._controller.position]

        self._fig = Figure(figsize=(width, height), dpi=dpi)
        super(AnatomicalCanvas, self).__init__(self._fig)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def _init_ui(self, data, aspect):
        self._fig.canvas.mpl_connect('button_release_event', self.on_update)
        self._fig.canvas.mpl_connect('scroll_event', self.on_zoom)

        self._axes = self._fig.add_axes([0, 0, 1, 1], frameon=True)
        self._axes.axis('off')
        self.view = self._axes.imshow(
            data,
            aspect=aspect,
            cmap=self._params.cmap,
            interpolation=self._params.interp,
            vmin=self._params.vmin,
            vmax=self._params.vmax,
            alpha=self._params.alpha)

        if self._crosshair:
            self.cursor = Cursor(self._axes, useblit=True, color='r', linewidth=1)
        self.points = self._axes.plot([], [], '.r', markersize=7)[0]

    def title(self, message):
        self._fig.suptitle(message)

    def annotate(self, x, y, label):
        self.annotations.append(self._axes.annotate(label, xy=(x, y), xytext=(-3, 3),
                                                    textcoords='offset points', ha='right',
                                                    va='bottom', color='r'))

    def clear(self):
        for i in self.annotations:
            i.remove()
        self.annotations = []
        self.points.set_xdata([])
        self.points.set_ydata([])

    def plot_data(self, xdata, ydata, labels):
        self.points.set_xdata(xdata)
        self.points.set_ydata(ydata)

        if self._annotate_points:
            for x, y, label in zip(xdata, ydata, labels):
                self.annotate(x, y, label)

    def on_zoom(self, event):
        if event.button == 'up':
            scale_factor = 1.3
        else:
            scale_factor = 1 / 1.3

        x = event.xdata
        y = event.ydata

        x_lim = self._axes.get_xlim()
        y_lim = self._axes.get_ylim()

        left = (x - x_lim[0]) * scale_factor
        right = (x_lim[1] - x) * scale_factor
        top = (y - y_lim[0]) * scale_factor
        bottom = (y_lim[1] - y) * scale_factor

        if x + right - left >= self.x_max or y + bottom - top >= self.y_max:
            return

        self._axes.set_xlim(x - left, x + right)
        self._axes.set_ylim(y - top, y + bottom)
        self.view.figure.canvas.draw()

    def __repr__(self):
        return '{}: {}, {}, {}'.format(self.__class__, self._x, self._y, self._z)

    def __str__(self):
        return '{}: {}, {}'.format(self._x, self._y, self._z)


class SagittalCanvas(AnatomicalCanvas):
    def __init__(self, *args, **kwargs):
        super(SagittalCanvas, self).__init__(*args, **kwargs)
        x, y, z, _, dx, dy, dz, _ = self._image.dim
        self._init_ui(self._image.data[:, :, self._z], dx / dy)
        self.annotations = []
        self.x_max = y
        self.y_max = x

    def refresh(self):
        self._x, self._y, self._z = [int(i) for i in self._parent._controller.position]
        data = self._image.data[:, :, self._z]
        self.view.set_array(data)
        self.plot_position()
        self.plot_points()
        self.view.figure.canvas.draw()

    def on_update(self, event):
        if event.xdata > -1 and event.ydata > -1 and event.button == 1:
            self.point_selected_signal.emit(event.ydata, event.xdata, self._z)

    def plot_points(self):
        """Plot the controller's list of points (x, y) and annotate the point with the label"""
        if self._plot_points:
            logger.debug('Plotting points {}'.format(self._parent._controller.points))
            points = self._parent._controller.points
            try:
                xs, ys, zs, labels = zip(*points)
                self.clear()
                self.plot_data(ys, xs, labels)
            except ValueError:
                self.clear()

    def plot_position(self):
        if self._plot_position:
            position = self._parent._controller.position
            if self.position:
                self.position.remove()
            self.position = self._axes.axhline(position[0], color='r')


class CorrinalCanvas(AnatomicalCanvas):
    def __init__(self, parent, width=4, height=8, dpi=100, crosshair=False):
        super(CorrinalCanvas, self).__init__(parent, width, height, dpi, crosshair)
        x, y, z, _, dx, dy, dz, _ = self._image.dim
        self._init_ui(self._image.data[:, self._y, :], dx / dz)
        self.x_max = x
        self.y_max = z

    def refresh(self):
        self._x, self._y, self._z = [int(i) for i in self._parent._controller.position]
        data = self._image.data[:, self._y, :]
        self.view.set_array(data)
        self.view.figure.canvas.draw()

    def on_update(self, event):
        if event.xdata > -1 and event.ydata > -1 and event.button == 1:
            self.point_selected_signal.emit(event.xdata, self._y, event.ydata)

    def plot_points(self):
        logger.debug('Plotting points {}'.format(self._parent._controller.points))
        if self._parent._controller.points:
            points = [x for x in self._parent._controller.points]
            try:
                xs, ys, zs, _ = zip(*points)
                self.clear()
                self.plot_data(xs, zs)
            except ValueError:
                self.clear()
            self.view.figure.canvas.draw()


class AxialCanvas(AnatomicalCanvas):
    def __init__(self, parent, width=4, height=8, dpi=100, crosshair=False):
        super(AxialCanvas, self).__init__(parent, width, height, dpi, crosshair)
        x, y, z, _, dx, dy, dz, _ = self._image.dim
        self._init_ui(self._image.data[self._x, :, :], dy / dz)
        self.x_max = z
        self.y_max = y

    def refresh(self):
        self._x, self._y, self._z = [int(i) for i in self._parent._controller.position]
        data = self._image.data[self._x, :, :]
        self.view.set_array(data)
        self.view.figure.canvas.draw()

    def on_update(self, event):
        if event.xdata > 0 and event.ydata > 0 and event.button == 1:
            self.point_selected_signal.emit(self._x, event.ydata, event.xdata)

    def plot_points(self):
        logger.debug('Plotting points {}'.format(self._parent._controller.points))
        if self._parent._controller.points:
            points = [x for x in self._parent._controller.points]
            try:
                xs, ys, zs, _ = zip(*points)
                self.points.set_xdata(ys)
                self.points.set_ydata(zs)
            except ValueError:
                self.points.set_xdata([])
                self.points.set_ydata([])
            self.view.figure.canvas.draw()


class AnatomicalToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        self.toolitems = (('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
                          ('Back', 'Back to previous view', 'back', 'back'),
                          ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'))
        super(AnatomicalToolbar, self).__init__(canvas, parent)
