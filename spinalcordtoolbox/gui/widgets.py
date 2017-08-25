#  Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
#
# About the license: see the file LICENSE.TXT

""" Qt widgets for manually segementing spinalcord images """

import logging
from itertools import dropwhile, takewhile

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor

from PyQt4 import QtCore, QtGui

from spinalcordtoolbox.gui.base import MissingLabelWarning

logger = logging.getLogger(__name__)


class VertebraeWidget(QtGui.QWidget):
    """A group of checkboxes that list labels."""
    LABELS = ((50, 'anterior base of pontomedullary junction (label=50)'),
              (49, 'pontomedullary groove (label=49)'),
              (1, 'top of C1 vertebrae (label=1)'),
              (3, 'posterior edge of C2/C3 intervertebral disk (label=3)'),
              (4, 'posterior edge of C3/C4 intervertebral disk (label=4)'),
              (5, 'posterior edge of C4/C5 intervertebral disk (label=5)'),
              (6, 'posterior edge of C5/C6 intervertebral disk (label=6)'),
              (7, 'posterior edge of C6/C7 intervertebral disk (label=7)'),
              (8, 'posterior edge of C7/T1 intervertebral disk (label=8)'),
              (9, 'posterior edge of T1/T2 intervertebral disk (label=9)'),
              (10, 'posterior edge of T2/T3 intervertebral disk (label=10)'),
              (11, 'posterior edge of T3/T4 intervertebral disk (label=11)'),
              (12, 'posterior edge of T4/T5 intervertebral disk (label=12)'),
              (13, 'posterior edge of T5/T6 intervertebral disk (label=13)'),
              (14, 'posterior edge of T6/T7 intervertebral disk (label=14)'),
              (15, 'posterior edge of T7/T8 intervertebral disk (label=15)'),
              (16, 'posterior edge of T8/T9 intervertebral disk (label=16)'),
              (17, 'posterior edge of T9/T10 intervertebral disk (label=17)'),
              (18, 'posterior edge of T10/T11 intervertebral disk (label=18)'),
              (19, 'posterior edge of T11/T12 intervertebral disk (label=19)'),
              (20, 'posterior edge of T12/L1 intervertebral disk (label=20)'),
              (21, 'posterior edge of L1/L2 intervertebral disk (label=21)'),
              (22, 'posterior edge of L2/L3 intervertebral disk (label=22)'),
              (23, 'posterior edge of L3/L4 intervertebral disk (label=23)'),
              (24, 'posterior edge of L4/S1 intervertebral disk (label=24)'),
              (25, 'posterior edge of S1/S2 intervertebral disk (label=25)'),
              (26, 'posterior edge of S2/S3 intervertebral disk (label=26)'))
    _active_label = None
    _check_boxes = {}

    def __init__(self, parent):
        super(VertebraeWidget, self).__init__(parent)
        self.parent = parent
        self._init_ui(parent.params)
        self.refresh()

    def _init_ui(self, params):
        labels = dropwhile(lambda x: x[0] != params.start_label, self.LABELS)
        labels = takewhile(lambda x: x[0] != params.end_label, labels)

        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        font = QtGui.QFont()
        font.setPointSize(8)

        for label, title in labels:
            rdo = QtGui.QCheckBox(title)
            rdo.label = label
            rdo.setFont(font)
            rdo.setTristate()
            self._check_boxes[label] = rdo
            rdo.clicked.connect(self.on_select_label)
            layout.addWidget(rdo)

    def on_select_label(self):
        label = self.sender()
        if self._active_label and self._active_label.checkState() == QtCore.Qt.PartiallyChecked:
            self._active_label.setCheckState(QtCore.Qt.Unchecked)
        self._active_label = label

    def refresh(self):
        for x in self._check_boxes.values():
            x.setCheckState(QtCore.Qt.Unchecked)

        logger.debug('refresh labels {}'.format(self.parent._controller.points))
        for point in self.parent._controller.points:
            self._check_boxes[point[3]].setCheckState(QtCore.Qt.Checked)

    def selected_label(self, index):
        label = self._check_boxes[index]
        if self._active_label == label:
            self._active_label = None
        self._check_boxes[index].setCheckState(QtCore.Qt.Checked)

    @property
    def label(self):
        if self._active_label:
            return self._active_label.label
        raise MissingLabelWarning('No vertebrae was selected')


class AnatomicalCanvas(FigureCanvas):
    """Base canvas for anatomical views

    Attributes
    ----------
    point_selected_signal : QtCore.Signal
        Create a event when user clicks on the canvas

    """
    point_selected_signal = QtCore.Signal(int, int, int)

    def __init__(self, parent, width=8, height=8, dpi=100, crosshair=False, plot_points=False, annotate=False, plot_position=False):
        self._parent = parent
        self._image = parent.image
        self._params = parent.params
        self._crosshair = crosshair
        self._plot_points = plot_points
        self._annotate_points = annotate
        self._plot_position = plot_position

        self._x, self._y, self._z = self._parent._controller.position

        self._fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super(AnatomicalCanvas, self).__init__(self._fig)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def _init_ui(self, data):
        self._axes = self._fig.add_axes([0, 0, 1, 1])
        self._axes.axis('off')
        self._axes.set_frame_on(True)
        self._fig.canvas.mpl_connect('button_release_event', self.on_update)
        self.view = self._axes.imshow(
            data,
            aspect=self._params.aspect,
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
        self.annotations.append(self._axes.annotate(label, (x + 3, y + 3), color='r'))

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

    def __repr__(self):
        return '{}: {}, {}, {}'.format(self.__class__, self._x, self._y, self._z)

    def __str__(self):
        return '{}: {}, {}'.format(self._x, self._y, self._z)


class SagittalCanvas(AnatomicalCanvas):
    def __init__(self, *args, **kwargs):
        super(SagittalCanvas, self).__init__(*args, **kwargs)
        self._init_ui(self._image.data[:, :, self._z])
        self.position = None
        self.annotations = []

    def refresh(self):
        self._x, self._y, self._z = self._parent._controller.position
        data = self._image.data[:, :, self._z]
        self.view.set_array(data)
        self.plot_position()
        self.plot_points()
        self.view.figure.canvas.draw()

    def on_update(self, event):
        if event.xdata > 0 and event.ydata > 0 and event.button == 1:
            self.point_selected_signal.emit(int(event.ydata), int(event.xdata), self._z)

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
        self._init_ui(self._image.data[:, self._y, :])

    def refresh(self):
        self._x, self._y, self._z = self._parent._controller.position
        data = self._image.data[:, self._y, :]
        self.view.set_array(data)
        self.view.figure.canvas.draw()

    def on_update(self, event):
        if event.xdata > 0 and event.ydata > 0 and event.button == 1:
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
        self._init_ui(self._image.data[self._x, :, :])

    def refresh(self):
        self._x, self._y, self._z = self._parent._controller.position
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
