import logging

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor

from PyQt4 import QtCore, QtGui


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
    _selected_label = None
    _check_boxes = {}

    def __init__(self, parent):
        super(VertebraeWidget, self).__init__(parent)
        self.parent = parent
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        font = QtGui.QFont()
        font.setPointSize(8)

        for label, title in self.LABELS:
            rdo = QtGui.QCheckBox(title)
            rdo.label = label
            rdo.setFont(font)
            rdo.setTristate()
            self._check_boxes[label] = rdo
            rdo.clicked.connect(self.on_select_label)
            layout.addWidget(rdo)

    def _init_data(self, labels):
        for label in labels:
            self._selected_label[label].setCheckState(QtCore.Qt.Checked)

    def on_select_label(self):
        label = self.sender()
        if self._selected_label and self._selected_label.checkState() == QtCore.Qt.PartiallyChecked:
            self._selected_label.setCheckState(QtCore.Qt.Unchecked)
        self._selected_label = label

    def on_refresh(self):
        for point in self.parent._controller.points:
            self._check_boxes[point[3]].setCheckState(QtCore.Qt.Checked)


class AnatomicalCanvas(FigureCanvas):
    """Base canvas for anatomical views

    Attributes
    ----------
    point_selected_signal : QtCore.Signal
        Create a event when user clicks on the canvas

    """
    point_selected_signal = QtCore.Signal(int, int, int)
    _hslices = False
    _slices = []

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
        self.axes = self.fig.add_axes([0, 0, 1, 1])
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

    def on_refresh_slice(self, x, y, z):
        logger.debug('Current slice {}'.format((x, y, z)))
        self._x, self._y, self._z = x, y, z
        self.refresh_slice()
        self.view.figure.canvas.draw()

    def __repr__(self):
        return '{}: {}, {}, {}'.format(self.__class__, self._x, self._y, self._z)

    def __str__(self):
        return '{}: {}, {}'.format(self._x, self._y, self._z)


class SagittalCanvas(AnatomicalCanvas):
    def __init__(self, *args, **kwargs):
        super(SagittalCanvas, self).__init__(*args, **kwargs)
        self._init_ui(self.image.data[:, :, self._z])

    def refresh_slice(self):
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
        for s in self._slices:
            s.remove()

        self._slices = []

        for x in slices:
            self._slices.append(self.axes.axhline(x, color='w'))


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

    def plot_points(self):
        if self._parent._controller.points:
            points = [x for x in self._parent._controller.points if self._y == x[0]]
            cols = zip(*points)
            if not self.points:
                self.points = self.axes.plot(cols[1], cols[2], '.r', markersize=10)
            else:
                self.points.set_xdata(cols[0])
                self.points.set_ydata(cols[1])
                self.view.figure.canvas.draw()


class AxialCanvas(AnatomicalCanvas):
    def __init__(self, parent, width=4, height=8, dpi=100, interactive=False):
        super(AxialCanvas, self).__init__(parent, width, height, dpi, interactive)
        self._init_ui(self.image.data[self._x, :, :])

    def refresh_slice(self):
        data = self.image.data[self._x, :, :]
        self.view.set_array(data)

    def on_update(self, event):
        if event.xdata > 0 and event.ydata > 0 and event.button == 1:
            self.point_selected_signal.emit(self._x, event.ydata, event.xdata)

    def plot_points(self):
        if self._parent._controller.points:
            points = [x for x in self._parent._controller.points if self._x == x[0]]
            cols = zip(*points)
            if not self.points:
                self.points = self.axes.plot(cols[1], cols[2], '.r', markersize=10)
            else:
                self.points.set_xdata(cols[0])
                self.points.set_ydata(cols[1])
                self.view.figure.canvas.draw()


class AnatomicalToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        self.toolitems = (('Pan', 'Pan axes with left mouse, zoom with right',
                           'move', 'pan'), ('Back', 'Back to previous view',
                                            'back', 'back'),
                          ('Zoom', 'Zoom to rectangle', 'zoom_to_rect',
                           'zoom'))
        super(AnatomicalToolbar, self).__init__(canvas, parent)
