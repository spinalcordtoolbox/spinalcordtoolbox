from __future__ import division
from __future__ import absolute_import
import logging

import matplotlib as mpl

mpl.use('Qt4Agg')

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from PyQt4 import QtCore, QtGui

logger = logging.getLogger(__name__)


"""Base classes for creating GUI objects to create manually selected points.

    Example
    -------
    >>> dialog = BaseDialog(img_obj)
    >>> dialog.show()

"""


class AnatomicalParams(object):
    """The base parameter object for GUI configuration
    """
    def __init__(self, cmap='gray', aspect=1.0, interp='nearest', min=5., max=95.,
                 vmode='percentile', alpha=1.0):
        """

        Parameters
        ----------
        cmap : str
        aspect : float
        interp : str
        min : int
        max : int
        vmode : str
        alpha : float
        """
        self.cmap = cmap
        self.aspect = aspect
        self.interp = interp
        self.min = min
        self.max = max
        self.vmode = vmode
        self.alpha = alpha


class CrossHair(object):
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='m')
        self.ly = ax.axvline(color='m')

        self.txt = ax.text(.7, .9, '', transform=ax.transAxes)

    def refresh(self, event):
        x, y = event.xdata, event.ydata
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('%1.2f, %1.2f' % (x, y))


class AnatomicalCanvas(FigureCanvas):
    """Base canvas for anatomical views

    Attributes
    ----------
    point_selected_signal : QtCore.Signal
        Create a event when user clicks on the canvas

    """
    point_selected_signal = QtCore.Signal(int, int, int)

    def __init__(self, param, image, width=8, height=8, dpi=100):
        self.image = image
        self.param = param
        shape = image.data.shape
        self._x = shape[0] // 2
        self._y = shape[1] // 2
        self._z = shape[2] // 2

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
        self.x_points = []
        self.y_points = []
        self.plot_points = self.axes.plot(self.x_points,
                                          self.y_points,
                                          '.r',
                                          markersize=10)
        self.fig.canvas.mpl_connect('button_release_event', self.on_update)
        self.view = self.axes.imshow(data,
                                     aspect=self.param.aspect,
                                     cmap=self.param.cmap,
                                     interpolation=self.param.interp,
                                     vmin=self.param.vmin,
                                     vmax=self.param.vmax,
                                     alpha=self.param.alpha)

    def __repr__(self):
        return '{}: {}, {}, {}'.format(self.__class__, self._x, self._y, self._z)

    @QtCore.Slot(int, int, int)
    def on_update_plot(self, x, y, z):
        logger.debug((x, y, z))
        self._x, self._y, self._z = x, y, z
        self.update_plot()
        self.view.figure.canvas.draw()


class SagittalCanvas(AnatomicalCanvas):
    def __init__(self, param, img, width=8, height=8, dpi=100):
        super(SagittalCanvas, self).__init__(param, img, width, height, dpi)
        self._init_ui(self.image.data[:, :, self._z])
        self.crosshair = CrossHair(self.axes)

    def add_point(self, x, y, _):
        self.x_points.append(x)
        self.y_points.append(y)
        self.plot_points[0].set_xdata(self.x_points)
        self.plot_points[0].set_ydata(self.y_points)

    def update_plot(self):
        data = self.image.data[:, :, self._z]
        self.view.set_array(data)

    def on_update(self, event):
        if event.xdata > 0 and event.ydata > 0:
            self.point_selected_signal.emit(event.xdata, event.ydata, self._z)
            self.crosshair.refresh(event)


class CorrinalCanvas(AnatomicalCanvas):
    def __init__(self, param, img, width=8, height=8, dpi=100):
        super(CorrinalCanvas, self).__init__(param, img, width, height, dpi)
        self._init_ui(self.image.data[:, self._y, :])

    def add_point(self, x, y, z):
        pass

    def update_plot(self):
        data = self.image.data[:, self._y, :]
        self.view.set_array(data)

    def on_update(self, event):
        if event.xdata > 0 and event.ydata > 0:
            self.point_selected_signal.emit(event.xdata, self._y, event.ydata)


class AxialCanvas(AnatomicalCanvas):
    def __init__(self, param, img, width=8, height=8, dpi=100):
        super(AxialCanvas, self).__init__(param, img, width, height, dpi)
        self._init_ui(self.image.data[self._x, :, :])

    def add_point(self, x, y, z):
        pass

    def update_plot(self):
        data = self.image.data[self._x, :, :]
        self.view.set_array(data)

    def on_update(self, event):
        if event.xdata > 0 and event.ydata > 0:
            self.point_selected_signal.emit(self._x, event.xdata, event.ydata)


class AnatomicalToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        self.toolitems = (
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Back', 'Back to previous view', 'back', 'back'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom')
        )
        super(AnatomicalToolbar, self).__init__(canvas, parent)


class BaseDialog(QtGui.QDialog):
    update_canvas_signal = QtCore.Signal(int, int, int)

    def __init__(self, params, img, overlay=None):
        super(BaseDialog, self).__init__()

        self.params = params
        self.img = img
        self.overlay = overlay
        self._selected_points = []
        self._align_image()
        self._initUI()

    def _align_image(self):
        x, y, z, t, dx, dy, dz, dt = img.dim
        self.params.aspect = dx / dy
        self.params.offset = x * dx
        clip = np.percentile(self.img.data, (self.params.min, self.params.max))
        self.params.vmin, self.params.vmax = clip
        shape = self.img.data.shape
        dimension = [shape[0] * dx, shape[1] * dy, shape[2] * dz]
        max_size = max(dimension)
        self.x_offset = int(round(max_size - dimension[0]) / dx / 2)
        self.y_offset = int(round(max_size - dimension[1]) / dy / 2)
        self.z_offset = int(round(max_size - dimension[2]) / dz / 2)
        self.img.data = np.pad(self.img.data,
                               ((self.x_offset, self.x_offset),
                                (self.y_offset, self.y_offset),
                                (self.z_offset, self.z_offset)),
                               'constant',
                               constant_values=(0, 0))

    def _initUI(self):
        self.resize(600, 800)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        layout = QtGui.QVBoxLayout(self)

        self._initHeaderUI(layout)
        self._initMainUI(layout)
        self._initSliders(layout)
        self._initToolbar(layout)
        self._initFooter(layout)
        self.setFocus()

    def _initHeaderUI(self, parent):
        self.lb_status = QtGui.QLabel('Label Status')
        self.lb_status.setStyleSheet("color:black")
        self.lb_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_warning = QtGui.QLabel()
        self.lb_warning.setStyleSheet('color:red')
        self.lb_warning.setAlignment(QtCore.Qt.AlignCenter)

        parent.addWidget(self.lb_status)
        parent.addWidget(self.lb_warning)
        parent.addItem(QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding))

    def _initMainUI(self, parent):
        self.canvases = []
        for obj in [SagittalCanvas, AxialCanvas, CorrinalCanvas]:
            canvas = obj(self.params, self.img)
            parent.addWidget(canvas)
            canvas.point_selected_signal.connect(self.update_points)
            self.update_canvas_signal.connect(canvas.on_update_plot)
            self.canvases.append(canvas)

    def _initSliders(self, parent):
        self.sliders = []
        for x in self.img.data.shape:
            slider = QtGui.QSlider(QtCore.Qt.Horizontal)
            slider.setMaximum(x)
            slider.setTickPosition(QtGui.QSlider.TicksAbove)
            slider.singleStep()
            parent.addWidget(slider)
            self.sliders.append(slider)

        self.sliders[2].valueChanged.connect(self.update_z_axis)
        self.sliders[1].valueChanged.connect(self.update_y_axis)
        self.sliders[0].valueChanged.connect(self.update_x_axis)
        self.sliders[0].setValue(self.canvases[0]._x)
        self.sliders[1].setValue(self.canvases[0]._y)
        self.sliders[2].setValue(self.canvases[0]._z)

    def _initToolbar(self, parent):
        self.toolbar = AnatomicalToolbar(self.canvases[0], self)
        parent.addWidget(self.toolbar)

    def _initFooter(self, parent):
        ctrl_layout = QtGui.QHBoxLayout()

        btn_save_and_quit = QtGui.QPushButton('Save & Quit')
        btn_undo = QtGui.QPushButton('Undo')
        btn_help = QtGui.QPushButton('Help')

        ctrl_layout.addWidget(btn_help)
        ctrl_layout.addItem(QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        ctrl_layout.addWidget(btn_undo)
        ctrl_layout.addWidget(btn_save_and_quit)

        btn_save_and_quit.clicked.connect(self.press_save_and_quit)
        btn_help.clicked.connect(self.press_help)
        btn_undo.clicked.connect(self.press_undo)

        parent.addLayout(ctrl_layout)

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
        self._selected_points.append((x, y, z))
        self.update_canvas_signal.emit(x, y, z)

    @QtCore.Slot(int)
    def update_x_axis(self, x):
        _z = self.canvases[0]._z
        _y = self.canvases[0]._y
        self.update_points(x, _y, _z)

    @QtCore.Slot(int)
    def update_y_axis(self, y):
        _z = self.canvases[0]._z
        _x = self.canvases[0]._x
        self.update_points(_x, y, _z)

    @QtCore.Slot(int)
    def update_z_axis(self, z):
        _x = self.canvases[0]._x
        _y = self.canvases[0]._y
        self.update_points(_x, _y, z)

    def press_save_and_quit(self):
        self.data = None
        self.close()

    def press_help(self):
        pass
        # webbrowser.open(self.help_web_adress, new=0, autoraise=True)

    def press_undo(self):
        try:
            dump = self._selected_points.pop()
            self.update_status("Undo point ({})".format(dump))
        except IndexError:
            self.update_warning("There's no points to undo")
        logger.debug('{}'.format(dump))


class TestDialog(BaseDialog):
    pass


if __name__ == '__main__':
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    from scripts.msct_image import Image

    app = QtGui.QApplication(sys.argv)
    params = AnatomicalParams()
    img = Image('/Users/geper_admin/sct_testing_data/t2/t2.nii.gz')
    img.change_orientation('SAL')
    base_win = TestDialog(params, img)
    base_win.show()
    app.exec_()
