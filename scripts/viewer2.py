import sys
import PyQt4.QtGui as QtGui
import PyQt4.QtCore as QtCore
import msct_image as Image

import sys
from msct_parser import Parser
from msct_image import Image
from bisect import bisect
from numpy import arange, max, pad, linspace, mean, median, std, percentile
import numpy as np
from msct_types import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import sct_utils as sct
from time import time
from copy import copy

from matplotlib.widgets import Slider, Button, RadioButtons

import webbrowser



class HeaderCore(object):

    def __init__(self):
        self.define_layout_header()
        self.add_lb_status()
        self.add_lb_warning()

    def add_lb_status(self):
        self.lb_status = QtGui.QLabel('Label Alerte')
        self.lb_status.setContentsMargins(10, 10, 10, 0)
        self.lb_status.setAlignment(QtCore.Qt.AlignCenter)
        self.layout_header.addWidget(self.lb_status)

    def add_lb_warning(self):
        self.lb_warning = QtGui.QLabel('Label Warning')
        self.lb_warning.setContentsMargins(10, 10, 10, 10)
        self.lb_warning.setAlignment(QtCore.Qt.AlignCenter)
        self.layout_header.addWidget(self.lb_warning)

    def define_layout_header(self):
        self.layout_header = QtGui.QVBoxLayout()
        self.layout_header.setAlignment(QtCore.Qt.AlignTop)
        self.layout_header.setContentsMargins(0,30,0,80)

class Header(HeaderCore):

    def update_text(self,key):
        if(key=='start'):
            self.lb_status.setText('header.lb_status')
            self.lb_warning.setText('header.lb_warning')
            self.lb_warning.setStyleSheet("color:red")

class MainPannelCore(object):

    def __init__(self):
        self.layout_global=QtGui.QVBoxLayout()
        self.layout_option_settings = QtGui.QHBoxLayout()
        self.layout_central = QtGui.QHBoxLayout()

    def add_main_view(self):
        layout_view = QtGui.QVBoxLayout()
        layout_view.setAlignment(QtCore.Qt.AlignTop)
        layout_view.setAlignment(QtCore.Qt.AlignRight)

        self.lb_title_main_view = QtGui.QLabel('Main View')
        self.lb_title_main_view.setAlignment(QtCore.Qt.AlignCenter)
        layout_view.addWidget(self.lb_title_main_view)
        layout_view.addWidget(self.create_image())
        self.layout_central.addLayout(layout_view)

    def add_secondary_view(self):
        layout_view = QtGui.QVBoxLayout()
        layout_view.setAlignment(QtCore.Qt.AlignTop)
        layout_view.setAlignment(QtCore.Qt.AlignRight)

        self.lb_title_secondary_view = QtGui.QLabel('Secondary View')
        self.lb_title_secondary_view.setAlignment(QtCore.Qt.AlignCenter)

        layout_view.addWidget(self.lb_title_secondary_view)
        layout_view.addWidget(self.create_image())
        self.layout_central.addLayout(layout_view)

    def add_controller_pannel(self):
        pass

    def create_image(self):
        image_label = QtGui.QLabel('')
        image_test = QtGui.QPixmap('/home/apopov/Documents/dev/sct/image_test.jpg')
        image_label.setPixmap(image_test)
        return image_label

    def merge_layouts(self):
        self.layout_global.addLayout(self.layout_option_settings)
        self.layout_global.addLayout(self.layout_central)

    def add_option_settings(self):
        pass

class MainPannel(MainPannelCore):

    def add_controller_pannel(self):
        layout_title_and_controller=QtGui.QVBoxLayout()
        lb_title = QtGui.QLabel('Label Choice')
        lb_title.setAlignment(QtCore.Qt.AlignCenter)
        lb_title.setContentsMargins(0,30,0,0)
        layout_title_and_controller.addWidget(lb_title)

        layout_controller = QtGui.QHBoxLayout()
        layout_controller.setAlignment(QtCore.Qt.AlignTop)
        layout_controller.setAlignment(QtCore.Qt.AlignCenter)

        l1=QtGui.QLabel('1')
        l1.setAlignment(QtCore.Qt.AlignCenter)
        l1.setContentsMargins(0,0,35,0)
        l2 = QtGui.QLabel('2')
        l2.setAlignment(QtCore.Qt.AlignCenter)
        l2.setContentsMargins(20, 0, 0, 0)

        s1=QtGui.QSlider()
        s2 = QtGui.QSlider()

        s1.setMaximumHeight(250)
        s2.setMaximumHeight(250)

        layout_controller.addWidget(l1)
        layout_controller.addWidget(s1)
        layout_controller.addWidget(s2)
        layout_controller.addWidget(l2)

        layout_title_and_controller.addLayout(layout_controller)

        self.layout_central.addLayout(layout_title_and_controller)

    def __init__(self):
        super(MainPannel, self).__init__()

        """ Left Pannel"""
        self.add_secondary_view()
        #self.add_controller_pannel()
        """ Right Pannel"""
        self.add_main_view()

        self.merge_layouts()

class ControlButtonsCore(object):
    def __init__(self):
        self.layout_buttons=QtGui.QHBoxLayout()
        self.layout_buttons.setAlignment(QtCore.Qt.AlignRight)
        self.layout_buttons.setContentsMargins(10,80,15,160)
        self.add_help_button()
        self.add_undo_button()
        self.add_save_and_quit_button()

    def add_save_and_quit_button(self):
        btn_save_and_quit=QtGui.QPushButton('Save & Quit')
        self.layout_buttons.addWidget(btn_save_and_quit)

    def add_undo_button(self):
        self.btn_undo=QtGui.QPushButton('Undo')
        self.layout_buttons.addWidget(self.btn_undo)

    def add_help_button(self):
        self.btn_help=QtGui.QPushButton('Help')
        self.layout_buttons.addWidget(self.btn_help)

class WindowCore(object):

    def __init__(self,list_input, visualization_parameters=None):
        self.images = self.keep_only_images(list_input)
        self.im_params = visualization_parameters

        """ Initialisation of plot """
        self.fig = plt.figure(figsize=(8, 8))
        self.fig.subplots_adjust(bottom=0.1, left=0.1)
        self.fig.patch.set_facecolor('lightgrey')

        """ Pad the image so that it is square in axial view (useful for zooming) """
        self.image_dim = self.images[0].data.shape
        nx, ny, nz, nt, px, py, pz, pt = self.images[0].dim
        self.im_spacing = [px, py, pz]
        self.aspect_ratio = [float(self.im_spacing[1]) / float(self.im_spacing[2]),
                             float(self.im_spacing[0]) / float(self.im_spacing[2]),
                             float(self.im_spacing[0]) / float(self.im_spacing[1])]
        self.offset = [0.0, 0.0, 0.0]
        self.current_point = Coordinate([int(nx / 2), int(ny / 2), int(nz / 2)])

        self.windows = []
        self.press = [0, 0]

        self.mean_intensity = []
        self.std_intensity = []

        self.last_update = time()

        self.update_freq = 1.0 / 15.0  # 10 Hz


    def keep_only_images(self,list_input):
        # TODO: check same space
        # TODO: check if at least one image
        images=[]
        for im in list_input:
            if isinstance(im, Image):
                images.append(im)
            else:
                print "Error, one of the images is actually not an image..."
        return images

    def compute_offset(self):
        array_dim = [self.image_dim[0]*self.im_spacing[0], self.image_dim[1]*self.im_spacing[1], self.image_dim[2]*self.im_spacing[2]]
        index_max = np.argmax(array_dim)
        max_size = array_dim[index_max]
        self.offset = [int(round((max_size - array_dim[0]) / self.im_spacing[0]) / 2),
                       int(round((max_size - array_dim[1]) / self.im_spacing[1]) / 2),
                       int(round((max_size - array_dim[2]) / self.im_spacing[2]) / 2)]

    def pad_data(self):
        for image in self.images:
            image.data = pad(image.data,
                             ((self.offset[0], self.offset[0]),
                              (self.offset[1], self.offset[1]),
                              (self.offset[2], self.offset[2])),
                             'constant',
                             constant_values=(0, 0))

    def setup_intensity(self):
        # TODO: change for segmentation images
        for i, image in enumerate(self.images):
            if str(i) in self.im_params.images_parameters:
                vmin = self.im_params.images_parameters[str(i)].vmin
                vmax = self.im_params.images_parameters[str(i)].vmax
                vmean = self.im_params.images_parameters[str(i)].vmean
                if self.im_params.images_parameters[str(i)].vmode == 'percentile':
                    flattened_volume = image.flatten()
                    first_percentile = percentile(flattened_volume[flattened_volume > 0], int(vmin))
                    last_percentile = percentile(flattened_volume[flattened_volume > 0], int(vmax))
                    mean_intensity = percentile(flattened_volume[flattened_volume > 0], int(vmean))
                    std_intensity = last_percentile - first_percentile
                elif self.im_params.images_parameters[str(i)].vmode == 'mean-std':
                    mean_intensity = (float(vmax) + float(vmin)) / 2.0
                    std_intensity = (float(vmax) - float(vmin)) / 2.0

            else:
                flattened_volume = image.flatten()
                first_percentile = percentile(flattened_volume[flattened_volume > 0], 0)
                last_percentile = percentile(flattened_volume[flattened_volume > 0], 99)
                mean_intensity = percentile(flattened_volume[flattened_volume > 0], 98)
                std_intensity = last_percentile - first_percentile

            self.mean_intensity.append(mean_intensity)
            self.std_intensity.append(std_intensity)

            min_intensity = mean_intensity - std_intensity
            max_intensity = mean_intensity + std_intensity

            for window in self.windows:
                window.figs[i].set_clim(min_intensity, max_intensity)

    def is_point_in_image(self, target_point):
        return 0 <= target_point.x < self.image_dim[0] and 0 <= target_point.y < self.image_dim[1] and 0 <= target_point.z < self.image_dim[2]

    def change_intensity(self, event, plot=None):
        if abs(event.xdata - self.press[0]) < 1 and abs(event.ydata - self.press[1]) < 1:
            self.press = event.xdata, event.ydata
            return

        if time() - self.last_update <= self.update_freq:
            return

        self.last_update = time()

        xlim, ylim = self.windows[0].axes.get_xlim(), self.windows[0].axes.get_ylim()
        mean_intensity_factor = (event.xdata - xlim[0]) / float(xlim[1] - xlim[0])
        std_intensity_factor = (event.ydata - ylim[1]) / float(ylim[0] - ylim[1])
        mean_factor = self.mean_intensity[0] - (mean_intensity_factor - 0.5) * self.mean_intensity[0] * 3.0
        std_factor = self.std_intensity[0] + (std_intensity_factor - 0.5) * self.std_intensity[0] * 2.0
        min_intensity = mean_factor - std_factor
        max_intensity = mean_factor + std_factor

        for window in self.windows:
            window.change_intensity(min_intensity, max_intensity)

    def get_event_coordinates(self, event, plot=None):
        point = None
        if plot.view == 1:
            point = Coordinate([self.current_point.x,
                                int(round(event.ydata)),
                                int(round(event.xdata)), 1])
        elif plot.view == 2:
            point = Coordinate([int(round(event.ydata)),
                                self.current_point.y,
                                int(round(event.xdata)), 1])
        elif plot.view == 3:
            point = Coordinate([int(round(event.ydata)),
                                int(round(event.xdata)),
                                self.current_point.z, 1])
        return point

    def draw(self):
        for window in self.windows:
            window.fig.figure.canvas.draw()

    def start(self):
        plt.show()




class Window(WindowCore):

    def __init__(self):
        super(WindowCore,self).__init__()
        self.set_layout_and_launch_viewer()


    def set_layout_and_launch_viewer(self):
        def launch_main_window():
            system = QtGui.QApplication(sys.argv)
            w = QtGui.QWidget()
            w.resize(740, 850)
            w.setWindowTitle('Hello world')
            w.show()
            return (w, system)

        def add_layout_main( window):
            layout_main = QtGui.QVBoxLayout()
            layout_main.setAlignment(QtCore.Qt.AlignTop)
            window.setLayout(layout_main)
            return layout_main

        def add_header(layout_main):
            header = Header()
            layout_main.addLayout(header.layout_header)
            header.update_text('start')
            return (header)

        def add_main_pannel(layout_main):
            main_pannel = MainPannel()
            layout_main.addLayout(main_pannel.layout_global)
            return main_pannel

        def add_control_buttons(layout_main):
            control_buttons = ControlButtonsCore()
            layout_main.addLayout(control_buttons.layout_buttons)
            return control_buttons

        (window, system) = launch_main_window()
        layout_main = add_layout_main(window)
        self.header = add_header(layout_main)
        self.main_pannel = add_main_pannel(layout_main)
        self.control_buttons = add_control_buttons(layout_main)
        window.setLayout(layout_main)
        sys.exit(system.exec_())


Window()







