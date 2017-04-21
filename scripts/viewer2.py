import sys
import PyQt4.QtGui as QtGui
import PyQt4.QtCore as QtCore

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
from msct_image import Image

from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from abc import ABCMeta, abstractmethod

import webbrowser

class SinglePlot:
    """
        This class manages mouse events on one image.
    """
    def __init__(self, ax, images, viewer,canvas, view=2, display_cross='hv', im_params=None):
        self.axes = ax
        self.images = images
        self.viewer = viewer
        self.view = view
        self.display_cross = display_cross
        self.image_dim = self.images[0].data.shape
        self.figs = []
        self.cross_to_display = None
        self.aspect_ratio = None
        self.canvas=canvas
        self.last_update=time()
        self.mean_intensity = []
        self.std_intensity = []
        self.list_intensites=[]
        self.im_params = im_params
        self.current_point = Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2),
                                         int(self.images[0].data.shape[2] / 2)])
        self.plot_points, = self.axes.plot([], [], '.r', markersize=10)

        self.show_image(self.im_params,current_point=None)
        self.remove_axis_number()
        self.connect_mpl_events()
        self.setup_intensity()

    def show_image(self,im_params,current_point):
        def set_data_to_display(image, current_point):
            if self.view == 1:
                self.cross_to_display = [[[current_point.y, current_point.y], [-10000, 10000]],
                                         [[-10000, 10000], [current_point.z, current_point.z]]]
                self.aspect_ratio = self.viewer.aspect_ratio[0]
                return (image.data[int(self.image_dim[0] / 2), :, :])
            elif self.view == 2:
                self.cross_to_display = [[[current_point.x, current_point.x], [-10000, 10000]],
                                         [[-10000, 10000], [current_point.z, current_point.z]]]
                self.aspect_ratio = self.viewer.aspect_ratio[1]
                return (image.data[:, int(self.image_dim[1] / 2), :])
            elif self.view == 3:
                self.cross_to_display = [[[current_point.x, current_point.x], [-10000, 10000]],
                                         [[-10000, 10000], [current_point.y, current_point.y]]]
                self.aspect_ratio = self.viewer.aspect_ratio[2]
                return (image.data[:, :, int(self.image_dim[2] / 2)])

        if not current_point:
            current_point=Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2),
                                         int(self.images[0].data.shape[2] / 2)])
        for i, image in enumerate(self.images):
            data_to_display = set_data_to_display(image,current_point)
            (my_cmap,my_interpolation,my_alpha)=self.set_image_parameters(im_params,i,mpl.cm)
            my_cmap.set_under('b', alpha=0)
            self.figs.append(self.axes.imshow(data_to_display, aspect=self.aspect_ratio, alpha=my_alpha))
            self.figs[-1].set_cmap(my_cmap)
            self.figs[-1].set_interpolation(my_interpolation)
            if(self.list_intensites):
                self.figs[-1].set_clim(self.list_intensites[0],self.list_intensites[1])

    def update_slice(self, target, data_update=True):
        """
        This function change the viewer to update the current slice
        :param target: number of the slice to go on
        :param data_update: False if you don't want to update data
        :return:
        """
        if isinstance(target, list):
            target_slice = target[self.view - 1]
            list_remaining_views = list([0, 1, 2])
            list_remaining_views.remove(self.view - 1)
            self.cross_to_display[0][0] = [target[list_remaining_views[0]], target[list_remaining_views[0]]]
            self.cross_to_display[1][1] = [target[list_remaining_views[1]], target[list_remaining_views[1]]]
        else:
            target_slice = target

        if 0 <= target_slice < self.images[0].data.shape[int(self.view)-1]:
            if data_update:
                for i, image in enumerate(self.images):
                    if(self.view==1):
                        self.figs[i].set_data(image.data[target_slice, :, :])
                    elif(self.view==2):
                        self.figs[i].set_data(image.data[:, target_slice, :])
                    elif(self.view==3):
                        self.figs[i].set_data(image.data[:, :, target_slice])
            self.set_line_to_display()
        self.figs[0].figure.canvas.draw()

    def set_line_to_display(self):
        if 'v' in self.display_cross:
            self.line_vertical.set_ydata(self.cross_to_display[0][0])
        if 'h' in self.display_cross:
            self.line_horizontal.set_xdata(self.cross_to_display[1][1])

    def set_image_parameters(self,im_params,i,cm):
        if str(i) in im_params.images_parameters:
            return(copy(cm.get_cmap(im_params.images_parameters[str(i)].cmap)),im_params.images_parameters[str(i)].interp,float(im_params.images_parameters[str(i)].alpha))
        else:
            return (cm.get_cmap('gray'), 'nearest', 1.0)

    def refresh(self):
        self.figs[0].figure.canvas.draw()

    def remove_axis_number(self):
        self.axes.set_axis_bgcolor('black')
        self.axes.set_xticks([])
        self.axes.set_yticks([])

    def change_intensity(self, event):
        def calc_min_max_intensities(x, y):
            xlim, ylim = self.axes.get_xlim(), self.axes.get_ylim()
            mean_intensity_factor = (x - xlim[0]) / float(xlim[1] - xlim[0])
            std_intensity_factor = (y - ylim[1]) / float(ylim[0] - ylim[1])
            mean_factor = self.mean_intensity[0] - (mean_intensity_factor - 0.5) * self.mean_intensity[0] * 3.0
            std_factor = self.std_intensity[0] + (std_intensity_factor - 0.5) * self.std_intensity[0] * 2.0
            self.list_intensites=[mean_factor - std_factor, mean_factor + std_factor]
            return (mean_factor - std_factor, mean_factor + std_factor)
        def check_time_last_update(last_update):
            if time() - last_update < 1.0/15.0: # 10 Hz:
                return False
            else:
                return True
        if(check_time_last_update(self.last_update)):
            self.last_update = time()
            self.figs[-1].set_clim(calc_min_max_intensities(event.xdata,event.ydata))
            self.refresh()

    def setup_intensity(self):
        for i, image in enumerate(self.images):
            flattened_volume = image.flatten()
            first_percentile = percentile(flattened_volume[flattened_volume > 0], 0)
            last_percentile = percentile(flattened_volume[flattened_volume > 0], 99)
            mean_intensity = percentile(flattened_volume[flattened_volume > 0], 98)
            std_intensity = last_percentile - first_percentile

            self.mean_intensity.append(mean_intensity)
            self.std_intensity.append(std_intensity)

    def update_xy_lim(self, x_center=None, y_center=None, x_scale_factor=1.0, y_scale_factor=1.0, zoom=True):
        # get the current x and y limits
        zoom_factor = 1.0
        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()

        if x_center is None:
            x_center = (cur_xlim[1] - cur_xlim[0]) / 2.0
        if y_center is None:
            y_center = (cur_ylim[1] - cur_ylim[0]) / 2.0

        # Get distance from the cursor to the edge of the figure frame
        x_left = x_center - cur_xlim[0]
        x_right = cur_xlim[1] - x_center
        y_top = y_center - cur_ylim[0]
        y_bottom = cur_ylim[1] - y_center

        if zoom:
            scale_factor = (x_scale_factor + y_scale_factor) / 2.0
            if 0.005 < zoom_factor * scale_factor <= 3.0:
                zoom_factor *= scale_factor

                self.axes.set_xlim([x_center - x_left * x_scale_factor, x_center + x_right * x_scale_factor])
                self.axes.set_ylim([y_center - y_top * y_scale_factor, y_center + y_bottom * y_scale_factor])
                self.figs[0].figure.canvas.draw()
        else:
            self.axes.set_xlim([x_center - x_left * x_scale_factor, x_center + x_right * x_scale_factor])
            self.axes.set_ylim([y_center - y_top * y_scale_factor, y_center + y_bottom * y_scale_factor])
            self.figs[0].figure.canvas.draw()

    def connect_mpl_events(self):
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()
        self.canvas.mpl_connect('button_release_event',self.on_event_release)
        self.canvas.mpl_connect('scroll_event',self.on_event_scroll)
        self.canvas.mpl_connect('motion_notify_event',self.on_event_motion)

    def on_event_motion(self, event):
        if event.button == 1 and event.inaxes == self.axes: #left click
            pass
        elif event.button == 3 and event.inaxes == self.axes: #right click
            self.change_intensity(event)

    def on_event_release(self, event):
        if event.button == 1: # left click
            self.draw_dots(self.get_event_coordinates(event,1))
        elif event.button == 3: # right click
            self.change_intensity(event)

    def on_event_scroll(self, event):
        def calc_scale_factor(direction):
            base_scale = 0.5
            if direction == 'up':  # deal with zoom in
                return 1 / base_scale
            elif direction == 'down':  # deal with zoom out
                return base_scale
            else:  # deal with something that should never happen
                return 1

        if event.inaxes == self.axes:
            scale_factor=calc_scale_factor(event.button)
            self.update_xy_lim(x_center=event.xdata, y_center=event.ydata,
                               x_scale_factor=scale_factor, y_scale_factor=scale_factor,
                               zoom=True)

    def get_event_coordinates(self, event, axis):
        point = None
        if axis == 1:
            point = Coordinate([self.current_point.x,
                                int(round(event.ydata)),
                                int(round(event.xdata)), 1])
        elif axis == 2:
            point = Coordinate([int(round(event.ydata)),
                                self.current_point.y,
                                int(round(event.xdata)), 1])
        elif axis == 3:
            point = Coordinate([int(round(event.ydata)),
                                int(round(event.xdata)),
                                self.current_point.z, 1])
        return point

    def draw_dots(self, current_point):
        x_data, y_data = [], []
        self.refresh()
        x_data.append(current_point.z)
        y_data.append(current_point.y)
        self.plot_points.set_xdata(x_data)
        self.plot_points.set_ydata(y_data)
        self.show_image(self.im_params,current_point)
        self.refresh()

class SinglePlotSecond(SinglePlot,object):
    #TODO : inheritance etrange

    def __init__(self, ax, images, viewer,canvas,main_single_plot, view=2, display_cross='hv', im_params=None):
        super(SinglePlotSecond,self).__init__(ax, images, viewer,canvas, view, display_cross, im_params)
        self.main_plot=main_single_plot

        self.add_line('v')  # add_line is used in stead of draw_line because in draw_line we also remove the previous line.
        self.refresh()

    def add_line(self,display_cross):
        if 'h' in display_cross:
            self.line_horizontal = Line2D(self.cross_to_display[1][1], self.cross_to_display[1][0], color='white')
            self.axes.add_line(self.line_horizontal)
        if 'v' in display_cross:
            self.line_vertical = Line2D(self.cross_to_display[0][1], self.cross_to_display[0][0], color='white')
            self.axes.add_line(self.line_vertical)

    def draw_line(self,display_cross):
        self.line_vertical.remove()
        self.add_line('v')
        self.refresh()

    def show_image(self,im_params,current_point):
        def set_data_to_display(image, current_point=None):
            if self.view == 1:
                self.cross_to_display = [[[self.current_point.y, self.current_point.y], [-10000, 10000]],
                                         [[-10000, 10000], [self.current_point.z, self.current_point.z]]]
                self.aspect_ratio = self.viewer.aspect_ratio[0]
                return (image.data[int(self.image_dim[0] / 2), :, :])
            elif self.view == 2:
                self.cross_to_display = [[[self.current_point.x, self.current_point.x], [-10000, 10000]],
                                         [[-10000, 10000], [self.current_point.z, self.current_point.z]]]
                self.aspect_ratio = self.viewer.aspect_ratio[1]
                return (image.data[:, int(self.image_dim[1] / 2), :])
            elif self.view == 3:
                self.cross_to_display = [[[self.current_point.x, self.current_point.x], [-10000, 10000]],
                                         [[-10000, 10000], [self.current_point.y, self.current_point.y]]]
                self.aspect_ratio = self.viewer.aspect_ratio[2]
                return (image.data[:, :, int(self.image_dim[2] / 2)])

        if not current_point:
            current_point=Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2),
                                         int(self.images[0].data.shape[2] / 2)])
        for i, image in enumerate(self.images):
            data_to_display = set_data_to_display(image,current_point)
            (my_cmap,my_interpolation,my_alpha)=self.set_image_parameters(im_params,i,mpl.cm)
            my_cmap.set_under('b', alpha=0)
            self.figs.append(self.axes.imshow(data_to_display, aspect=self.aspect_ratio, alpha=my_alpha))
            self.figs[-1].set_cmap(my_cmap)
            self.figs[-1].set_interpolation(my_interpolation)

    def refresh(self):
        self.show_image(self.im_params,self.current_point)
        self.figs[0].figure.canvas.draw()

    def on_event_motion(self, event):
        if event.button == 1 and event.inaxes == self.axes: #left click
            self.current_point=self.get_event_coordinates(event,3)
            self.draw_line('v')
            self.main_plot.show_image(self.im_params,self.current_point)
            self.main_plot.refresh()
        elif event.button == 3 and event.inaxes == self.axes: #right click
            self.change_intensity(event)

    def on_event_release(self, event):
        if event.button == 1: # left click
            self.current_point=self.get_event_coordinates(event,3)
            self.draw_line('v')
            self.refresh()
        elif event.button == 3: # right click
            self.change_intensity(event)

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

    def __init__(self,
                 images,
                 im_params):
        self.layout_global=QtGui.QVBoxLayout()
        self.layout_option_settings = QtGui.QHBoxLayout()
        self.layout_central = QtGui.QHBoxLayout()
        self.layout_central.setDirection(1)
        self.images=images
        self.im_params=im_params
        self.current_point = Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2), int(self.images[0].data.shape[2] / 2)])
        nx, ny, nz, nt, px, py, pz, pt = self.images[0].dim
        self.im_spacing = [px, py, pz]
        self.aspect_ratio = [float(self.im_spacing[1]) / float(self.im_spacing[2]),
                             float(self.im_spacing[0]) / float(self.im_spacing[2]),
                             float(self.im_spacing[0]) / float(self.im_spacing[1])]

    def add_main_view(self):
        layout_view = QtGui.QVBoxLayout()

        fig = plt.figure()
        self.canvas_main = FigureCanvas(fig)

        layout_view.addWidget(self.canvas_main)
        self.layout_central.addLayout(layout_view)

        if not self.im_params:
            self.im_params = ParamMultiImageVisualization([ParamImageVisualization()])
        gs = mpl.gridspec.GridSpec(1, 1)
        axis = fig.add_subplot(gs[0, 0], axisbg='k')
        self.main_plot=SinglePlot(axis, self.images, self, view=1, display_cross='', im_params=self.im_params,canvas=self.canvas_main)

    def add_secondary_view(self):
        layout_view = QtGui.QVBoxLayout()

        fig = plt.figure()
        self.canvas_second = FigureCanvas(fig)

        layout_view.addWidget(self.canvas_second)
        self.layout_central.addLayout(layout_view)

        if not self.im_params:
            self.im_params = ParamMultiImageVisualization([ParamImageVisualization()])
        gs = mpl.gridspec.GridSpec(1, 1)
        axis = fig.add_subplot(gs[0, 0], axisbg='k')
        self.second_plot=SinglePlotSecond(axis, self.images, self, view=3, display_cross='', im_params=self.im_params,canvas=self.canvas_second,main_single_plot=self.main_plot)

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

    def __init__(self,images,im_params):
        super(MainPannel, self).__init__(images,im_params)

        self.add_main_view()
        self.add_secondary_view()
        #self.add_controller_pannel()

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
        #self.fig = plt.figure(figsize=(8, 8))
        #self.fig.subplots_adjust(bottom=0.1, left=0.1)
        #self.fig.patch.set_facecolor('lightgrey')

        """ Pad the image so that it is square in axial view (useful for zooming) """
        #self.image_dim = self.images[0].data.shape
        #nx, ny, nz, nt, px, py, pz, pt = self.images[0].dim
        #self.im_spacing = [px, py, pz]
        #self.aspect_ratio = [float(self.im_spacing[1]) / float(self.im_spacing[2]),
        #                     float(self.im_spacing[0]) / float(self.im_spacing[2]),
        #                     float(self.im_spacing[0]) / float(self.im_spacing[1])]
        #self.offset = [0.0, 0.0, 0.0]
        #self.current_point = Coordinate([int(nx / 2), int(ny / 2), int(nz / 2)])

        #self.windows = []
        #self.press = [0, 0]

        self.mean_intensity = []
        self.std_intensity = []

        #self.last_update = time()
        #self.update_freq = 1.0 / 15.0  # 10 Hz

    def keep_only_images(self,list_input):
        # TODO: check same space
        # TODO: check if at least one image
        from msct_image import Image
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



    def is_point_in_image(self, target_point):
        return 0 <= target_point.x < self.image_dim[0] and 0 <= target_point.y < self.image_dim[1] and 0 <= target_point.z < self.image_dim[2]

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
    def __init__(self,
                 list_images,
                 visualization_parameters=None,
                 orientation_subplot=['ax', 'sag'],
                 input_type='centerline'):


        # Ajust the input parameters into viewer objects.
        if isinstance(list_images, Image):
            list_images = [list_images]
        if not visualization_parameters:
            visualization_parameters = ParamMultiImageVisualization([ParamImageVisualization()])

        super(Window, self).__init__(list_images, visualization_parameters)

        #self.declaration_global_variables_general(orientation_subplot)

        #self.compute_offset()
        #self.pad_data()

        #self.current_point = Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2),
        #                                 int(self.images[0].data.shape[2] / 2)])

        #import matplotlib.gridspec as gridspec
        #gs = gridspec.GridSpec(1, 3)

        #ax = self.fig.add_subplot(gs[0, 1:], axisbg='k')
        #self.windows.append(
        #    SinglePlot(ax, self.images, self, view=self.orientation[self.primary_subplot], display_cross='',
        #               im_params=visualization_parameters))
        #self.set_main_plot()

        #ax = self.fig.add_subplot(gs[0, 0], axisbg='k')
        #self.windows.append(SinglePlot(ax, self.images, self, view=self.orientation[self.secondary_subplot],
                                       #display_cross=self.set_display_cross(), im_params=visualization_parameters))
        #self.windows[1].axes.set_title('Select the slice \n '
                                       #'to inspect. \n')

        #self.calculate_list_slices()


        #self.input_type = input_type



        self.set_layout_and_launch_viewer()

    def set_main_plot(self):
        self.plot_points, = self.windows[0].axes.plot([], [], '.r', markersize=10)
        if self.primary_subplot == 'ax':
            self.windows[0].axes.set_xlim([0, self.images[0].data.shape[2]])
            self.windows[0].axes.set_ylim([self.images[0].data.shape[1], 0])
        elif self.primary_subplot == 'cor':
            self.windows[0].axes.set_xlim([0, self.images[0].data.shape[2]])
            self.windows[0].axes.set_ylim([self.images[0].data.shape[0], 0])
        elif self.primary_subplot == 'sag':
            self.windows[0].axes.set_xlim([0, self.images[0].data.shape[0]])
            self.windows[0].axes.set_ylim([self.images[0].data.shape[1], 0])

    def declaration_global_variables_general(self, orientation_subplot):
        self.help_web_adress = 'https://sourceforge.net/p/spinalcordtoolbox/wiki/Home/'
        self.orientation = {'ax': 1, 'cor': 2, 'sag': 3}
        self.primary_subplot = orientation_subplot[0]
        self.secondary_subplot = orientation_subplot[1]
        self.dic_axis_buttons = {}
        self.closed = False

        self.current_slice = 0
        self.number_of_slices = 0
        self.gap_inter_slice = 0

        # specialized for Click viewer
        self.list_points = []
        self.list_points_useful_notation = ''

        # compute slices to display
        self.list_slices = []

    def set_display_cross(self):
        if self.primary_subplot == 'ax':
            return ('v')
        else:
            return ('h')

    def calculate_list_slices(self, starting_slice=-1):
        if self.number_of_slices != 0 and self.gap_inter_slice != 0:  # mode multiple points with fixed gap

            # if starting slice is not provided, middle slice is used
            # starting slice must be an integer, in the range of the image [0, #slices]
            if starting_slice == -1:
                starting_slice = int(self.image_dim[self.orientation[self.primary_subplot] - 1] / 2)

            first_slice = starting_slice - (self.number_of_slices / 2) * self.gap_inter_slice
            last_slice = starting_slice + (self.number_of_slices / 2) * self.gap_inter_slice
            if first_slice < 0:
                first_slice = 0
            if last_slice >= self.image_dim[self.orientation[self.primary_subplot] - 1]:
                last_slice = self.image_dim[self.orientation[self.primary_subplot] - 1] - 1
            self.list_slices = [int(item) for item in
                                linspace(first_slice, last_slice, self.number_of_slices, endpoint=True)]
        elif self.number_of_slices != 0:
            self.list_slices = [int(item) for item in
                                linspace(0, self.image_dim[self.orientation[self.primary_subplot] - 1] - 1,
                                         self.number_of_slices, endpoint=True)]
            if self.list_slices[-1] != self.image_dim[self.orientation[self.primary_subplot] - 1] - 1:
                self.list_slices.append(self.image_dim[self.orientation[self.primary_subplot] - 1] - 1)
        elif self.gap_inter_slice != 0:
            self.list_slices = list(
                arange(0, self.image_dim[self.orientation[self.primary_subplot] - 1], self.gap_inter_slice))
            if self.list_slices[-1] != self.image_dim[self.orientation[self.primary_subplot] - 1] - 1:
                self.list_slices.append(self.image_dim[self.orientation[self.primary_subplot] - 1] - 1)
        else:
            self.gap_inter_slice = int(
                max([round(self.image_dim[self.orientation[self.primary_subplot] - 1] / 15.0), 1]))
            self.number_of_slices = int(
                round(self.image_dim[self.orientation[self.primary_subplot] - 1] / self.gap_inter_slice))
            self.list_slices = [int(item) for item in
                                linspace(0, self.image_dim[self.orientation[self.primary_subplot] - 1] - 1,
                                         self.number_of_slices, endpoint=True)]
            if self.list_slices[-1] != self.image_dim[self.orientation[self.primary_subplot] - 1] - 1:
                self.list_slices.append(self.image_dim[self.orientation[self.primary_subplot] - 1] - 1)

        point = [self.current_point.x, self.current_point.y, self.current_point.z]
        point[self.orientation[self.primary_subplot] - 1] = self.list_slices[self.current_slice]
        for window in self.windows:
            if window.view == self.orientation[self.secondary_subplot]:
                window.update_slice(point, data_update=False)
            else:
                window.update_slice(point, data_update=True)

    def compute_offset(self):
        print(self.image_dim)
        if self.primary_subplot == 'ax':
            array_dim = [self.image_dim[1] * self.im_spacing[1], self.image_dim[2] * self.im_spacing[2]]
            index_max = np.argmax(array_dim)
            max_size = array_dim[index_max]
            self.offset = [0,
                           int(round((max_size - array_dim[0]) / self.im_spacing[1]) / 2),
                           int(round((max_size - array_dim[1]) / self.im_spacing[2]) / 2)]
        elif self.primary_subplot == 'cor':
            array_dim = [self.image_dim[0] * self.im_spacing[0], self.image_dim[2] * self.im_spacing[2]]
            index_max = np.argmax(array_dim)
            max_size = array_dim[index_max]
            self.offset = [int(round((max_size - array_dim[0]) / self.im_spacing[0]) / 2),
                           0,
                           int(round((max_size - array_dim[1]) / self.im_spacing[2]) / 2)]
        elif self.primary_subplot == 'sag':
            array_dim = [self.image_dim[0] * self.im_spacing[0], self.image_dim[1] * self.im_spacing[1]]
            index_max = np.argmax(array_dim)
            max_size = array_dim[index_max]
            self.offset = [int(round((max_size - array_dim[0]) / self.im_spacing[0]) / 2),
                           int(round((max_size - array_dim[1]) / self.im_spacing[1]) / 2),
                           0]

    def check_point_is_valid(self, target_point):
        if (self.is_point_in_image(target_point)):
            return True
        else:
            self.update_title_text_general('warning_selected_point_not_in_image')
            return False

    def update_title_text_general(self, key):
        if (key == 'ready_to_save_and_quit'):
            title_obj = self.windows[0].axes.set_title('You can save and quit. \n')
            plt.setp(title_obj, color='g')

        elif (key == 'warning_all_slices_are_done_already'):
            title_obj = self.windows[0].axes.set_title('You have processed all slices \n'
                                                       'If you made a mistake please use \'Redo\' \n'
                                                       'Otherwise, you can save and quit. \n')
            plt.setp(title_obj, color='g')

        elif (key == 'warning_redo_beyond_first_dot'):
            title_obj = self.windows[0].axes.set_title('Please, place your first dot. \n')
            plt.setp(title_obj, color='r')

        elif (key == 'warning_skip_not_defined'):
            title_obj = self.windows[0].axes.set_title('This option is not used in Manual Mode. \n')
            plt.setp(title_obj, color='r')

        elif (key == 'warning_selected_point_not_in_image'):
            title_obj = self.windows[0].axes.set_title('The point you selected in not in the image. Please try again.')
            plt.setp(title_obj, color='r')

        self.windows[0].draw()

    def is_there_next_slice(self):
        if self.current_slice < len(self.list_slices):
            return True
        else:
            self.update_title_text('ready_to_save_and_quit')
            return False

    def on_release(self, event, plot=None):
        """
        This subplot refers to the secondary window. It captures event "release"
        :param event:
        :param plot:
        :return:
        """
        if event.button == 1 and event.inaxes == plot.axes and plot.view == self.orientation[self.secondary_subplot]:
            point = [self.current_point.x, self.current_point.y, self.current_point.z]
            point[self.orientation[self.primary_subplot] - 1] = self.list_slices[self.current_slice]
            for window in self.windows:
                if window is plot:
                    window.update_slice(point, data_update=False)
                else:
                    window.update_slice(point, data_update=True)
                    self.draw_points(window, self.current_point.x)
        return
        pass
    '''
    def on_motion(self, event, plot=None):
        """
        This subplot refers to the secondary window. It captures event "motion"
        :param event:
        :param plot:
        :return:
        """
        if event.button == 1 and event.inaxes and plot.view == self.orientation[
            self.secondary_subplot] and time() - self.last_update > self.update_freq:
            is_in_axes = False
            for window in self.windows:
                if event.inaxes == window.axes:
                    is_in_axes = True
            if not is_in_axes:
                return

            self.last_update = time()
            self.current_point = self.get_event_coordinates(event, plot)
            point = [self.current_point.x, self.current_point.y, self.current_point.z]
            for window in self.windows:
                if window is plot:
                    window.update_slice(point, data_update=False)
                else:
                    self.draw_points(window, self.current_point.x)
                    window.update_slice(point, data_update=True)
        return
    '''

    def get_results(self):
        if self.list_points:
            return self.list_points
        else:
            return None

    def create_button_redo(self):
        ax = plt.axes([0.70, 0.90, 0.1, 0.075])
        self.dic_axis_buttons['redo'] = ax
        button_help = Button(ax, 'Redo')
        self.fig.canvas.mpl_connect('button_press_event', self.press_redo)

    def create_button_save_and_quit(self):
        ax = plt.axes([0.81, 0.90, 0.1, 0.075])
        self.dic_axis_buttons['save_and_quit'] = ax
        button_help = Button(ax, 'Save &\n'
                                 'Quit')
        self.fig.canvas.mpl_connect('button_press_event', self.press_save_and_quit)

    def remove_last_dot(self):
        if (len(self.list_points) > 1):
            self.list_points = self.list_points[0:len(self.list_points) - 1]
        else:
            self.list_points = []

    def save_data(self):
        for coord in self.list_points:
            if self.list_points_useful_notation != '':
                self.list_points_useful_notation += ':'
            self.list_points_useful_notation = self.list_points_useful_notation + str(coord.x) + ',' + \
                                               str(coord.y) + ',' + str(coord.z) + ',' + str(coord.value)

    def press_save_and_quit(self, event):
        if event.inaxes == self.dic_axis_buttons['save_and_quit']:
            self.save_data()
            self.closed = True
            plt.close('all')

    def press_redo(self, event):
        if event.inaxes == self.dic_axis_buttons['redo']:
            if self.current_slice > 0:
                self.current_slice += -1
                self.windows[0].update_slice(self.list_slices[self.current_slice])
                self.remove_last_dot()
                self.update_ui_after_redo()
            else:
                self.update_title_text('warning_redo_beyond_first_dot')

    def update_ui_after_redo(self):
        self.update_title_text('redo_done')
        self.draw_points(self.windows[0], self.current_point.z)

    def start(self):
        #super(ClickViewer, self).start()
        return self.list_points_useful_notation

    def are_all_slices_done(self):
        if self.current_slice < len(self.list_slices):
            return False
        else:
            self.update_title_text('warning_all_slices_are_done_already')
            return True

    def press_help(self, event):
        if event.inaxes == self.dic_axis_buttons['help']:
            webbrowser.open(self.help_web_adress, new=0, autoraise=True)

    def create_button_help(self):
        ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.dic_axis_buttons['help'] = ax
        button_help = Button(ax, 'Help')
        self.fig.canvas.mpl_connect('button_press_event', self.press_help)

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
            main_pannel = MainPannel(self.images,self.im_params)
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

class ParamMultiImageVisualization(object):
    """
    This class contains a dictionary with the params of multiple images visualization
    """

    def __init__(self, list_param):
        self.ids = []
        self.images_parameters = dict()
        for param_image in list_param:
            if isinstance(param_image, ParamImageVisualization):
                self.images_parameters[param_image.id] = param_image
            else:
                self.addImage(param_image)

    def addImage(self, param_image):
        param_im = ParamImageVisualization()
        param_im.update(param_image)
        if param_im.id != 0:
            if param_im.id in self.images_parameters:
                self.images_parameters[param_im.id].update(param_image)
            else:
                self.images_parameters[param_im.id] = param_im
        else:
            sct.printv("ERROR: parameters must contain 'id'", 1, 'error')

class ParamImageVisualization(object):
    def __init__(self, id='0', mode='image', cmap='gray', interp='nearest', vmin='0', vmax='99', vmean='98', vmode='percentile', alpha='1.0'):
        self.id = id
        self.mode = mode
        self.cmap = cmap
        self.interp = interp
        self.vmin = vmin
        self.vmax = vmax
        self.vmean = vmean
        self.vmode = vmode
        self.alpha = alpha

    def update(self, params):
        list_objects = params.split(',')
        for obj in list_objects:
            if len(obj) < 2:
                sct.printv('Please check parameter -param (usage changed from previous version)', 1, type='error')
            objs = obj.split('=')
            setattr(self, objs[0], objs[1])


class Observer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

class Observable(object):
    def __init__(self):
        self.observers = []

    def register(self, observer):
        if not observer in self.observers:
            self.observers.append(observer)

    def unregister(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def unregister_all(self):
        if self.observers:
            del self.observers[:]

    def update_observers(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)




















