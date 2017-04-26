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



class SinglePlot(object):
    def __init__(self, ax, images, viewer,canvas, view, display_cross='hv', im_params=None,header=None):
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
        self.current_position = Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2),
                                         int(self.images[0].data.shape[2] / 2)])
        self.list_points=[]
        self.header=header


        self.remove_axis_number()
        self.connect_mpl_events()
        self.setup_intensity()

    def set_data_to_display(self,image, current_point,view):
        if view == 'ax':
            self.cross_to_display = [[[current_point.y, current_point.y], [-10000, 10000]],
                                     [[-10000, 10000], [current_point.z, current_point.z]]]
            self.aspect_ratio = self.viewer.aspect_ratio[0]
            return (image.data[current_point.x, :, :])
        elif view == 'cor':
            self.cross_to_display = [[[current_point.x, current_point.x], [-10000, 10000]],
                                     [[-10000, 10000], [current_point.z, current_point.z]]]
            self.aspect_ratio = self.viewer.aspect_ratio[1]
            return (image.data[:, current_point.y, :])
        elif view == 'sag':
            self.cross_to_display = [[[current_point.x, current_point.x], [-10000, 10000]],
                                     [[-10000, 10000], [current_point.y, current_point.y]]]
            self.aspect_ratio = self.viewer.aspect_ratio[2]
            return (image.data[:, :, current_point.z])

    def show_image(self,im_params,current_point):
        if not current_point:
            current_point=Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2),
                                         int(self.images[0].data.shape[2] / 2)])
        for i, image in enumerate(self.images):
            data_to_display = self.set_data_to_display(image,current_point,self.view)
            (my_cmap,my_interpolation,my_alpha)=self.set_image_parameters(im_params,i,mpl.cm)
            my_cmap.set_under('b', alpha=0)
            self.figs.append(self.axes.imshow(data_to_display, aspect=self.aspect_ratio, alpha=my_alpha))
            self.figs[-1].set_cmap(my_cmap)
            self.figs[-1].set_interpolation(my_interpolation)
            if(self.list_intensites):
                self.figs[-1].set_clim(self.list_intensites[0],self.list_intensites[1])

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

    def get_event_coordinates(self, event):
        #TODO : point bizarre
        point = None
        try:
            if self.view == 'ax':
                point = Coordinate([self.current_position.x,
                                    int(round(event.ydata)),
                                    int(round(event.xdata)), 1])
            elif self.view == 'cor':
                point = Coordinate([int(round(event.ydata)),
                                    self.current_position.y,
                                    int(round(event.xdata)), 1])
            elif self.view == 'sag':
                point = Coordinate([int(round(event.ydata)),
                                    int(round(event.xdata)),
                                    self.current_position.z, 1])
        except TypeError:
            self.header.update_text('warning_selected_point_not_in_image')
        return point

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

    def is_point_in_image(self, target_point):
        return 0 <= target_point.x < self.image_dim[0] and 0 <= target_point.y < self.image_dim[1] and 0 <= target_point.z < self.image_dim[2]

class SinglePlotMain(SinglePlot):
    def __init__(self, ax, images, viewer,canvas, view, display_cross='hv', im_params=None, secondary_plot=None,header=None,number_of_points=0):
        super(SinglePlotMain, self).__init__(ax, images, viewer, canvas, view, display_cross, im_params,header)
        self.secondary_plot=secondary_plot
        self.plot_points, = self.axes.plot([], [], '.r', markersize=10)
        self.show_image(self.im_params, current_point=None)
        self.current_slice=self.current_position.x
        self.number_of_points=number_of_points
        self.calculate_list_slices()
        self.update_slice(self.list_slices[0])
        #print(self.list_slices)

    def update_slice(self,new_slice):
        self.current_slice = new_slice
        self.current_position.x=new_slice
        if (self.view == 'ax'):
            self.figs[-1].set_data(self.images[0].data[self.current_slice, :, :])
        elif (self.view == 'cor'):
            self.figs[-1].set_data(self.images[0].data[:, self.current_slice, :])
        elif (self.view == 'sag'):
            self.figs[-1].set_data(self.images[0].data[:, :, self.current_slice])
        self.figs[-1].figure.canvas.draw()

    def set_line_to_display(self):
        if 'v' in self.display_cross:
            self.line_vertical.set_ydata(self.cross_to_display[0][0])
        if 'h' in self.display_cross:
            self.line_horizontal.set_xdata(self.cross_to_display[1][1])

    def add_point_to_list_points(self,current_point):
        if len(self.list_points)<self.number_of_points:
            self.list_points.append(current_point)
            if len(self.list_points)==self.number_of_points:
                self.header.update_text('ready_to_save_and_quit')
            else:
                self.header.update_text('update',len(self.list_points),self.number_of_points)
        else:
            self.header.update_text('warning_all_points_done_already')

    def on_event_motion(self, event):
        if event.button == 3 and event.inaxes == self.axes:  # right click
            if self.get_event_coordinates(event):
                self.change_intensity(event)
                self.change_intensity_on_secondary_plot(event)

    def on_event_release(self, event):
        if self.get_event_coordinates(event):
            if event.button == 1:  # left click
                self.add_point_to_list_points(self.get_event_coordinates(event))
                self.draw_dots()
                self.jump_to_new_slice()
            elif event.button == 3:  # right click
                self.change_intensity(event)
                self.change_intensity_on_secondary_plot(event)

    def jump_to_new_slice(self):
        if len(self.list_points)<self.number_of_points:
            self.update_slice(self.list_slices[len(self.list_points)])
            self.secondary_plot.draw_line('v')

    def change_intensity_on_secondary_plot(self,event):
        if self.secondary_plot:
            self.secondary_plot.change_intensity(event)

    def refresh(self):
        self.figs[-1].figure.canvas.draw()

    def draw_dots(self):
        def select_right_dimensions(ipoint,view):
            if view =='ax':
                return ipoints.z,ipoints.y
            elif view=='cor':
                return ipoints.x,ipoints.z
            elif view == 'sag':
                return ipoints.z,ipoints.y

        x_data, y_data = [], []
        for ipoints in self.list_points:
            if ipoints.x == self.current_slice:
                x,y=select_right_dimensions(ipoints,self.view)
                x_data.append(x)
                y_data.append(y)
        self.plot_points.set_xdata(x_data)
        self.plot_points.set_ydata(y_data)
        self.refresh()

    def calculate_list_slices(self):
        self.list_slices=[]
        increment=int(self.image_dim[0]/self.number_of_points)
        for ii in range (0,self.number_of_points-1):
            self.list_slices.append(ii*increment)
        self.list_slices.append(self.image_dim[0]-1)


class SinglePlotSecond(SinglePlot):
    def __init__(self, ax, images, viewer,canvas,main_single_plot, view, display_cross='hv', im_params=None,header=None):
        super(SinglePlotSecond,self).__init__(ax, images, viewer,canvas, view, display_cross, im_params,header)
        self.main_plot=main_single_plot

        self.show_image(self.im_params, current_point=None)
        self.add_line('v')  # add_line is used in stead of draw_line because in draw_line we also remove the previous line.

    def add_line(self,display_cross):

        if 'h' in display_cross:
            self.line_horizontal = Line2D(self.cross_to_display[1][1], self.cross_to_display[1][0], color='white')
            self.axes.add_line(self.line_horizontal)
        if 'v' in display_cross:
            self.line_vertical = Line2D(self.cross_to_display[0][1], self.cross_to_display[0][0], color='white')
            self.axes.add_line(self.line_vertical)

    def draw_line(self,display_cross):
        self.line_vertical.remove()
        self.add_line(display_cross)
        self.refresh()

    def refresh(self):
        self.show_image(self.im_params,self.current_position)
        self.figs[0].figure.canvas.draw()

    def on_event_motion(self, event):
        if event.button == 1 and event.inaxes == self.axes:  # left click
            if self.get_event_coordinates(event):
                self.current_position = self.get_event_coordinates(event)
                self.draw_line('v')
                self.main_plot.show_image(self.im_params, self.current_position)
                self.main_plot.update_slice(self.current_position.x)
                self.main_plot.draw_dots()

        elif event.button == 3 and event.inaxes == self.axes:  # right click
            if self.get_event_coordinates(event):
                self.change_intensity(event)

    def on_event_release(self, event):
        if self.get_event_coordinates(event):
            if event.button == 1:  # left click
                self.current_position = self.get_event_coordinates(event)
                self.draw_line('v')
                self.main_plot.show_image(self.im_params, self.current_position)
                self.main_plot.refresh()
            elif event.button == 3:  # right click
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

    def update_title_text_general(self, key,nbpt=0,nbfin=0):
        if(key=='ready_to_save_and_quit'):
            self.lb_status.setText('You can save and quit')
            self.lb_status.setStyleSheet("color:green")
        elif(key=='warning_all_points_done_already'):
            self.lb_warning.setText('You have placed all needed points. \n'
                                   'If you made a mistake, you may use \'undo\'.')
            self.lb_warning.setStyleSheet("color:red")
        elif(key=='warning_undo_beyond_first_point'):
            self.lb_warning.setText('Please place your first dot.')
            self.lb_warning.setStyleSheet("color:red")
        elif(key=='warning_selected_point_not_in_image'):
            self.lb_warning.setText('The point you selected in not in the image. Please try again.')
            self.lb_warning.setStyleSheet("color:red")
        elif(key=='update'):
            self.lb_status.setText('You have maid ' +str(nbpt)+ ' points out of ' + str(nbfin)+'.')
            self.lb_status.setStyleSheet("color:black")
        else:
            self.lb_warning.setText(key + ' : Unknown key')
            self.lb_warning.setStyleSheet("color:red")

class Header(HeaderCore):
    def update_text(self,key,nbpt=0,nbfin=0):
        self.lb_warning.setText('\n')
        if(key=='welcome'):
            self.lb_status.setText('Please click in the the center of the center line. \n'
                                   'If it is invisible, you may skip it.')
            self.lb_status.setStyleSheet("color:black")
        elif(key=='warning_skip_not_defined'):
            self.lb_warning.setText('This option is not used in Manual Mode. \n')
            self.lb_warning.setStyleSheet("color:red")
        else:
            self.update_title_text_general(key,nbpt,nbfin)



class MainPannelCore(object):
    def __init__(self,
                 images,
                 im_params,window,header):
        self.header=header
        self.window=window
        self.layout_global=QtGui.QVBoxLayout()
        self.layout_option_settings = QtGui.QHBoxLayout()
        self.layout_central = QtGui.QHBoxLayout()
        self.layout_central.setDirection(1)
        self.images=images
        self.im_params=im_params
        self.current_position = Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2), int(self.images[0].data.shape[2] / 2)])
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
        self.main_plot=SinglePlotMain(axis, self.images, self, view='ax', display_cross='', im_params=self.im_params,canvas=self.canvas_main,header=self.header,number_of_points=5)

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
        self.second_plot=SinglePlotSecond(axis, self.images, self, view='sag', display_cross='', im_params=self.im_params,canvas=self.canvas_second,main_single_plot=self.main_plot,header=self.header)
        self.main_plot.secondary_plot=self.second_plot

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

    def __init__(self,images,im_params,window,header):
        super(MainPannel, self).__init__(images,im_params,window,header)

        self.add_main_view()
        self.add_secondary_view()
        #self.add_controller_pannel()

        self.merge_layouts()

class ControlButtonsCore(object):
    def __init__(self,main_plot,window,header):
        self.main_plot = main_plot
        self.window=window
        self.help_web_adress='http://www.google.com'
        self.header=header

        self.layout_buttons=QtGui.QHBoxLayout()
        self.layout_buttons.setAlignment(QtCore.Qt.AlignRight)
        self.layout_buttons.setContentsMargins(10,80,15,160)
        self.add_help_button()
        self.add_undo_button()
        self.add_save_and_quit_button()

    def add_save_and_quit_button(self):
        btn_save_and_quit=QtGui.QPushButton('Save & Quit')
        self.layout_buttons.addWidget(btn_save_and_quit)
        btn_save_and_quit.clicked.connect(self.press_save_and_quit)

    def add_undo_button(self):
        btn_undo=QtGui.QPushButton('Undo')
        self.layout_buttons.addWidget(btn_undo)
        btn_undo.clicked.connect(self.press_undo)

    def add_help_button(self):
        btn_help=QtGui.QPushButton('Help')
        self.layout_buttons.addWidget(btn_help)
        btn_help.clicked.connect(self.press_help)

    def press_help(self):
        webbrowser.open(self.help_web_adress, new=0, autoraise=True)

    def press_save_and_quit(self):
        def rewrite_list_points(list_points):
            list_points_useful_notation=''
            for coord in list_points:
                if list_points_useful_notation:
                    list_points_useful_notation += ':'
                list_points_useful_notation = list_points_useful_notation + str(coord.x) + ',' + \
                                              str(coord.y) + ',' + str(coord.z) + ',' + str(coord.value)
            return list_points_useful_notation

        self.window.str_points_final=rewrite_list_points(self.main_plot.list_points)

    def press_undo(self):
        if self.main_plot.list_points:
            del self.main_plot.list_points[-1]
            self.main_plot.draw_dots()
            self.header.update_text('update',len(self.main_plot.list_points),self.main_plot.number_of_points)
            self.main_plot.jump_to_new_slice()
        else:
            self.header.update_text('warning_undo_beyond_first_point')


class WindowCore(object):

    def __init__(self,list_input, visualization_parameters=None):
        self.images = self.keep_only_images(list_input)
        self.im_params = visualization_parameters
        self.str_points_final=''

        self.mean_intensity = []
        self.std_intensity = []

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

    def start(self):
        return self.list_points_useful_notation

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
            header.update_text('welcome')
            return (header)

        def add_main_pannel(layout_main,window,header):
            main_pannel = MainPannel(self.images,self.im_params,window,header)
            layout_main.addLayout(main_pannel.layout_global)
            return main_pannel

        def add_control_buttons(layout_main,window):
            control_buttons = ControlButtonsCore(self.main_pannel.main_plot,window,self.header)
            layout_main.addLayout(control_buttons.layout_buttons)
            return control_buttons


        (window, system) = launch_main_window()
        layout_main = add_layout_main(window)
        self.header = add_header(layout_main)
        self.main_pannel = add_main_pannel(layout_main,self,self.header)
        self.control_buttons = add_control_buttons(layout_main,self)
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






















