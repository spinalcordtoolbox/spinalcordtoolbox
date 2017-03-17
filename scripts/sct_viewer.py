#!/usr/bin/env python
#########################################################################################
#
# Visualizer for MRI volumes
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Created: 2015-01-30
#
# Notes on how to use classes in this script.
# If you are interested into selecting manually some points in an image, you can use the following code.

# from sct_viewer import ClickViewer
# from msct_image import Image
#
# im_input = Image('my_image.nii.gz')
#
# im_input_SAL = im_input.copy()
# # SAL orientation is mandatory
# im_input_SAL.change_orientation('SAL')
# # The viewer is composed by a primary plot and a secondary plot. The primary plot is the one you will click points in.
# # The secondary plot will help you go throughout slices in another dimensions to help manual selection.
# viewer = ClickViewer(im_input_SAL, orientation_subplot=['sag', 'ax'])
# viewer.number_of_slices = X  # Change X appropriately.
# viewer.gap_inter_slice = Y  # this number should reflect image spacing
# viewer.calculate_list_slices()
# # start the viewer that ask the user to enter a few points along the spinal cord
# mask_points = viewer.start()
# print mask_points

#
# About the license: see the file LICENSE.TXT
#########################################################################################

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


class SinglePlot:
    """
        This class manages mouse events on one image.
    """
    def __init__(self, ax, images, viewer, view=2, display_cross='hv', im_params=None):
        self.axes = ax
        self.images = images  # this is a list of images
        self.viewer = viewer
        self.view = view
        self.display_cross = display_cross
        self.image_dim = self.images[0].data.shape
        self.figs = []
        self.cross_to_display = None
        self.aspect_ratio = None
        self.zoom_factor = 1.0

        for i, image in enumerate(images):
            data_to_display = self.set_data_to_display(image)
            # ?! image parameters ? C'est des options avancees pour plus tard ?
            (my_cmap,my_interpolation,my_alpha)=self.set_image_parameters(im_params,i,cm)
            my_cmap.set_under('b', alpha=0)
            self.figs.append(self.axes.imshow(data_to_display, aspect=self.aspect_ratio, alpha=my_alpha))
            self.figs[-1].set_cmap(my_cmap)
            self.figs[-1].set_interpolation(my_interpolation)

        # ?! pourquoi on a besoin de ticks et qu'est ce que c'est que set_axis_bgcolor.
        self.axes.set_axis_bgcolor('black')
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        self.draw_line(display_cross)


    def draw_line(self,display_cross):
        self.line_horizontal = Line2D(self.cross_to_display[1][1], self.cross_to_display[1][0], color='white')
        self.line_vertical = Line2D(self.cross_to_display[0][1], self.cross_to_display[0][0], color='white')
        if 'h' in display_cross:
            self.axes.add_line(self.line_horizontal)
        if 'v' in display_cross:
            self.axes.add_line(self.line_vertical)


    def set_image_parameters(self,im_params,i,cm):
        if str(i) in im_params.images_parameters:
            return(copy(cm.get_cmap(im_params.images_parameters[str(i)].cmap)),im_params.images_parameters[str(i)].interp,float(im_params.images_parameters[str(i)].alpha))
        else:
            return (cm.get_cmap('gray'), 'nearest', 1.0)

    def set_data_to_display(self,image):
        #?! cross to display, est ce que c'est une croix a afficher ? sur toutes les images ?
        if self.view == 1:
            self.cross_to_display = [[[self.viewer.current_point.y, self.viewer.current_point.y], [-10000, 10000]],
                                     [[-10000, 10000], [self.viewer.current_point.z, self.viewer.current_point.z]]]
            self.aspect_ratio = self.viewer.aspect_ratio[0]
            return( image.data[int(self.image_dim[0] / 2), :, :] )

        elif self.view == 2:
            self.cross_to_display = [[[self.viewer.current_point.x, self.viewer.current_point.x], [-10000, 10000]],
                                     [[-10000, 10000], [self.viewer.current_point.z, self.viewer.current_point.z]]]
            self.aspect_ratio = self.viewer.aspect_ratio[1]
            return (image.data[:, int(self.image_dim[1] / 2), :])
        elif self.view == 3:
            self.cross_to_display = [[[self.viewer.current_point.x, self.viewer.current_point.x], [-10000, 10000]],
                                     [[-10000, 10000], [self.viewer.current_point.y, self.viewer.current_point.y]]]
            self.aspect_ratio = self.viewer.aspect_ratio[2]
            return (image.data[:, :, int(self.image_dim[2] / 2)])

    def connect(self):
        """
        connect to all the events we need
        :return:
        """
        self.cidpress_click = self.figs[0].figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidscroll = self.figs[0].figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cidrelease = self.figs[0].figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.figs[0].figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def draw(self):
        self.figs[0].figure.canvas.draw()

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

        if self.view == 1:
            if 0 <= target_slice < self.images[0].data.shape[0]:
                if data_update:
                    for i, image in enumerate(self.images):
                        self.figs[i].set_data(image.data[target_slice, :, :])
                if 'v' in self.display_cross:
                    self.line_vertical.set_ydata(self.cross_to_display[0][0])
                if 'h' in self.display_cross:
                    self.line_horizontal.set_xdata(self.cross_to_display[1][1])
        elif self.view == 2:
            if 0 <= target_slice < self.images[0].data.shape[1]:
                if data_update:
                    for i, image in enumerate(self.images):
                        self.figs[i].set_data(image.data[:, target_slice, :])
                if 'v' in self.display_cross:
                    self.line_vertical.set_ydata(self.cross_to_display[0][0])
                if 'h' in self.display_cross:
                    self.line_horizontal.set_xdata(self.cross_to_display[1][1])
        elif self.view == 3:
            if 0 <= target_slice < self.images[0].data.shape[2]:
                if data_update:
                    for i, image in enumerate(self.images):
                        self.figs[i].set_data(image.data[:, :, target_slice])
                if 'v' in self.display_cross:
                    self.line_vertical.set_ydata(self.cross_to_display[0][0])
                if 'h' in self.display_cross:
                    self.line_horizontal.set_xdata(self.cross_to_display[1][1])

        self.figs[0].figure.canvas.draw()

    def on_press(self, event):
        """
        when pressing on the screen, add point into a list, then change current slice
        if finished, close the window and send the result
        :param event:
        :return:
        """
        if event.button == 1 and event.inaxes == self.axes:
            self.viewer.on_press(event, self)

        return

    def change_intensity(self, min_intensity, max_intensity, id_image=0):
        self.figs[id_image].set_clim(min_intensity, max_intensity)
        self.figs[id_image].figure.canvas.draw()

    def on_motion(self, event):
        if event.button == 1 and event.inaxes == self.axes:
            return self.viewer.on_motion(event, self)

        elif event.button == 3 and event.inaxes == self.axes:
            return self.viewer.change_intensity(event, self)

        else:
            return

    def on_release(self, event):
        if event.button == 1:
            return self.viewer.on_release(event, self)

        elif event.button == 3:
            return self.viewer.change_intensity(event, self)

        else:
            return

    def update_xy_lim(self, x_center=None, y_center=None, x_scale_factor=1.0, y_scale_factor=1.0, zoom=True):
        # get the current x and y limits
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
            if 0.005 < self.zoom_factor * scale_factor <= 3.0:
                self.zoom_factor *= scale_factor

                self.axes.set_xlim([x_center - x_left * x_scale_factor, x_center + x_right * x_scale_factor])
                self.axes.set_ylim([y_center - y_top * y_scale_factor, y_center + y_bottom * y_scale_factor])
                self.figs[0].figure.canvas.draw()
        else:
            self.axes.set_xlim([x_center - x_left * x_scale_factor, x_center + x_right * x_scale_factor])
            self.axes.set_ylim([y_center - y_top * y_scale_factor, y_center + y_bottom * y_scale_factor])
            self.figs[0].figure.canvas.draw()

    def on_scroll(self, event):
        """
        when scrooling with the wheel, image is zoomed toward position on the screen
        :param event:
        :return:
        """
        if event.inaxes == self.axes:
            base_scale = 0.5
            xdata, ydata = event.xdata, event.ydata

            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1.0
                print event.button

            self.update_xy_lim(x_center=xdata, y_center=ydata,
                               x_scale_factor=scale_factor, y_scale_factor=scale_factor,
                               zoom=True)

        return




class Viewer(object):
    def __init__(self, list_input, visualization_parameters=None):
        self.images = self.keep_only_images(list_input)
        self.im_params = visualization_parameters

        """ Initialisation of plot """
        self.fig = plt.figure(figsize=(8, 8))
        self.fig.subplots_adjust(bottom=0.1, left=0.1)
        self.fig.patch.set_facecolor('lightgrey')

        """ Pad the image so that it is square in axial view (useful for zooming) """
        # ?! definition des attributs de la classe Image
        self.image_dim = self.images[0].data.shape
        nx, ny, nz, nt, px, py, pz, pt = self.images[0].dim
        self.im_spacing = [px, py, pz]
        self.aspect_ratio = [float(self.im_spacing[1]) / float(self.im_spacing[2]),
                             float(self.im_spacing[0]) / float(self.im_spacing[2]),
                             float(self.im_spacing[0]) / float(self.im_spacing[1])]
        self.offset = [0.0, 0.0, 0.0]
        # ?! Coordinate ? Pourquoi pas seulement mettre les positions dans une liste ?
        self.current_point = Coordinate([int(nx / 2), int(ny / 2), int(nz / 2)])

        self.windows = []
        self.press = [0, 0]

        self.mean_intensity = []
        self.std_intensity = []

        self.last_update = time()
        self.update_freq = 1.0/15.0  # 10 Hz

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
        # ?! Est ce que c'est le contraste, et si oui, comment exactement il fonctionne sur la page
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
        #?! On part tjs de 0 ? donc ca veut dire qu'il y a des coordonees negatives ?
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


class ThreeViewer(Viewer):
    """
    This class is a visualizer for volumes (3D images) and ask user to click on axial slices.
    Assumes AIL orientation
    """
    def __init__(self, list_images, visualization_parameters=None):
        if isinstance(list_images, Image):
            list_images = [list_images]
        if not visualization_parameters:
            visualization_parameters = ParamMultiImageVisualization([ParamImageVisualization()])
        super(ThreeViewer, self).__init__(list_images, visualization_parameters)

        self.compute_offset()
        self.pad_data()

        self.current_point = Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2), int(self.images[0].data.shape[2] / 2)])

        ax = self.fig.add_subplot(222)
        self.windows.append(SinglePlot(ax=ax, images=self.images, viewer=self, view=1, im_params=visualization_parameters))  # SAL --> axial

        ax = self.fig.add_subplot(223)
        self.windows.append(SinglePlot(ax=ax, images=self.images, viewer=self, view=2, im_params=visualization_parameters))  # SAL --> frontal

        ax = self.fig.add_subplot(221)
        self.windows.append(SinglePlot(ax=ax, images=self.images, viewer=self, view=3, im_params=visualization_parameters))  # SAL --> sagittal

        for window in self.windows:
            window.connect()

        self.setup_intensity()

    def move(self, event, plot):
        is_in_axes = False
        for window in self.windows:
            if event.inaxes == window.axes:
                is_in_axes = True
        if not is_in_axes:
            return

        if event.xdata and abs(event.xdata - self.press[0]) < 0.5 and abs(event.ydata - self.press[1]) < 0.5:
            self.press = event.xdata, event.ydata
            return

        if time() - self.last_update <= self.update_freq:
            return

        self.last_update = time()
        self.current_point = self.get_event_coordinates(event, plot)
        point = [self.current_point.x, self.current_point.y, self.current_point.z]
        for window in self.windows:
            if window is plot:
                window.update_slice(point, data_update=False)
            else:
                window.update_slice(point, data_update=True)

        self.press = event.xdata, event.ydata
        return

    def on_press(self, event, plot=None):
        if event.button == 1:
            return self.move(event, plot)
        else:
            return

    def on_motion(self, event, plot=None):
        if event.button == 1:
            return self.move(event, plot)
        else:
            return

    def on_release(self, event, plot=None):
        if event.button == 1:
            return self.move(event, plot)
        else:
            return


class ClickViewer(Viewer):
    """
    This class is a visualizer for volumes (3D images) and ask user to click on axial slices.
    Assumes SAL orientation
    orientation_subplot: list of two views that will be plotted next to each other. The first view is the main one (right) and the second view is the smaller one (left). Orientations are: ax, sag, cor.
    """
    def __init__(self, list_images, visualization_parameters=None, orientation_subplot=['ax', 'sag'], title='', input_type='centerline'):

        # Ajust the input parameters into viewer objects.
        if isinstance(list_images, Image):
            list_images = [list_images]
        if not visualization_parameters:
            visualization_parameters = ParamMultiImageVisualization([ParamImageVisualization()])
        super(ClickViewer, self).__init__(list_images, visualization_parameters)

        self.declaration_global_variables(orientation_subplot,title)

        self.compute_offset() # ?!
        self.pad_data()       # ?!

        self.current_point = Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2), int(self.images[0].data.shape[2] / 2)]) #?!

        """ Display axes, specific to viewer """
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(1, 3)

        """ Main plot on the right"""
        ax = self.fig.add_subplot(gs[0, 1:], axisbg='k')
        self.windows.append(SinglePlot(ax, self.images, self, view=self.orientation[self.primary_subplot], display_cross='', im_params=visualization_parameters))
        self.set_main_plot()

        """Smaller plot on the left"""
        ax = self.fig.add_subplot(gs[0, 0], axisbg='k')
        self.windows.append(SinglePlot(ax, self.images, self, view=self.orientation[self.secondary_subplot], display_cross=self.set_display_cross(), im_params=visualization_parameters))

        """ Connect buttons to user actions"""
        for window in self.windows:
            window.connect()

        """ Create Buttons"""
        self.create_button_help()

        """ Compute slices to display """
        self.calculate_list_slices()

        """ Variable to check if all slices have been processed """
        self.all_processed = False
        self.setup_intensity()

        """ Manage closure of viewer"""
        self.enable_custom_points = False
        self.fig.canvas.mpl_connect('close_event', self.close_window)
        self.closed = False
        self.input_type = input_type

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

    def declaration_global_variables(self,orientation_subplot,title):
        self.title = title  # title to display in main figure
        self.orientation = {'ax': 1, 'cor': 2, 'sag': 3}
        self.primary_subplot = orientation_subplot[0]
        self.secondary_subplot = orientation_subplot[1]
        self.dic_axis_buttons={}

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
            return('v')
        else:
            return('h')

    def create_button_help(self):
        ax_help = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.dic_axis_buttons['help']=ax_help
        button_help = Button(ax_help, 'Help')
        self.fig.canvas.mpl_connect('button_press_event', self.help)

    def calculate_list_slices(self, starting_slice=-1):
        if self.number_of_slices != 0 and self.gap_inter_slice != 0:  # mode multiple points with fixed gap

            # if starting slice is not provided, middle slice is used
            # starting slice must be an integer, in the range of the image [0, #slices]
            if starting_slice == -1:
                starting_slice = int(self.image_dim[self.orientation[self.primary_subplot]-1] / 2)

            first_slice = starting_slice - (self.number_of_slices / 2) * self.gap_inter_slice
            last_slice = starting_slice + (self.number_of_slices / 2) * self.gap_inter_slice
            if first_slice < 0:
                first_slice = 0
            if last_slice >= self.image_dim[self.orientation[self.primary_subplot]-1]:
                last_slice = self.image_dim[self.orientation[self.primary_subplot]-1] - 1
            self.list_slices = [int(item) for item in
                                linspace(first_slice, last_slice, self.number_of_slices, endpoint=True)]
        elif self.number_of_slices != 0:
            self.list_slices = [int(item) for item in
                                linspace(0, self.image_dim[self.orientation[self.primary_subplot]-1] - 1, self.number_of_slices, endpoint=True)]
            if self.list_slices[-1] != self.image_dim[self.orientation[self.primary_subplot]-1] - 1:
                self.list_slices.append(self.image_dim[self.orientation[self.primary_subplot]-1] - 1)
        elif self.gap_inter_slice != 0:
            self.list_slices = list(arange(0, self.image_dim[self.orientation[self.primary_subplot]-1], self.gap_inter_slice))
            if self.list_slices[-1] != self.image_dim[self.orientation[self.primary_subplot]-1] - 1:
                self.list_slices.append(self.image_dim[self.orientation[self.primary_subplot]-1] - 1)
        else:
            self.gap_inter_slice = int(max([round(self.image_dim[self.orientation[self.primary_subplot]-1] / 15.0), 1]))
            self.number_of_slices = int(round(self.image_dim[self.orientation[self.primary_subplot]-1] / self.gap_inter_slice))
            self.list_slices = [int(item) for item in
                                linspace(0, self.image_dim[self.orientation[self.primary_subplot]-1] - 1, self.number_of_slices, endpoint=True)]
            if self.list_slices[-1] != self.image_dim[self.orientation[self.primary_subplot]-1] - 1:
                self.list_slices.append(self.image_dim[self.orientation[self.primary_subplot]-1] - 1)

        point = [self.current_point.x, self.current_point.y, self.current_point.z]
        point[self.orientation[self.primary_subplot]-1] = self.list_slices[self.current_slice]
        for window in self.windows:
            if window.view == self.orientation[self.secondary_subplot]:
                window.update_slice(point, data_update=False)
            else:
                window.update_slice(point, data_update=True)

        self.windows[1].axes.set_title('Click and hold\nto move around')
        if self.title == '':
            self.title = 'Please select a new point on slice ' + str(self.list_slices[self.current_slice]) + '/' + str(
                self.image_dim[self.orientation[self.primary_subplot]-1] - 1) + ' (' + str(self.current_slice + 1) + '/' + str(len(self.list_slices)) + ')'
        self.windows[0].axes.set_title(self.title)

    def compute_offset(self):
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

    def set_not_custom_target_points(self,event):
        if self.primary_subplot == 'ax':
            return( Coordinate([int(self.list_slices[self.current_slice]), int(event.ydata) - self.offset[1], int(event.xdata) - self.offset[2], 1]))
        elif self.primary_subplot == 'cor':
            return ( Coordinate([int(event.ydata) - self.offset[0], int(self.list_slices[self.current_slice]), int(event.xdata) - self.offset[2], 1]) )
        elif self.primary_subplot == 'sag':
            return ( Coordinate([int(event.ydata) - self.offset[0], int(event.xdata) - self.offset[1], int(self.list_slices[self.current_slice]), 1]) )

    def set_custom_target_points(self,event):
        if self.primary_subplot == 'ax':
            return ( Coordinate( [int(self.current_point.x), int(event.ydata) - self.offset[1], int(event.xdata) - self.offset[2], 1]))
        elif self.primary_subplot == 'cor':
            return (Coordinate( [int(event.ydata) - self.offset[0], int(self.current_point.y), int(event.xdata) - self.offset[2], 1]))
        elif self.primary_subplot == 'sag':
            return ( Coordinate([int(event.ydata) - self.offset[0], int(event.xdata) - self.offset[1], self.current_point.z, 1]))

    def check_point_is_valid(self,target_point,plot):
        if(self.is_point_in_image(target_point)):
            return True
        else:
            title_obj = self.windows[0].axes.set_title('The point you selected in not in the image. Please try again.')
            plt.setp(title_obj, color='r')
            plot.draw()
            return False

    def update_title_text(self,key):
        if(key=='way_automatic_next_point'):
            title_obj = self.windows[0].axes.set_title('Please select a new point on slice ' +
                                                       str(self.list_slices[self.current_slice]) + '/' +
                                                       str(self.image_dim[
                                                               self.orientation[self.primary_subplot] - 1] - 1) + ' (' +
                                                       str(self.current_slice + 1) + '/' +
                                                        str(len(self.list_slices)) + ')')
            plt.setp(title_obj, color='k')

        elif(key=='way_custom_next_point'):
            title_obj = self.windows[0].axes.set_title(
                'Automatic sliding disabled\nPlease click on spinal cord center\nand close the window once finished\n(# points = ' + str(
                    len(self.list_points)) + ')')
            plt.setp(title_obj, color='k')

        elif(key=='way_custom_start'):
            title_obj = self.windows[0].axes.set_title('Automatic sliding disabled\nPlease click on spinal cord center\nand close the window once finished\n(# points = ' + str(len(self.list_points)) + ')')
            plt.setp(title_obj, color='k')

    def are_all_images_processed(self):
        if self.current_slice < len(self.list_slices):
            return False
        else:
            for coord in self.list_points:
                if self.list_points_useful_notation != '':
                    self.list_points_useful_notation += ':'
                self.list_points_useful_notation = self.list_points_useful_notation + str(coord.x) + ',' + str(
                    coord.y) + ',' + str(coord.z) + ',' + str(coord.value)
            self.all_processed = True
            plt.close()
            return True

    def on_press_main_window(self,event,plot):
        if not self.enable_custom_points:
            target_point = self.set_not_custom_target_points(event)
        else:
            target_point = self.set_custom_target_points(event)

        if self.check_point_is_valid(target_point, plot):
            self.list_points.append(target_point)
            point = [self.current_point.x, self.current_point.y, self.current_point.z]

            if not self.enable_custom_points:
                self.current_slice += 1

                if not self.are_all_images_processed():
                    point[self.orientation[self.secondary_subplot] - 1] = self.list_slices[self.current_slice]
                    self.current_point = Coordinate(point)
                    self.windows[1].update_slice([point[2], point[0], point[1]], data_update=False)
                    self.windows[0].update_slice(self.list_slices[self.current_slice])
                    self.update_title_text('way_automatic_next_point')
                    plot.draw()

            else:
                self.draw_points(self.windows[0], self.current_point.x)
                self.windows[0].update_slice(point, data_update=True)
                self.update_title_text('way_custom_next_point')
                plot.draw()

    def on_press_secondary_window(self,event,plot):
        is_in_axes = False
        for window in self.windows:
            if event.inaxes == window.axes:
                is_in_axes = True
        if not is_in_axes:  # ?!
            return

        if self.input_type == 'centerline':
            self.enable_custom_points = True

        self.update_title_text('way_custom_start')
        plot.draw()

        self.last_update = time()
        self.current_point = self.get_event_coordinates(event, plot)
        point = [self.current_point.x, self.current_point.y, self.current_point.z]
        for window in self.windows:
            if window is plot:
                window.update_slice(point, data_update=False)
            else:
                self.draw_points(window, self.current_point.x)
                window.update_slice(point, data_update=True)

    def on_press(self, event, plot=None):
        # event inaxes ?!
        if event.inaxes and plot.view == self.orientation[self.primary_subplot]:
            self.on_press_main_window(event,plot)
        elif event.inaxes and plot.view == self.orientation[self.secondary_subplot]:
            self.on_press_secondary_window(event,plot)

    def draw_points(self, window, current_slice):
        if window.view == self.orientation[self.primary_subplot]:
            x_data, y_data = [], []
            for pt in self.list_points:
                if pt.x == current_slice:
                    x_data.append(pt.z + self.offset[2])
                    y_data.append(pt.y + self.offset[1])
            self.plot_points.set_xdata(x_data)
            self.plot_points.set_ydata(y_data)
            self.fig.canvas.draw()

    def on_release(self, event, plot=None):
        """
        This subplot refers to the secondary window. It captures event "release"
        :param event:
        :param plot:
        :return:
        """
        if event.button == 1 and event.inaxes == plot.axes and plot.view == self.orientation[self.secondary_subplot]:
            point = [self.current_point.x, self.current_point.y, self.current_point.z]
            if not self.enable_custom_points:
                point[self.orientation[self.primary_subplot]-1] = self.list_slices[self.current_slice]
            for window in self.windows:
                if window is plot:
                    window.update_slice(point, data_update=False)
                else:
                    window.update_slice(point, data_update=True)
                    self.draw_points(window, self.current_point.y)
        return

    def on_motion(self, event, plot=None):
        """
        This subplot refers to the secondary window. It captures event "motion"
        :param event:
        :param plot:
        :return:
        """
        if event.button == 1 and event.inaxes and plot.view == self.orientation[self.secondary_subplot] and time() - self.last_update > self.update_freq:
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

    def get_results(self):
        if self.list_points:
            return self.list_points
        else:
            return None

    def help(self, event):
        if event.inaxes == self.dic_axis_buttons['help']:
            webbrowser.open('https://sourceforge.net/p/spinalcordtoolbox/wiki/Home/', new=0, autoraise=True)

    def start(self):
        super(ClickViewer, self).start()

        if self.all_processed:
            return self.list_points_useful_notation
        else:
            return None

    def close_window(self, event):
        for coord in self.list_points:
            if self.list_points_useful_notation != '':
                self.list_points_useful_notation += ':'
            self.list_points_useful_notation = self.list_points_useful_notation + str(coord.x) + ',' + str(coord.y) + ',' + str(coord.z) + ',' + str(coord.value)
        self.closed = True


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Volume Viewer')
    parser.add_option(name="-i",
                      type_value=[[','], 'file'],
                      description="Images to display.",
                      mandatory=True,
                      example="anat.nii.gz")

    parser.add_option(name='-mode',
                      type_value='multiple_choice',
                      description='Display mode.'
                                  '\nviewer: standard three-window viewer.'
                                  '\naxial: one-window viewer for manual centerline.\n',
                      mandatory=False,
                      default_value='viewer',
                      example=['viewer', 'axial'])

    parser.add_option(name='-param',
                      type_value=[[':'], 'str'],
                      description='Parameters for visualization. '
                                  'Separate images with \",\". Separate parameters with \":\".'
                                  '\nid: number of image in the "-i" list'
                                  '\ncmap: image colormap'
                                  '\ninterp: image interpolation. Accepts: [\'nearest\' | \'bilinear\' | \'bicubic\' | \'spline16\' | '
                                                                            '\'spline36\' | \'hanning\' | \'hamming\' | \'hermite\' | \'kaiser\' | '
                                                                            '\'quadric\' | \'catrom\' | \'gaussian\' | \'bessel\' | \'mitchell\' | '
                                                                            '\'sinc\' | \'lanczos\' | \'none\' |]'
                                  '\nvmin:'
                                  '\nvmax:'
                                  '\nvmean:'
                                  '\nperc: ',
                      mandatory=False,
                      example=['cmap=red:vmin=0:vmax=1', 'cmap=grey'])

    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1', '2'])

    return parser


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

def prepare(list_images):
    fname_images, orientation_images = [], []
    for fname_im in list_images:
        from sct_image import orientation
        orientation_images.append(orientation(Image(fname_im), get=True, verbose=False))
        path_fname, file_fname, ext_fname = sct.extract_fname(fname_im)
        reoriented_image_filename = 'tmp.' + sct.add_suffix(file_fname + ext_fname, "_SAL")
        sct.run('sct_image -i ' + fname_im + ' -o ' + reoriented_image_filename + ' -setorient SAL -v 0', verbose=False)
        fname_images.append(reoriented_image_filename)
    return fname_images, orientation_images


def clean():
    sct.run('rm -rf ' + 'tmp.*', verbose=False)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    parser = get_parser()

    arguments = parser.parse(sys.argv[1:])

    fname_images, orientation_images = prepare(arguments["-i"])
    list_images = [Image(fname) for fname in fname_images]

    mode = arguments['-mode']

    param_image1 = ParamImageVisualization()
    visualization_parameters = ParamMultiImageVisualization([param_image1])
    if "-param" in arguments:
        param_images = arguments['-param']
        # update registration parameters
        for param in param_images:
            visualization_parameters.addImage(param)

    if mode == 'viewer':
        # 3 views
        viewer = ThreeViewer(list_images, visualization_parameters)
        viewer.start()
    elif mode == 'axial':
        # only one axial view
        viewer = ClickViewer(list_images, visualization_parameters)
        viewer.start()
    clean()
