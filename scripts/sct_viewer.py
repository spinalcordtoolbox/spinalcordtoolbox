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
# Modified: 2016-02-26
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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import sct_utils as sct
from time import time
from copy import copy


# from matplotlib.widgets import Slider, Button, RadioButtons


class SinglePlot:
    """
        This class manages mouse events on one image.
    """
    def __init__(self, ax, images, viewer, view=2, display_cross=True, im_params=None):
        self.axes = ax
        self.images = images  # this is a list of images
        self.viewer = viewer
        self.view = view
        self.display_cross = display_cross

        self.image_dim = self.images[0].data.shape
        self.figs = []

        self.cross_to_display = None
        self.aspect_ratio = None
        for i, image in enumerate(images):
            data_to_display = None
            if self.view == 1:
                self.cross_to_display = [[[self.viewer.current_point.y, self.viewer.current_point.y], [0, self.image_dim[1]]],
                                         [[0, self.image_dim[2]], [self.viewer.current_point.z, self.viewer.current_point.z]]]
                self.aspect_ratio = self.viewer.aspect_ratio[0]
                data_to_display = image.data[int(self.image_dim[0] / 2), :, :]

            elif self.view == 2:
                self.cross_to_display = [[[self.viewer.current_point.x, self.viewer.current_point.x], [0, self.image_dim[2]]],
                                         [[0, self.image_dim[0]], [self.viewer.current_point.z, self.viewer.current_point.z]]]
                self.aspect_ratio = self.viewer.aspect_ratio[1]
                data_to_display = image.data[:, int(self.image_dim[1] / 2), :]
            elif self.view == 3:
                self.cross_to_display = [[[self.viewer.current_point.x, self.viewer.current_point.x], [0, self.image_dim[1]]],
                                         [[0, self.image_dim[0]], [self.viewer.current_point.y, self.viewer.current_point.y]]]
                self.aspect_ratio = self.viewer.aspect_ratio[2]
                data_to_display = image.data[:, :, int(self.image_dim[2] / 2)]

            if str(i) in im_params.images_parameters:
                my_cmap = copy(cm.get_cmap(im_params.images_parameters[str(i)].cmap))
                my_interpolation = im_params.images_parameters[str(i)].interp
                my_alpha = float(im_params.images_parameters[str(i)].alpha)
            else:
                my_cmap = cm.get_cmap('gray')
                my_interpolation = 'nearest'
                my_alpha = 1.0

            my_cmap.set_under('b', alpha=0)
            self.figs.append(self.axes.imshow(data_to_display, aspect=self.aspect_ratio, alpha=my_alpha))
            self.figs[-1].set_cmap(my_cmap)
            self.figs[-1].set_interpolation(my_interpolation)

        self.axes.set_axis_bgcolor('black')
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        if display_cross:
            self.line_vertical = Line2D(self.cross_to_display[0][1], self.cross_to_display[0][0], color='white')
            self.line_horizontal = Line2D(self.cross_to_display[1][1], self.cross_to_display[1][0], color='white')
            self.axes.add_line(self.line_vertical)
            self.axes.add_line(self.line_horizontal)

        self.zoom_factor = 1.0

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
                if self.display_cross:
                    self.line_vertical.set_ydata(self.cross_to_display[0][0])
                    self.line_horizontal.set_xdata(self.cross_to_display[1][1])
        elif self.view == 2:
            if 0 <= target_slice < self.images[0].data.shape[1]:
                if data_update:
                    for i, image in enumerate(self.images):
                        self.figs[i].set_data(image.data[:, target_slice, :])
                if self.display_cross:
                    self.line_vertical.set_ydata(self.cross_to_display[0][0])
                    self.line_horizontal.set_xdata(self.cross_to_display[1][1])
        elif self.view == 3:
            if 0 <= target_slice < self.images[0].data.shape[2]:
                if data_update:
                    for i, image in enumerate(self.images):
                        self.figs[i].set_data(image.data[:, :, target_slice])
                if self.display_cross:
                    self.line_vertical.set_ydata(self.cross_to_display[0][0])
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
    def __init__(self, list_images, visualization_parameters=None):
        self.images = []
        for im in list_images:
            if isinstance(im, Image):
                self.images.append(im)
            else:
                print "Error, one of the images is actually not an image..."

            # TODO: check same space
            # TODO: check if at least one image

        self.im_params = visualization_parameters

        # initialisation of plot
        self.fig = plt.figure(figsize=(8, 8))
        self.fig.subplots_adjust(bottom=0.1, left=0.1)
        self.fig.patch.set_facecolor('lightgrey')

        # pad the image so that it is square in axial view (useful for zooming)
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
        self.update_freq = 1.0/15.0  # 10 Hz

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
        if event.xdata and abs(event.xdata - self.press[0]) < 1 and abs(event.ydata - self.press[1]) < 1:
            self.press = event.xdata, event.ydata
            return

        xlim, ylim = plot.axes.get_xlim(), plot.axes.get_ylim()
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
    Assumes AIL orientation
    """
    def __init__(self, list_images, visualization_parameters=None):
        if isinstance(list_images, Image):
            list_images = [list_images]
        if not visualization_parameters:
            visualization_parameters = ParamMultiImageVisualization([ParamImageVisualization()])
        super(ClickViewer, self).__init__(list_images, visualization_parameters)

        self.current_slice = 0
        self.number_of_slices = 0
        self.gap_inter_slice = 0

        self.compute_offset()
        self.pad_data()

        self.current_point = Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2), int(self.images[0].data.shape[2] / 2)])

        # display axes, specific to viewer
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(1, 3)

        ax = self.fig.add_subplot(gs[0, 1:], axisbg='k')
        self.windows.append(SinglePlot(ax, self.images, self, view=1, display_cross=False, im_params=visualization_parameters))
        self.plot_points, = self.windows[0].axes.plot([], [], '.r', markersize=10)
        self.windows[0].axes.set_xlim([0, self.images[0].data.shape[1]])
        self.windows[0].axes.set_ylim([self.images[0].data.shape[2], 0])

        ax = self.fig.add_subplot(gs[0, 0], axisbg='k')
        self.windows.append(SinglePlot(ax, self.images, self, view=3, display_cross=True, im_params=visualization_parameters))

        for window in self.windows:
            window.connect()

        # specialized for Click viewer
        self.list_points = []
        self.list_points_useful_notation = ''

        # compute slices to display
        self.list_slices = []
        if self.number_of_slices != 0 and self.gap_inter_slice != 0:  # mode multiple points with fixed gap
            central_slice = int(self.image_dim[0] / 2)
            first_slice = central_slice - (self.number_of_slices / 2) * self.gap_inter_slice
            last_slice = central_slice + (self.number_of_slices / 2) * self.gap_inter_slice
            if first_slice < 0:
                first_slice = 0
            if last_slice >= self.image_dim[0]:
                last_slice = self.image_dim[0] - 1
            self.list_slices = [int(item) for item in
                                linspace(first_slice, last_slice, self.number_of_slices, endpoint=True)]
        elif self.number_of_slices != 0:
            self.list_slices = [int(item) for item in
                                linspace(0, self.image_dim[0] - 1, self.number_of_slices, endpoint=True)]
            if self.list_slices[-1] != self.image_dim[0] - 1:
                self.list_slices.append(self.image_dim[0] - 1)
        elif self.gap_inter_slice != 0:
            self.list_slices = list(arange(0, self.image_dim[0], self.gap_inter_slice))
            if self.list_slices[-1] != self.image_dim[0] - 1:
                self.list_slices.append(self.image_dim[0] - 1)
        else:
            self.gap_inter_slice = int(max([round(self.image_dim[0] / 15.0), 1]))
            self.number_of_slices = int(round(self.image_dim[0] / self.gap_inter_slice))
            self.list_slices = [int(item) for item in
                                linspace(0, self.image_dim[0] - 1, self.number_of_slices, endpoint=True)]
            if self.list_slices[-1] != self.image_dim[0] - 1:
                self.list_slices.append(self.image_dim[0] - 1)

        self.current_point.x = self.list_slices[self.current_slice]
        point = [self.current_point.x, self.current_point.y, self.current_point.z]
        for window in self.windows:
            if window.view == 3:
                window.update_slice(point, data_update=False)
            else:
                window.update_slice(point, data_update=True)

        self.title = self.windows[0].axes.set_title('Please select a new point on slice ' + str(self.list_slices[self.current_slice]) + '/' + str(
            self.image_dim[1] - 1) + ' (' + str(self.current_slice + 1) + '/' + str(len(self.list_slices)) + ')')

        # variable to check if all slices have been processed
        self.all_processed = False

        self.setup_intensity()

    def compute_offset(self):
        array_dim = [self.image_dim[1] * self.im_spacing[1], self.image_dim[2] * self.im_spacing[2]]
        index_max = np.argmax(array_dim)
        max_size = array_dim[index_max]
        self.offset = [0,
                       int(round((max_size - array_dim[0]) / self.im_spacing[1]) / 2),
                       int(round((max_size - array_dim[1]) / self.im_spacing[2]) / 2)]

    def on_press(self, event, plot=None):
        if plot.view == 1:
            target_point = Coordinate([int(self.list_slices[self.current_slice]), int(event.ydata) - self.offset[1], int(event.xdata) - self.offset[2], 1])
            if self.is_point_in_image(target_point):
                self.list_points.append(target_point)

                self.current_slice += 1
                if self.current_slice < len(self.list_slices):
                    self.current_point.x = self.list_slices[self.current_slice]
                    self.windows[0].update_slice(self.list_slices[self.current_slice])
                    title_obj = self.windows[0].axes.set_title('Please select a new point on slice ' +
                                                    str(self.list_slices[self.current_slice]) + '/' +
                                                    str(self.image_dim[1] - 1) + ' (' +
                                                    str(self.current_slice + 1) + '/' +
                                                    str(len(self.list_slices)) + ')')
                    plt.setp(title_obj, color='k')
                    plot.draw()

                    point = [self.current_point.x, self.current_point.y, self.current_point.z]
                    self.windows[1].update_slice(point, data_update=False)
                else:
                    for coord in self.list_points:
                        if self.list_points_useful_notation != '':
                            self.list_points_useful_notation += ':'
                        self.list_points_useful_notation = self.list_points_useful_notation + str(coord.x) + ',' + str(
                            coord.y) + ',' + str(coord.z) + ',' + str(coord.value)
                    self.all_processed = True
                    plt.close()
            else:
                title_obj = self.windows[0].axes.set_title('The point you selected in not in the image. Please try again.')
                plt.setp(title_obj, color='r')
                plot.draw()

    def draw_points(self, window, current_slice):
        if window.view == 1:
            x_data, y_data = [], []
            for pt in self.list_points:
                if pt.x == current_slice:
                    x_data.append(pt.z + self.offset[2])
                    y_data.append(pt.y + self.offset[1])
            self.plot_points.set_xdata(x_data)
            self.plot_points.set_ydata(y_data)

    def on_release(self, event, plot=None):
        if event.button == 1 and plot.view == 3:
            self.current_point.x = self.list_slices[self.current_slice]
            point = [self.current_point.x, self.current_point.y, self.current_point.z]
            for window in self.windows:
                if window is plot:
                    window.update_slice(point, data_update=False)
                else:
                    self.draw_points(window, self.current_point.y)
                    window.update_slice(point, data_update=True)
        return

    def on_motion(self, event, plot=None):
        if event.button == 1 and plot.view == 3 and time() - self.last_update > self.update_freq:
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

    def start(self):
        super(ClickViewer, self).start()

        if self.all_processed:
            return self.list_points_useful_notation
        else:
            return None


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
        viewer = ThreeViewer(list_images, visualization_parameters)
        viewer.start()
    elif mode == 'axial':
        viewer = ClickViewer(list_images, visualization_parameters)
        viewer.start()
    clean()
