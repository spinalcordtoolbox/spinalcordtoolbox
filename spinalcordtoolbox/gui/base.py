import sys
import webbrowser
from copy import copy
from time import time
import os

import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
import matplotlib as mpl
import matplotlib.pyplot as plt
import sct_utils as sct
from matplotlib import cm
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D
from msct_image import Image
from msct_types import Coordinate
from numpy import pad, percentile

class ImagePlot(object):
    """
    Core class for displaying pictures.
    Defines the data to display, image parameters, on_scroll event, manages intensity and manages the click (in or out the picture,...).
    """
    def __init__(self, ax, images, viewer, canvas, view, line_direction='hv', im_params=None, header=None):
        self.axes = ax
        self.images = images
        self.viewer = viewer
        self.view = view
        self.line_direction = line_direction
        self.image_dim = self.images[0].data.shape
        self.figs = []
        self.cross_to_display = None
        self.aspect_ratio = None
        self.canvas = canvas
        self.last_update = time()
        self.mean_intensity = []
        self.std_intensity = []
        self.list_intensites = []
        self.im_params = im_params
        self.current_position = Coordinate(
            [int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2),
             int(self.images[0].data.shape[2] / 2)])
        self.list_points = []
        self.header = header
        self.dic_translate_label = self.define_translate_dic()

        self.remove_axis_number()
        self.connect_mpl_events()
        self.setup_intensity()

    def define_translate_dic(self):
        """
        Defines dictionnary to translate the software labels which range is [1;27] into anatomical labels which range is:
        {50;49} U {1} U [3,26]

        Returns
        -------
        dic        dictionnary
        """
        dic = {'1': 50,
               '2': 49,
               '3': 1,
               '4': 3, }
        for ii in range(5, 30):             # does not matter if the dictionnary is a bit too long. The number of possible labels is still 27.
            dic[str(ii)] = ii - 1
        return dic

    def set_data_to_display(self, image, current_point, view):
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

    def show_image(self, im_params, current_point):
        if not current_point:
            current_point = Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2),
                                        int(self.images[0].data.shape[2] / 2)])
        for i, image in enumerate(self.images):
            data_to_display = self.set_data_to_display(image, current_point, self.view)
            (my_cmap, my_interpolation, my_alpha) = self.set_image_parameters(im_params, i, mpl.cm)
            my_cmap.set_under('b', alpha=0)
            self.figs.append(self.axes.imshow(data_to_display, aspect=self.aspect_ratio, alpha=my_alpha))
            self.figs[-1].set_cmap(my_cmap)
            self.figs[-1].set_interpolation(my_interpolation)
            if (self.list_intensites):
                self.figs[-1].set_clim(self.list_intensites[0], self.list_intensites[1])

    def set_image_parameters(self, im_params, i, cm):
        if str(i) in im_params.images_parameters:
            return (
                copy(cm.get_cmap(im_params.images_parameters[str(i)].cmap)), im_params.images_parameters[str(i)].interp,
                float(im_params.images_parameters[str(i)].alpha))
        else:
            return (cm.get_cmap('gray'), 'nearest', 1.0)

    def remove_axis_number(self):
        self.axes.set_axis_bgcolor('black')
        self.axes.set_xticks([])
        self.axes.set_yticks([])

    def change_intensity(self, event):
        """
        Functions that deal with the change of intensity in the image.
        """

        def calc_min_max_intensities(x, y):
            xlim, ylim = self.axes.get_xlim(), self.axes.get_ylim()
            mean_intensity_factor = (x - xlim[0]) / float(xlim[1] - xlim[0])
            std_intensity_factor = (y - ylim[1]) / float(ylim[0] - ylim[1])
            mean_factor = self.mean_intensity[0] - (mean_intensity_factor - 0.5) * self.mean_intensity[0] * 3.0
            std_factor = self.std_intensity[0] + (std_intensity_factor - 0.5) * self.std_intensity[0] * 2.0
            self.list_intensites = [mean_factor - std_factor, mean_factor + std_factor]
            return (mean_factor - std_factor, mean_factor + std_factor)

        def check_time_last_update(last_update):
            if time() - last_update < 1.0 / 15.0:  # 10 Hz:
                return False
            else:
                return True

        if (check_time_last_update(self.last_update)):
            self.last_update = time()
            self.figs[-1].set_clim(calc_min_max_intensities(event.xdata, event.ydata))
            self.refresh()

    def setup_intensity(self):
        """
        Defines the default intensity
        """
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
        self.canvas.mpl_connect('button_release_event', self.on_event_release)
        self.canvas.mpl_connect('scroll_event', self.on_event_scroll)
        self.canvas.mpl_connect('motion_notify_event', self.on_event_motion)

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
            scale_factor = calc_scale_factor(event.button)
            self.update_xy_lim(x_center=event.xdata, y_center=event.ydata,
                               x_scale_factor=scale_factor, y_scale_factor=scale_factor,
                               zoom=True)

    def get_event_coordinates(self, event, label=1):
        """
        Parameters
        ----------
        event           event we must get the coordinates of
        label           label of the dot

        Returns
        -------
        point           Coordinate( [ event.x , event.y , event.z , label ] )

        """
        point = None
        try:
            if self.view == 'ax':
                point = Coordinate([self.current_position.x,
                                    int(round(event.ydata)),
                                    int(round(event.xdata)), self.dic_translate_label[str(label)]])
            elif self.view == 'cor':
                point = Coordinate([int(round(event.ydata)),
                                    self.current_position.y,
                                    int(round(event.xdata)), self.dic_translate_label[str(label)]])
            elif self.view == 'sag':
                point = Coordinate([int(round(event.ydata)),
                                    int(round(event.xdata)),
                                    self.current_position.z, self.dic_translate_label[str(label)]])
        except TypeError:
            self.header.update_text('warning_selected_point_not_in_image')
        return point

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

    def is_point_in_image(self, target_point):
        return 0 <= target_point.x < self.image_dim[0] and 0 <= target_point.y < self.image_dim[
            1] and 0 <= target_point.z < self.image_dim[2]
