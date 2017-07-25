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
        self.dict_translate_label = self.define_translate_dict()

        self.remove_axis_number()
        self.connect_mpl_events()
        self.setup_intensity()

    def define_translate_dict(self):
        """
        Defines dictionnary to translate the software labels which range is [1;27] into anatomical labels which range is:
        {50;49} U {1} U [3,26]

        Returns
        -------
        dic        dictionnary
        """
        dict = {'1': 50,
               '2': 49,
               '3': 1,
               '4': 3, }
        for ii in range(5, 30):             # does not matter if the dictionnary is a bit too long. The number of possible labels is still 27.
            dict[str(ii)] = ii - 1
        return dict

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
            if direction == 'down':  # deal with zoom in
                return 1 / base_scale
            elif direction == 'up':  # deal with zoom out
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
                                    int(round(event.xdata)), self.dict_translate_label[str(label)]])
            elif self.view == 'cor':
                point = Coordinate([int(round(event.ydata)),
                                    self.current_position.y,
                                    int(round(event.xdata)), self.dict_translate_label[str(label)]])
            elif self.view == 'sag':
                point = Coordinate([int(round(event.ydata)),
                                    int(round(event.xdata)),
                                    self.current_position.z, self.dict_translate_label[str(label)]])
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


class ImagePlotMainPropseg(ImagePlot):
    """
    Inherites ImagePlot
    Class used for displaying the main (right) picture in Propseg Viewer
    Defines the action on mouse events, draw dots, and manages the list of results : list_points.
    """
    def __init__(self, ax, images, viewer, canvas, view, line_direction='hv', im_params=None, secondary_plot=None,
                 header=None, number_of_points=0):
        super(ImagePlotMainPropseg, self).__init__(ax, images, viewer, canvas, view, line_direction, im_params, header)
        self.secondary_plot = secondary_plot
        self.plot_points, = self.axes.plot([], [], '.r', markersize=10)
        self.show_image(self.im_params, current_point=None)
        self.number_of_points = number_of_points
        self.calculate_list_slices()
        self.update_slice(Coordinate([self.list_slices[0], self.current_position.y, self.current_position.z]))
        self.bool_is_mode_auto = True

    def update_slice(self, new_position):
        self.current_position = new_position
        if (self.view == 'ax'):
            self.figs[-1].set_data(self.images[0].data[self.current_position.x, :, :])
        elif (self.view == 'cor'):
            self.figs[-1].set_data(self.images[0].data[:, self.current_position.y, :])
        elif (self.view == 'sag'):
            self.figs[-1].set_data(self.images[0].data[:, :, self.current_position.z])
        self.figs[-1].figure.canvas.draw()

    def add_point_to_list_points(self, current_point):
        """
        Manages the adding of a point to self.list_points :
            - Auto way : checks if there is a next point
            - Manual way : remplaces the dot that has already be made on the slice.

        Parameters
        ----------
        current_point Coordinate
        """

        def add_point_auto(self):
            if len(self.list_points) < self.number_of_points:
                self.list_points.append(current_point)
                if len(self.list_points) == self.number_of_points:
                    self.header.update_text('ready_to_save_and_quit')
                else:
                    self.header.update_text('update', len(self.list_points), self.number_of_points)
            else:
                self.header.update_text('warning_all_points_done_already')

        def add_point_custom(self):
            bool_remplaced = False
            for ipoint in self.list_points:
                if ipoint.x == current_point.x:
                    self.list_points.remove(ipoint)
                    self.list_points.append(current_point)
                    bool_remplaced = True
            if not bool_remplaced:
                self.list_points.append(current_point)
            self.header.update_text('update', len(self.list_points), self.number_of_points)

        if self.bool_is_mode_auto:
            add_point_auto(self)
        else:
            add_point_custom(self)

    def on_event_motion(self, event):
        if event.button == 3 and event.inaxes == self.axes:  # right click
            if self.get_event_coordinates(event):
                self.change_intensity(event)
                self.change_intensity_on_secondary_plot(event)

    def on_event_release(self, event):
        if self.get_event_coordinates(event):
            if event.button == 1:  # left click
                self.add_point_to_list_points(self.get_event_coordinates(event))
                if self.bool_is_mode_auto:
                    self.jump_to_new_slice()
                self.draw_dots()
            elif event.button == 3:  # right click
                self.change_intensity(event)
                self.change_intensity_on_secondary_plot(event)

    def jump_to_new_slice(self):
        if len(self.list_points) < self.number_of_points:
            self.update_slice(
                Coordinate([self.list_slices[len(self.list_points)], self.current_position.y, self.current_position.z]))
            self.secondary_plot.current_position = Coordinate(
                [self.list_slices[len(self.list_points)], self.current_position.y, self.current_position.z])
            self.secondary_plot.draw_lines('v')

    def change_intensity_on_secondary_plot(self, event):
        """
        Updates the intensity on the secondary plot according to the intensity of the first.
        """
        if self.secondary_plot:
            self.secondary_plot.change_intensity(event)

    def refresh(self):
        self.figs[-1].figure.canvas.draw()

    def draw_dots(self):
        """
        Draw dots on selected points on the main picture

        Warning : the picture in main plot image is the projection of a 3D image.
        That is why, we have to carefully determinate which coordinates are x or y, to properly draw the dot.
        """
        def select_right_dimensions(ipoint, view):
            if view == 'ax':
                return ipoints.z, ipoints.y
            elif view == 'cor':
                return ipoints.x, ipoints.z
            elif view == 'sag':
                return ipoints.y, ipoints.x

        def select_right_position_dim(current_position, view):
            if view == 'ax':
                return current_position.x
            elif view == 'cor':
                return current_position.y
            elif view == 'sag':
                return current_position.z

        x_data, y_data = [], []
        for ipoints in self.list_points:
            if select_right_position_dim(ipoints, self.view) == select_right_position_dim(self.current_position,
                                                                                          self.view):
                x, y = select_right_dimensions(ipoints, self.view)
                x_data.append(x)
                y_data.append(y)
        self.plot_points.set_xdata(x_data)
        self.plot_points.set_ydata(y_data)
        self.refresh()

    def calculate_list_slices(self):
        self.list_slices = []
        increment = int(self.image_dim[0] / (self.number_of_points - 1))
        for ii in range(0, self.number_of_points - 1):
            self.list_slices.append(ii * increment)
        self.list_slices.append(self.image_dim[0] - 1)

    def switch_mode_seg(self):
        self.bool_is_mode_auto = not self.bool_is_mode_auto
        self.reset_data()
        self.header.update_text('mode_switched')

    def reset_data(self):
        """
        Resets all the data when user switches mode, ie Manual Mode => Auto Mode or Auto Mode => Manual Mode.
        """
        self.list_points = []
        if self.bool_is_mode_auto:
            self.number_of_points = 7
        else:
            self.number_of_points = -1
        self.current_position.x = 0
        self.update_slice(self.current_position)
        self.draw_dots()
        self.secondary_plot.current_position = self.current_position
        self.secondary_plot.draw_lines('v')


class ImagePlotSecondPropseg(ImagePlot):
    """
    Inherites ImagePlot
    Class used for displaying the secondary (left) picture in Propseg Viewer
    Defines the action on mouse events, draw lines and update the slice on the main picture.
    """
    def __init__(self, ax, images, viewer, canvas, main_single_plot, view, line_direction='hv', im_params=None,
                 header=None):
        super(ImagePlotSecondPropseg, self).__init__(ax, images, viewer, canvas, view, line_direction, im_params, header)
        self.main_plot = main_single_plot
        self.current_position = self.main_plot.current_position
        self.list_previous_lines = []

        self.show_image(self.im_params, current_point=None)
        self.current_line = self.calc_line('v',
                                           self.current_position)  # add_line is used in stead of draw_line because in draw_line we also remove the previous line.
        self.axes.add_line(self.current_line)
        self.refresh()

    def calc_line(self, line_direction, line_position, line_color='white'):
        """
        Creates a line according to coordinate line_position and direnction line_direction.

        Parameters
        ----------
        line_direction      {'h','v'}
        line_position       Coordinate
        line_color          {'white','red'}

        Returns
        -------
        line                matplotlit.line2D

        """
        def calc_dict_line_coor(current_position, view):
            if view == 'ax':
                return {'v': [[current_position.y, current_position.y], [-10000, 10000]],
                        'h': [[-10000, 10000], [current_position.z, current_position.z]]}
            elif view == 'cor':
                return {'v': [[current_position.x, current_position.x], [-10000, 10000]],
                        'h': [[-10000, 10000], [current_position.z, current_position.z]]}
            elif view == 'sag':
                return {'v': [[current_position.x, current_position.x], [-10000, 10000]],
                        'h': [[-10000, 10000], [current_position.y, current_position.y]]}

        dict_line_coor = calc_dict_line_coor(line_position, self.view)
        line = Line2D(dict_line_coor[line_direction][1], dict_line_coor[line_direction][0], color=line_color)
        return line

    def draw_current_line(self, line_direction):
        self.current_line.remove()
        self.current_line = self.calc_line(line_direction, self.current_position)
        self.axes.add_line(self.current_line)

    def draw_previous_lines(self, line_direction):
        for iline in self.list_previous_lines:
            iline.remove()
        self.list_previous_lines = []
        for ipoint in self.main_plot.list_points:
            self.list_previous_lines.append(self.calc_line(line_direction, ipoint, line_color='red'))
            self.axes.add_line(self.list_previous_lines[-1])

    def draw_lines(self, line_direction):
        """
        Global function that manages the drawing of all the lines on the secondary image.
        """
        self.draw_current_line(line_direction)
        self.draw_previous_lines(line_direction)
        self.refresh()

    def refresh(self):
        self.show_image(self.im_params, self.current_position)
        self.figs[0].figure.canvas.draw()

    def on_event_motion(self, event):
        if event.button == 1 and event.inaxes == self.axes:  # left click
            if self.get_event_coordinates(event):
                self.change_main_slice(event)
        elif event.button == 3 and event.inaxes == self.axes:  # right click
            if self.get_event_coordinates(event):
                self.change_intensity(event)

    def on_event_release(self, event):
        if self.get_event_coordinates(event):
            if event.button == 1:  # left click
                if not self.main_plot.bool_is_mode_auto:
                    self.change_main_slice(event)
                else:
                    self.main_plot.jump_to_new_slice()
            elif event.button == 3:  # right click
                self.change_intensity(event)

    def change_main_slice(self, event):
        '''
        When the user chosees a new slice, this function :
        - updates the variable self.current_position in ImagePlotSecond
        - updates the slice to display in ImagePlotMain.
        '''
        self.current_position = self.get_event_coordinates(event)
        self.draw_lines('v')

        self.main_plot.show_image(self.im_params, self.current_position)
        self.main_plot.update_slice(self.current_position)
        self.main_plot.refresh()
        self.main_plot.draw_dots()


class ImagePlotMainLabelVertebrae(ImagePlot):
    """
    Inherites ImagePlot
    Class used for displaying the main (right) picture in Label Vertebrae Viewer
    Defines the action on mouse events, draw dots, and manages the list of results : list_points.
    """
    def __init__(self, ax, images, viewer, canvas, view, line_direction='hv', im_params=None, secondary_plot=None,
                 header=None, number_of_points=0):
        super(ImagePlotMainLabelVertebrae, self).__init__(ax, images, viewer, canvas, view, line_direction, im_params,
                                                           header)
        self.secondary_plot = secondary_plot
        self.plot_points, = self.axes.plot([], [], '.r', markersize=10)
        self.show_image(self.im_params, current_point=None)
        self.number_of_points = number_of_points
        self.current_label = 3

    def add_point_to_list_points(self, current_point):
        def add_point_custom(self):
            bool_remplaced = False
            for ipoint in self.list_points:
                if ipoint.z == current_point.z:
                    self.list_points.remove(ipoint)
                    self.list_points.append(current_point)
                    bool_remplaced = True
            if not bool_remplaced:
                self.list_points.append(current_point)
            self.header.update_text('update', len(self.list_points), self.number_of_points)

        add_point_custom(self)

    def on_event_motion(self, event):
        if event.button == 3 and event.inaxes == self.axes:  # right click
            if self.get_event_coordinates(event):
                self.change_intensity(event)

    def on_event_release(self, event):
        if self.get_event_coordinates(event):
            if event.button == 1:  # left click
                self.add_point_to_list_points(self.get_event_coordinates(event, self.current_label))
                self.draw_dots()
            elif event.button == 3:  # right click
                self.change_intensity(event)

    def refresh(self):
        self.figs[-1].figure.canvas.draw()

    def draw_dots(self):
        """
        Draw dots on selected points on the main picture

        Warning : the picture in main plot image is the projection of a 3D image.
        That is why, we have to carefully determinate which coordinates are x or y, to properly draw the dot.
        """
        def select_right_dimensions(ipoint, view):
            """
            Selects coordinates to diplay the dot right.

            Returns
            x (int) and y (int) to use as input to draw dot.

            """
            if view == 'ax':
                return ipoints.z, ipoints.y
            elif view == 'cor':
                return ipoints.x, ipoints.z
            elif view == 'sag':
                return ipoints.y, ipoints.x

        def select_right_position_dim(current_position, view):
            if view == 'ax':
                return current_position.x
            elif view == 'cor':
                return current_position.y
            elif view == 'sag':
                return current_position.z

        x_data, y_data = [], []
        for ipoints in self.list_points:
            if select_right_position_dim(ipoints, self.view) == select_right_position_dim(self.current_position,
                                                                                          self.view):
                x, y = select_right_dimensions(ipoints, self.view)
                x_data.append(x)
                y_data.append(y)
        self.plot_points.set_xdata(x_data)
        self.plot_points.set_ydata(y_data)
        self.refresh()


class ImagePlotMainGroundTruth(ImagePlot):
    """
    Inherites ImagePlot
    Class used for displaying the main (right) picture in Propseg Viewer
    Defines the action on mouse events, draw dots, and manages the list of results : list_points.
    """
    def __init__(self, ax, images, viewer, canvas, view, line_direction='hv', im_params=None, secondary_plot=None,
                 header=None, number_of_points=0,first_label=1):
        super(ImagePlotMainGroundTruth, self).__init__(ax, images, viewer, canvas, view, line_direction, im_params, header)
        self.secondary_plot = secondary_plot
        self.first_label=self.translate_labels_num_into_list_point_length(str(first_label))
        self.plot_points, = self.axes.plot([], [], '.r', markersize=10)
        self.show_image(self.im_params, current_point=None)
        self.number_of_points = number_of_points
        self.calculate_list_slices()
        self.update_slice(Coordinate([self.list_slices[0], self.current_position.y, self.current_position.z]))
        self.fill_first_labels()
        self.header.update_text('update',str(len(self.calc_list_points_on_slice())+1))


    def update_slice(self, new_position):
        self.current_position = new_position
        if (self.view == 'ax'):
            self.figs[-1].set_data(self.images[0].data[self.current_position.x, :, :])
        elif (self.view == 'cor'):
            self.figs[-1].set_data(self.images[0].data[:, self.current_position.y, :])
        elif (self.view == 'sag'):
            self.figs[-1].set_data(self.images[0].data[:, :, self.current_position.z])
        self.figs[-1].figure.canvas.draw()

    def add_point_to_list_points(self, current_point):
        """
        Manages the adding of a point to self.list_points :
            - Auto way : checks if there is a next point
            - Manual way : remplaces the dot that has already be made on the slice.

        Parameters
        ----------
        current_point Coordinate
        """
        list_points_on_slice=self.calc_list_points_on_slice()
        if len(list_points_on_slice) < self.number_of_points:
            self.list_points.append(Coordinate([current_point.x,
                                                current_point.y,
                                                current_point.z,
                                                self.dict_translate_label[str(len(self.list_points)+1)]
                                                #self.dict_translate_label[str(len(self.calc_list_points_on_slice())+1)]
                                                ]))
            list_points_on_slice = self.calc_list_points_on_slice()
            if len(list_points_on_slice) == self.number_of_points:
                self.header.update_text('ready_to_save_and_quit')
            else:
                self.header.update_text('update', str(len(self.list_points)+1))
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
            elif event.button == 3:  # right click
                self.change_intensity(event)
                self.change_intensity_on_secondary_plot(event)

    def change_intensity_on_secondary_plot(self, event):
        """
        Updates the intensity on the secondary plot according to the intensity of the first.
        """
        if self.secondary_plot:
            self.secondary_plot.change_intensity(event)

    def refresh(self):
        self.figs[-1].figure.canvas.draw()

    def calc_list_points_on_slice(self):
        def select_right_position_dim(current_position, view):
            if view == 'ax':
                return current_position.x
            elif view == 'cor':
                return current_position.y
            elif view == 'sag':
                return current_position.z

        list_points_on_slice=[]
        for ipoints in self.list_points:
            if select_right_position_dim(ipoints, self.view) == select_right_position_dim(self.current_position,self.view):
                list_points_on_slice.append(ipoints)
        return list_points_on_slice

    def draw_dots(self):
        """
        Draw dots on selected points on the main picture

        Warning : the picture in main plot image is the projection of a 3D image.
        That is why, we have to carefully determinate which coordinates are x or y, to properly draw the dot.
        """
        def select_right_dimensions(ipoint, view):
            if view == 'ax':
                return ipoints.z, ipoints.y
            elif view == 'cor':
                return ipoints.x, ipoints.z
            elif view == 'sag':
                return ipoints.y, ipoints.x

        x_data, y_data = [], []
        list_points_on_slice=self.calc_list_points_on_slice()
        for ipoints in list_points_on_slice:
            x, y = select_right_dimensions(ipoints, self.view)
            if x!=-1 and y!=-1:
                x_data.append(x)
                y_data.append(y)
        self.plot_points.set_xdata(x_data)
        self.plot_points.set_ydata(y_data)
        self.refresh()

    def calculate_list_slices(self):
        self.list_slices = []
        increment = int(self.image_dim[0] / (self.number_of_points - 1))
        for ii in range(0, self.number_of_points - 1):
            self.list_slices.append(ii * increment)
        self.list_slices.append(self.image_dim[0] - 1)

    def switch_mode_seg(self):
        self.bool_is_mode_auto = not self.bool_is_mode_auto
        self.reset_data()
        self.header.update_text('mode_switched')

    def fill_first_labels(self):
        if not self.calc_list_points_on_slice():
            for ilabels in range (1,self.first_label):
                self.list_points.append(Coordinate([-1,
                                                    -1,
                                                    self.current_position.z,
                                                    self.dict_translate_label[str(ilabels)]
                                                    ]))

    def check_if_selected_points_on_slice(self):
        bool_selected_points=False
        previous_slice=self.calc_list_points_on_slice()
        for ipoints in previous_slice:
            if ipoints.x!=-1:
                bool_selected_points=True
                return bool_selected_points
        return bool_selected_points

    def delete_all_points_on_slice(self):
        previous_slice=self.calc_list_points_on_slice()
        for ipoints in previous_slice:
            self.list_points.remove(ipoints)

    def translate_labels_num_into_list_point_length(self,value_to_translate):
        dict={'50':1,
             '49':2,
             '1':3,
             '3':4,}
        for ii in range (4,27):
            dict[str(ii)]=ii+1
        return dict[value_to_translate]


class ImagePlotSecondGroundTruth(ImagePlot):
    """
    Inherites ImagePlot
    Class used for displaying the secondary (left) picture in Propseg Viewer
    Defines the action on mouse events, draw lines and update the slice on the main picture.
    """
    def __init__(self, ax, images, viewer, canvas, main_single_plot, view, line_direction='hv', im_params=None,
                 header=None):
        super(ImagePlotSecondGroundTruth, self).__init__(ax, images, viewer, canvas, view, line_direction, im_params, header)
        self.main_plot = main_single_plot
        self.current_position = self.main_plot.current_position
        self.list_previous_lines = []

        self.show_image(self.im_params, current_point=None)
        self.current_line = self.calc_line(self.current_position)  # add_line is used in stead of draw_line because in draw_line we also remove the previous line.
        self.axes.add_line(self.current_line)
        self.refresh()

    def calc_line(self, line_position, line_color='white'):
        """
        Creates a line according to coordinate line_position and direnction line_direction.

        Parameters
        ----------
        line_direction      {'h','v'}
        line_position       Coordinate
        line_color          {'white','red'}

        Returns
        -------
        line                matplotlit.line2D

        """
        def calc_dict_line_coor(current_position, view):
            if view == 'ax':
                return {'h': [[current_position.y, current_position.y], [-10000, 10000]],
                        'v': [[-10000, 10000], [current_position.z, current_position.z]]}
            elif view == 'cor':
                return {'h': [[current_position.x, current_position.x], [-10000, 10000]],
                        'v': [[-10000, 10000], [current_position.z, current_position.z]]}
            elif view == 'sag':
                return {'h': [[current_position.x, current_position.x], [-10000, 10000]],
                        'v': [[-10000, 10000], [current_position.y, current_position.y]]}

        dict_line_coor = calc_dict_line_coor(line_position, self.view)
        line = Line2D(dict_line_coor[self.line_direction][1], dict_line_coor[self.line_direction][0], color=line_color)
        return line

    def draw_current_line(self):
        self.current_line.remove()
        self.current_line = self.calc_line(self.current_position)
        self.axes.add_line(self.current_line)

    def draw_previous_lines(self):
        for iline in self.list_previous_lines:
            iline.remove()
        self.list_previous_lines = []
        for ipoint in self.main_plot.list_points:
            self.list_previous_lines.append(self.calc_line(ipoint, line_color='red'))
            self.axes.add_line(self.list_previous_lines[-1])

    def draw_lines(self):
        """
        Global function that manages the drawing of all the lines on the secondary image.
        """
        self.draw_previous_lines()
        self.draw_current_line()
        self.refresh()

    def refresh(self):
        self.show_image(self.im_params, self.current_position)
        self.figs[0].figure.canvas.draw()

    def on_event_motion(self, event):
        if event.button == 1 and event.inaxes == self.axes:  # left click
            if self.get_event_coordinates(event):
                self.change_main_slice(event,bool_fill_first_labels=False)
        elif event.button == 3 and event.inaxes == self.axes:  # right click
            if self.get_event_coordinates(event):
                self.change_intensity(event)

    def on_event_release(self, event):
        if self.get_event_coordinates(event):
            if event.button == 1:  # left click
                self.change_main_slice(event,bool_fill_first_labels=True)
            elif event.button == 3:  # right click
                self.change_intensity(event)

    def change_main_slice(self, event,bool_fill_first_labels):
        '''
        When the user chosees a new slice, this function :
        - updates the variable self.current_position in ImagePlotSecond
        - updates the slice to display in ImagePlotMain.
        '''

        if not self.main_plot.check_if_selected_points_on_slice():
            self.main_plot.delete_all_points_on_slice()

        self.current_position = self.get_event_coordinates(event)
        self.draw_lines()

        self.main_plot.show_image(self.im_params, self.current_position)
        self.main_plot.update_slice(self.current_position)
        self.main_plot.refresh()
        self.main_plot.draw_dots()
        if bool_fill_first_labels:
            pass
            #self.main_plot.fill_first_labels()
        if self.main_plot.calc_list_points_on_slice():
            self.header.update_text('update',str(len(self.main_plot.calc_list_points_on_slice())+1))


class ImagePlotTest(ImagePlotMainGroundTruth):
    """
    def calc_mean_slices(self):
        data=self.images[0].data
        dataRacc=data[:,:,self.current_point.z-(self.number_of_slices_to_mean-1)/2:self.current_point.z+(self.number_of_slices_to_mean-1)/2+1]
        imMean=np.empty([data.shape[0],data.shape[1]])
        for ii in range (0,data.shape[0]):
            for jj in range (0,data.shape[1]):
                imMean[ii,jj]=np.mean(dataRacc[ii,jj,:])
        return imMean
    """

    def __init__(self, ax, images, viewer, canvas, view, line_direction='hv', im_params=None, secondary_plot=None,
                 header=None, number_of_points=0, first_label=1):
        super(ImagePlotTest, self).__init__(ax, images, viewer, canvas, view, line_direction, im_params, secondary_plot,
                 header, number_of_points, first_label)
        self.nb_slice_to_average=3

    def show_image_mean(self,nb_slice_to_average=3):
        def calc_mean_slices(nb_slice_to_average):
            import numpy as np
            data = self.images[0].data
            dataRacc = data[:, :, self.current_position.z - (nb_slice_to_average - 1) / 2:self.current_position.z + (nb_slice_to_average - 1) / 2 + 1]
            imMean = np.empty([data.shape[0], data.shape[1]])
            for ii in range(0, data.shape[0]):
                for jj in range(0, data.shape[1]):
                    imMean[ii, jj] = np.mean(dataRacc[ii, jj, :])
            return imMean

        (my_cmap, my_interpolation, my_alpha) = (cm.get_cmap('gray'), 'nearest', 1.0)
        image_averaged=calc_mean_slices(nb_slice_to_average)
        self.figs[-1]=self.axes.imshow(image_averaged, aspect=self.aspect_ratio, alpha=my_alpha)
        self.figs[-1].set_cmap(my_cmap)
        self.draw_dots()
        return image_averaged


class HeaderCore(object):
    """
    Core Class for Header
    Defines Layouts and some basic messages.
    """
    def __init__(self):
        self.define_layout_header()
        self.add_lb_status()
        self.add_lb_warning()
        self.dict_message_labels = self.define_dict_message_labels()

    def define_dict_message_labels(self):
        dict = {'1': 'Please click on anterior base \n'
                    'of pontomedullary junction (label=50) \n',
               '2': 'Please click on pontomedullary groove \n'
                    ' (label=49) \n',

               '3': 'Please click on top of C1 vertebrae \n'
                    '(label=1) \n',
               '4': 'Please click on posterior edge of  \n'
                    'C2/C3 intervertebral disk (label=3) \n',
               '5': 'Please click on posterior edge of  \n'
                    'C3/C4 intervertebral disk (label=4) \n',
               '6': 'Please click on posterior edge of  \n'
                    'C4/C5 intervertebral disk (label=5) \n',
               '7': 'Please click on posterior edge of  \n'
                    'C5/C6 intervertebral disk (label=6) \n',
               '8': 'Please click on posterior edge of  \n'
                    'C6/C7 intervertebral disk (label=7) \n',
               '9': 'Please click on posterior edge of  \n'
                    'C7/T1 intervertebral disk (label=8) \n',

               '10': 'Please click on posterior edge of  \n'
                     'T1/T2 intervertebral disk (label=9) \n',
               '11': 'Please click on posterior edge of  \n'
                     'T2/T3 intervertebral disk (label=10) \n',
               '12': 'Please click on posterior edge of  \n'
                     'T3/T4 intervertebral disk (label=11) \n',
               '13': 'Please click on posterior edge of  \n'
                     'T4/T5 intervertebral disk (label=12) \n',
               '14': 'Please click on posterior edge of  \n'
                     'T5/T6 intervertebral disk (label=13) \n',
               '15': 'Please click on posterior edge of  \n'
                     'T6/T7 intervertebral disk (label=14) \n',
               '16': 'Please click on posterior edge of  \n'
                     'T7/T8 intervertebral disk (label=15) \n',
               '17': 'Please click on posterior edge of  \n'
                     'T8/T9 intervertebral disk (label=16) \n',
               '18': 'Please click on posterior edge of  \n'
                     'T9/T10 intervertebral disk (label=17) \n',
               '19': 'Please click on posterior edge of  \n'
                     'T10/T11 intervertebral disk (label=18) \n',
               '20': 'Please click on posterior edge of  \n'
                     'T11/T12 intervertebral disk (label=19) \n',
               '21': 'Please click on posterior edge of  \n'
                     'T12/L1 intervertebral disk (label=20) \n',

               '22': 'Please click on posterior edge of  \n'
                     'L1/L2 intervertebral disk (label=21) \n',
               '23': 'Please click on posterior edge of  \n'
                     'L2/L3 intervertebral disk (label=22) \n',
               '24': 'Please click on posterior edge of  \n'
                     'L3/L4 intervertebral disk (label=23) \n',
               '25': 'Please click on posterior edge of  \n'
                     'L4/S1 intervertebral disk (label=24) \n',

               '26': 'Please click on posterior edge of  \n'
                     'S1/S2 intervertebral disk (label=25) \n',
               '27': 'Please click on posterior edge of  \n'
                     'S2/S3 intervertebral disk (label=26) \n',

               }
        return dict

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
        self.layout_header.setContentsMargins(0, 20, 0, 0)

    def update_title_text_general(self, key, nbpt=-1, nbfin=-1):
        if (key == 'ready_to_save_and_quit'):
            self.lb_status.setText('You can save and quit')
            self.lb_status.setStyleSheet("color:green")
        elif (key == 'warning_all_points_done_already'):
            self.lb_warning.setText('You have placed all needed points. \n'
                                    'If you made a mistake, you may use \'undo\'.')
            self.lb_warning.setStyleSheet("color:red")
        elif (key == 'warning_undo_beyond_first_point'):
            self.lb_warning.setText('Please place your first dot.')
            self.lb_warning.setStyleSheet("color:red")
        elif (key == 'warning_selected_point_not_in_image'):
            self.lb_warning.setText('The point you selected in not in the image. Please try again.')
            self.lb_warning.setStyleSheet("color:red")
        elif (key == 'update'):
            if nbfin == -1:
                self.lb_status.setText('You have made ' + str(nbpt) + ' points.')
                self.lb_status.setStyleSheet("color:black")
            else:
                self.lb_status.setText('You have made ' + str(nbpt) + ' points out of ' + str(nbfin) + '.')
                self.lb_status.setStyleSheet("color:black")

        else:
            self.lb_warning.setText(key + ' : Unknown key')
            self.lb_warning.setStyleSheet("color:red")


class HeaderPropseg(HeaderCore):
    """
    Inherites HeaderCore
    Class that defines header in Propseg Viewer
    Defines specific messages to display.
    """
    def update_text(self, key, nbpt=-1, nbfin=-1):
        self.lb_warning.setText('\n')
        if (key == 'welcome'):
            self.lb_status.setText('Please click in the the center of the center line. \n'
                                   'If it is invisible, you may skip it.')
            self.lb_status.setStyleSheet("color:black")
        elif (key == 'warning_skip_not_defined'):
            self.lb_warning.setText('This option is not used in Manual Mode. \n')
            self.lb_warning.setStyleSheet("color:red")
        elif (key == 'mode_switched'):
            self.lb_status.setText('You have switched on an other segmentation mode. \n'
                                   'All previous data have been erased.')
            self.lb_status.setStyleSheet("color:black")
        else:
            self.update_title_text_general(key, nbpt, nbfin)


class HeaderLabelVertebrae(HeaderCore):
    """
    Inherites HeaderCore
    Class that defines header in Label Vertebrae Viewer
    Defines specific messages to display.
    """
    def update_text(self, key, nbpt=-1, nbfin=-1):
        self.lb_warning.setText('\n')
        if (key == 'welcome'):
            self.lb_status.setText(self.dict_message_labels[nbpt])
            self.lb_status.setStyleSheet("color:black")
        elif (key == 'update'):
            if nbpt:
                self.update_text('ready_to_save_and_quit')
            else:
                self.update_text('welcome')
        elif (key == 'warning_cannot_change_the_label'):
            self.lb_warning.setText('You cannot change the label once you have placed a point. \n'
                                    'Please \'undo\' first.')
            self.lb_warning.setStyleSheet("color:red")
        else:
            self.update_title_text_general(key, nbpt, nbfin)


class HeaderGroundTruth(HeaderCore):
    """
    Inherites HeaderCore
    Class that defines header in Propseg Viewer
    Defines specific messages to display.
    """
    def update_text(self, key, nbpt=-1, nbfin=-1):
        self.lb_warning.setText('\n')
        if (key == 'welcome'):
            self.lb_status.setText(self.dict_message_labels[nbpt])
            self.lb_status.setStyleSheet("color:black")
        elif(key=='update'):
            self.lb_status.setText(self.dict_message_labels[nbpt])
            self.lb_status.setStyleSheet("color:black")
        else:
            self.update_title_text_general(key, nbpt, nbfin)


class MainPannelCore(object):
    """
    Class core that defines the layout of the Main pannel.
    Defines layout and manages their merging.
    Provides an example of how to call Main Image Plot and Secondary Image Plot.
    """
    def __init__(self,
                 images,
                 im_params, window, header):
        self.header = header
        self.window = window
        self.layout_global = QtGui.QVBoxLayout()
        self.layout_option_settings = QtGui.QHBoxLayout()
        self.layout_central = QtGui.QHBoxLayout()
        self.layout_central.setDirection(1)
        self.images = images
        self.im_params = im_params
        self.current_position = Coordinate(
            [int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2),
             int(self.images[0].data.shape[2] / 2)])
        nx, ny, nz, nt, px, py, pz, pt = self.images[0].dim
        self.im_spacing = [px, py, pz]
        self.aspect_ratio = [float(self.im_spacing[1]) / float(self.im_spacing[2]),
                             float(self.im_spacing[0]) / float(self.im_spacing[2]),
                             float(self.im_spacing[0]) / float(self.im_spacing[1])]
        self.number_of_points = -1

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
        self.main_plot = ImagePlotMainPropseg(axis, self.images, self, view='ax', line_direction='', im_params=self.im_params,
                                        canvas=self.canvas_main, header=self.header, number_of_points=7)

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
        self.second_plot = ImagePlotSecondPropseg(axis, self.images, self, view='sag', line_direction='',
                                            im_params=self.im_params, canvas=self.canvas_second,
                                            main_single_plot=self.main_plot, header=self.header)
        self.main_plot.secondary_plot = self.second_plot

    def merge_layouts(self):
        self.layout_global.addLayout(self.layout_option_settings)
        self.layout_global.addLayout(self.layout_central)


class MainPannelPropseg(MainPannelCore):
    """
    Inherites MainPannelCore
    Class that defines specific main image plot and secondary image plot for Propseg Viewer.
    """
    def __init__(self, images, im_params, window, header):
        super(MainPannelPropseg, self).__init__(images, im_params, window, header)

        self.number_of_points = 12
        self.add_main_view()
        self.add_secondary_view()
        # self.add_controller_pannel()
        self.add_option_settings()
        self.merge_layouts()

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
        self.main_plot = ImagePlotMainPropseg(axis, self.images, self, view='ax', line_direction='', im_params=self.im_params,
                                        canvas=self.canvas_main, header=self.header,
                                        number_of_points=self.number_of_points)

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
        self.second_plot = ImagePlotSecondPropseg(axis, self.images, self, view='sag', line_direction='',
                                            im_params=self.im_params, canvas=self.canvas_second,
                                            main_single_plot=self.main_plot, header=self.header)
        self.main_plot.secondary_plot = self.second_plot

    def add_option_settings(self):
        self.rb_mode_auto = QtGui.QRadioButton('Mode Auto')
        self.rb_mode_custom = QtGui.QRadioButton('Mode Custom')
        self.rb_mode_custom = QtGui.QRadioButton('Mode Custom')
        self.layout_option_settings.addWidget(self.rb_mode_auto)
        self.layout_option_settings.addWidget(self.rb_mode_custom)
        self.rb_mode_auto.setChecked(True)
        self.rb_mode_auto.clicked.connect(self.main_plot.switch_mode_seg)
        self.rb_mode_custom.clicked.connect(self.main_plot.switch_mode_seg)


class MainPannelLabelVertebrae(MainPannelCore):
    """
    Inherites MainPannelCore
    Class that defines specific main image plot and controller pannel for Label Vertebrae Viewer.
    """
    def __init__(self, images, im_params, window, header,wanted_label):
        super(MainPannelLabelVertebrae, self).__init__(images, im_params, window, header)

        self.number_of_points = 1
        self.add_main_view()
        self.add_controller_pannel(wanted_label=wanted_label)
        self.merge_layouts()
        self.number_of_points = 1

    def add_main_view(self):
        layout_view = QtGui.QVBoxLayout()

        fig = plt.figure()
        self.canvas_main = FigureCanvas(fig)

        layout_view.addWidget(self.canvas_main)
        self.layout_central.addLayout(layout_view, 1)

        if not self.im_params:
            self.im_params = ParamMultiImageVisualization([ParamImageVisualization()])
        gs = mpl.gridspec.GridSpec(1, 1)
        axis = fig.add_subplot(gs[0, 0], axisbg='k')
        self.main_plot = ImagePlotMainLabelVertebrae(axis, self.images, self, view='sag', line_direction='',
                                                      im_params=self.im_params, canvas=self.canvas_main,
                                                      header=self.header, number_of_points=self.number_of_points)

    def add_controller_pannel(self,wanted_label):
        def update_slider_label():
            if not self.main_plot.list_points:
                slider_real_value = slider_maximum - int(slider_maximum * self.slider_label.value() / 100)
                self.header.update_text('welcome', str(slider_real_value))
                self.main_plot.current_label = slider_real_value
                lb_title.setText('Label #' + str(self.main_plot.dict_translate_label[str(slider_real_value)]))
            else:
                self.header.update_text('warning_cannot_change_the_label')
                self.slider_label.setValue(int(100 * (slider_maximum - self.main_plot.current_label) / slider_maximum))

        layout_title_and_controller = QtGui.QVBoxLayout()
        lb_title = QtGui.QLabel('Label Choice')
        lb_title.setAlignment(QtCore.Qt.AlignCenter)
        layout_title_and_controller.addWidget(lb_title)
        layout_controller = QtGui.QHBoxLayout()
        layout_controller.setAlignment(QtCore.Qt.AlignTop)
        layout_controller.setAlignment(QtCore.Qt.AlignCenter)

        slider_maximum = 26
        init_label = slider_maximum - wanted_label
        self.slider_label = QtGui.QSlider()
        self.slider_label.setMaximumHeight(250)
        self.slider_label.setValue(init_label * 100 / slider_maximum)
        update_slider_label()
        self.slider_label.sliderMoved.connect(update_slider_label)
        self.slider_label.sliderPressed.connect(update_slider_label)
        self.slider_label.sliderReleased.connect(update_slider_label)

        layout_controller.addWidget(self.slider_label)
        layout_title_and_controller.addLayout(layout_controller)
        self.layout_central.addLayout(layout_title_and_controller, 1)


class MainPannelGroundTruth(MainPannelCore):
    """
    Inherites MainPannelCore
    Class that defines specific main image plot and secondary image plot for Propseg Viewer.
    """
    def __init__(self, images, im_params, window, header,first_label=1):
        super(MainPannelGroundTruth, self).__init__(images, im_params, window, header)
        self.number_of_points = 27
        self.first_label=first_label
        self.add_main_view()
        self.add_secondary_view()
        self.merge_layouts()

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
        self.main_plot = ImagePlotMainGroundTruth(axis,
                                                  self.images,
                                                  self,
                                                  view='sag',
                                                  line_direction='',
                                                  im_params=self.im_params,
                                                  canvas=self.canvas_main,
                                                  header=self.header,
                                                  number_of_points=self.number_of_points,
                                                  first_label=self.first_label)

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
        self.second_plot = ImagePlotSecondGroundTruth(axis, self.images, self, view='cor', line_direction='v',
                                            im_params=self.im_params, canvas=self.canvas_second,
                                            main_single_plot=self.main_plot, header=self.header)
        self.main_plot.secondary_plot = self.second_plot


class MainPannelTest(MainPannelCore):
    """
    Inherites MainPannelCore
    Class that defines specific main image plot and secondary image plot for Propseg Viewer.
    """
    def __init__(self, images, im_params, window, header,first_label=1,wanted_average=6):
        super(MainPannelTest, self).__init__(images, im_params, window, header)
        self.number_of_points = 27
        self.first_label=first_label
        self.add_main_view()
        self.add_controller_pannel(wanted_average=wanted_average)
        self.merge_layouts()

    def update_slider_average(self):
        def get_odd_number(i):
            if i%2:
                return i
            else:
                return i+1

        real_label_value = get_odd_number(11 * self.slider_average.value() / 100)
        self.lb_average.setText('Averages ' + str(real_label_value) + ' slices')
        self.main_plot.nb_slice_to_average=real_label_value
        self.main_plot.show_image_mean(real_label_value)

    def update_slider_average_title(self):
        def get_odd_number(i):
            if i%2:
                return i
            else:
                return i+1

        real_label_value = get_odd_number(11 * self.slider_average.value() / 100)
        self.lb_average.setText('Averages ' + str(real_label_value) + ' slices')

    def update_slider_slice(self):
        #print(self.main_plot.images[0].data.shape[2]/2)
        #print(int(11 * self.slider_slice.value() / 100) - 6 )
        real_label_value = self.main_plot.images[0].data.shape[2]/2 + ( int(11 * self.slider_slice.value() / 100) - 5 )
        self.lb_slice.setText('Slice #' + str(int(11 * self.slider_slice.value() / 100) - 5))
        self.main_plot.update_slice(Coordinate([self.current_position.x,self.current_position.y,real_label_value]))
        self.update_slider_average()

    def update_slider_slice_title(self):
        real_label_value = self.main_plot.images[0].data.shape[2]/2 + ( int(11 * self.slider_slice.value() / 100) - 5 )
        self.lb_slice.setText('Slice #' + str(int(11 * self.slider_slice.value() / 100) - 5))

    def add_controller_pannel(self,wanted_average):
        def define_lb_title():
            #lb_title = QtGui.QLabel('Averages ' + str(3) + ' slices')
            lb_title = QtGui.QLabel('Control Pannel')
            lb_title.setAlignment(QtCore.Qt.AlignCenter)
            layout_title_and_controller.addWidget(lb_title)

        def define_layout_controller():
            layout_controller = QtGui.QVBoxLayout()
            layout_controller.setAlignment(QtCore.Qt.AlignTop)
            layout_controller.setAlignment(QtCore.Qt.AlignCenter)
            return layout_controller
        def define_lb_average():
            lb = QtGui.QLabel('Averages ' + str(5) + ' slices')
            lb.setAlignment(QtCore.Qt.AlignCenter)
            layout_controller.addWidget(lb)
            return lb
        def define_slider_average(wanted_average=5):
            slider_maximum = 11
            sl = QtGui.QSlider(1)
            sl.setMaximumHeight(250)
            sl.setValue(wanted_average * 100 / slider_maximum)

            sl.sliderReleased.connect(self.update_slider_average)
            sl.sliderMoved.connect(self.update_slider_average_title)


            layout_controller.addWidget(sl)

            return sl

        def define_lb_slice():
            lb = QtGui.QLabel('Slice #'+str(28))
            lb.setAlignment(QtCore.Qt.AlignCenter)
            layout_controller.addWidget(lb)
            return lb
        def define_slider_slice(wanted_slice=6):
            slider_maximum = 11
            sl = QtGui.QSlider(1)
            sl.setMaximumHeight(250)
            sl.setValue(wanted_average * 100 / slider_maximum)

            sl.sliderReleased.connect(self.update_slider_slice)
            sl.sliderMoved.connect(self.update_slider_slice_title)

            layout_controller.addWidget(sl)

            return sl

        layout_title_and_controller = QtGui.QVBoxLayout()
        define_lb_title()

        layout_controller=define_layout_controller()

        self.lb_average=define_lb_average()
        self.slider_average=define_slider_average(wanted_average=wanted_average)
        self.update_slider_average()

        self.lb_slice=define_lb_slice()
        self.slider_slice=define_slider_slice()
        self.update_slider_slice()

        layout_title_and_controller.addLayout(layout_controller)

        self.layout_central.addLayout(layout_title_and_controller)

    def merge_layouts(self):
        #self.layout_global.addLayout(self.layout_option_settings)
        self.layout_global.addLayout(self.layout_central)

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
        self.main_plot = ImagePlotTest(axis,
                                       self.images,
                                       self,
                                       view='sag',
                                       line_direction='',
                                       im_params=self.im_params,
                                       canvas=self.canvas_main,
                                       header=self.header,
                                       number_of_points=self.number_of_points,
                                       first_label=self.first_label)


class ControlButtonsCore(object):
    """
    Core class for displaying and managing basic action buttons : help, undo and save & quit.
    Manages the layout, the adding of basic buttons and the associated functions.
    """
    def __init__(self, main_plot, window, header):
        self.main_plot = main_plot
        self.window = window
        self.help_web_adress = 'http://www.google.com'
        self.header = header

        self.layout_buttons = QtGui.QHBoxLayout()
        self.layout_buttons.setAlignment(QtCore.Qt.AlignRight)
        self.layout_buttons.setContentsMargins(10, 80, 15, 0)

    def add_classical_buttons(self):
        self.add_help_button()
        self.add_undo_button()
        self.add_save_and_quit_button()

    def add_save_and_quit_button(self):
        btn_save_and_quit = QtGui.QPushButton('Save & Quit')
        self.layout_buttons.addWidget(btn_save_and_quit)
        btn_save_and_quit.clicked.connect(self.press_save_and_quit)

    def add_undo_button(self):
        btn_undo = QtGui.QPushButton('Undo')
        self.layout_buttons.addWidget(btn_undo)
        btn_undo.clicked.connect(self.press_undo)

    def add_help_button(self):
        btn_help = QtGui.QPushButton('Help')
        self.layout_buttons.addWidget(btn_help)
        btn_help.clicked.connect(self.press_help)

    def press_help(self):
        webbrowser.open(self.help_web_adress, new=0, autoraise=True)

    def rewrite_list_points(self,list_points):
        list_points_useful_notation = ''
        for coord in list_points:
            if list_points_useful_notation:
                list_points_useful_notation += ':'
            list_points_useful_notation = list_points_useful_notation + str(coord.x) + ',' + \
                                          str(coord.y) + ',' + str(coord.z) + ',' + str(coord.value)
        return list_points_useful_notation

    def press_save_and_quit(self):
        self.window.str_points_final = self.rewrite_list_points(self.main_plot.list_points)

    def press_undo(self):
        if self.main_plot.list_points:
            del self.main_plot.list_points[-1]
            self.main_plot.draw_dots()
            self.header.update_text('update', len(self.main_plot.list_points), self.main_plot.number_of_points)
        else:
            self.header.update_text('warning_undo_beyond_first_point')


class ControlButtonsPropseg(ControlButtonsCore):
    """
    Inherites ControlButtonsCore
    Class that displays specific button for Propseg Viewer : Skip
    """
    def __init__(self, main_plot, window, header):
        super(ControlButtonsPropseg, self).__init__(main_plot, window, header)
        self.add_skip_button()
        self.add_classical_buttons()

    def add_skip_button(self):
        btn_skip = QtGui.QPushButton('Skip')
        self.layout_buttons.addWidget(btn_skip)
        btn_skip.clicked.connect(self.press_skip)

    def press_undo(self):
        if self.main_plot.list_points:
            del self.main_plot.list_points[-1]
            self.main_plot.draw_dots()
            self.header.update_text('update', len(self.main_plot.list_points), self.main_plot.number_of_points)
            self.main_plot.jump_to_new_slice()
        else:
            self.header.update_text('warning_undo_beyond_first_point')

    def press_skip(self):
        self.main_plot.list_points.append(Coordinate([-1, -1, -1]))
        self.header.update_text('update', len(self.main_plot.list_points), self.main_plot.number_of_points)
        self.main_plot.jump_to_new_slice()


class ControlButtonsLabelVertebrae(ControlButtonsCore):
    """
    Inherites ControlButtonsCore
    Class that could manage specific buttons for Label Vertebrae.
    """
    def __init__(self, main_plot, window, header):
        super(ControlButtonsLabelVertebrae, self).__init__(main_plot, window, header)
        self.add_classical_buttons()

    def press_undo(self):
        if self.main_plot.list_points:
            del self.main_plot.list_points[-1]
            self.main_plot.draw_dots()
            self.header.update_text('welcome', nbpt=str(self.main_plot.current_label))
        else:
            self.header.update_text('warning_undo_beyond_first_point')


class ControlButtonsGroundTruth(ControlButtonsCore):
    """
    Inherites ControlButtonsCore
    Class that displays specific button for Propseg Viewer : Skip
    """
    def __init__(self,
                 main_plot,
                 window,
                 header,
                 window_widget,
                 dict_save_niftii={},
                 bool_save_as_png=True):
        super(ControlButtonsGroundTruth, self).__init__(main_plot, window, header)
        self.bool_save_png_txt=bool_save_as_png
        self.dict_save_niftii=dict_save_niftii
        self.window_widget=window_widget

        self.add_save_options()
        self.add_skip_button()
        self.add_save_button()
        self.add_classical_buttons()

    def add_skip_button(self):
        btn_skip = QtGui.QPushButton('Skip')
        self.layout_buttons.addWidget(btn_skip)
        btn_skip.clicked.connect(self.press_skip)

    def add_save_button(self):
        btn_save = QtGui.QPushButton('Save')
        self.layout_buttons.addWidget(btn_save)
        btn_save.clicked.connect(self.press_save)

    def add_save_options(self):
        self.rm_png_txt = QtGui.QRadioButton('txt and png')
        self.rm_niftii = QtGui.QRadioButton('niftii')
        self.layout_buttons.addWidget(self.rm_png_txt)
        self.layout_buttons.addWidget(self.rm_niftii)

        if self.bool_save_png_txt:
            self.rm_png_txt.setChecked(True)
        else:
            self.rm_niftii.setChecked(True)
        self.rm_png_txt.clicked.connect(self.switch_save_format)
        self.rm_niftii.clicked.connect(self.switch_save_format)

    def switch_save_format(self):
        self.bool_save_png_txt=not self.bool_save_png_txt

    def find_point_with_max_label(self,list_points_on_slice):
        def translate_num_labels(lab):
            if lab==50:
                return -1
            elif lab==49:
                return 0
            else:
                return lab

        point_max=list_points_on_slice[0]
        for ipoints in list_points_on_slice:
            ipvalue = translate_num_labels(ipoints.value)
            mvalue = translate_num_labels(point_max.value)
            if ipvalue>mvalue:
                point_max = ipoints
        return point_max

    def press_undo(self):
        def redundant_removal(list_points,point_to_remove):
            """
            Function that does manually the removal of the point we want to remove.
            Its necessary as the usual comparaison of Coordinates does not take into account the Coordinate.value value,
            which is necessary to distinguish between the points that are filled in automatically in mainPlot.fill_first_labels.

            Parameters
            ----------
            list_points         : current self.list_points
            point_to_remove     : point we are about to undo

            Returns
            -------
            list_points_to_keep : new list_points without the point to remove.

            """
            list_points_to_keep=[]
            for ipoints in list_points:
                if ipoints.x==point_to_remove.x and ipoints.y==point_to_remove.y and ipoints.z==point_to_remove.z and ipoints.value==point_to_remove.value:
                    pass
                else:
                    list_points_to_keep.append(ipoints)
            return list_points_to_keep
        list_points_on_slice=self.main_plot.calc_list_points_on_slice()
        if len(list_points_on_slice)>0:
            self.main_plot.list_points=redundant_removal(self.main_plot.list_points,self.find_point_with_max_label(list_points_on_slice))
            self.main_plot.draw_dots()
            self.header.update_text('update', str(len(self.main_plot.calc_list_points_on_slice())+1), self.main_plot.number_of_points)
        else:
            self.header.update_text('warning_undo_beyond_first_point')

    def press_skip(self):
        self.main_plot.add_point_to_list_points(Coordinate([-1,
                                                            -1,
                                                            self.main_plot.current_position.z,
                                                            self.main_plot.dict_translate_label[str(len(self.main_plot.calc_list_points_on_slice())+1)]
                                                            ])) #pas necessaire

    def save_all_labels_as_txt(self):
        def calc_list_different_slices_in_list_point(list_points):
            list_slices = []
            for ipoints in list_points:
                if ipoints.x != -1 and not ipoints.z in list_slices:
                    list_slices.append(ipoints.z)
            return list_slices
        def calc_dict_labels_to_write(list_slice,list_points):
            dict_label_to_write={}
            for islice in list_slice:
                list_labels_to_write = []
                for ipoints in list_points:
                    if ipoints.z==islice:
                        list_labels_to_write.append(ipoints)
                dict_label_to_write[str(islice)]=list_labels_to_write
            return dict_label_to_write
        def fill_dict_list_points_with_missing_labels(dict_labels):
            for ikey in list(dict_labels.keys()):
                for imissing_labels in range(len(dict_labels[ikey]),27):
                    dict_labels[ikey].append(Coordinate([-1,-1,int(ikey),imissing_labels]))
            return dict_labels

        list_slices=calc_list_different_slices_in_list_point(self.main_plot.list_points)
        dict_label_to_write_uncomplete=calc_dict_labels_to_write(list_slices,self.main_plot.list_points)
        dict_label_to_write_complete=fill_dict_list_points_with_missing_labels(dict_label_to_write_uncomplete)

        file_path = self.manage_output_files_paths()
        (file_name,r) = self.seperate_file_name_and_path(self.window.file_name)
        for ikey in list(dict_label_to_write_complete.keys()):
            text_file = open(file_path+file_name+"_labels_slice_" + ikey + ".txt", "w")
            text_file.write(self.rewrite_list_points(dict_label_to_write_complete[ikey]))
        if list(dict_label_to_write_complete.keys()):
            text_file.close()

    def seperate_file_name_and_path(selfs,s):
        r=''
        while s!='' and s[-1]!='/':
            char = s[-1]
            r+=char
            s = s[:-1]
        return (r[::-1],s)

    def manage_output_files_paths(self):
        if self.window.output_name:
            (n,clean_path)=self.seperate_file_name_and_path(self.window.file_name)
            return clean_path+self.window.output_name+'/'
        else:
            return self.window.file_name + '_ground_truth/'

    def save_all_labelled_slices_as_png(self):
        def save_specific_slice_as_png(self,num_slice):
            file_path=self.manage_output_files_paths()
            (file_name,r)=self.seperate_file_name_and_path(self.window.file_name)
            image_array = self.main_plot.set_data_to_display(self.main_plot.images[0], Coordinate([-1, -1, num_slice]), self.main_plot.view)
            import scipy.misc
            scipy.misc.imsave(file_path+file_name+'_image_slice_'+str(num_slice)+'.png', image_array)
        def calc_list_different_slices_in_list_point(list_points):
            list_slices = []
            for ipoints in list_points:
                if ipoints.x != -1 and not ipoints.z in list_slices:
                    list_slices.append(ipoints.z)
            return list_slices

        list_slice=calc_list_different_slices_in_list_point(self.main_plot.list_points)
        for islice in list_slice:
            save_specific_slice_as_png(self,islice)

    def make_output_file(self):
        if not os.path.exists(self.manage_output_files_paths()):
            sct.run('mkdir ' + self.manage_output_files_paths())

    def press_save(self):
        if self.bool_save_png_txt:
            self.make_output_file()
            self.save_all_labelled_slices_as_png()
            self.save_all_labels_as_txt()
        else:
            if self.window.output_name:
                (f,clean_path)=self.seperate_file_name_and_path(self.window.file_name)
                file_name=clean_path+self.window.output_name
            else:
                file_name=self.window.file_name+'_ground_truth'
            file_name+='.nii.gz'

            self.dict_save_niftii['save_function'](self.rewrite_list_points(self.main_plot.list_points),
                                                  self.dict_save_niftii['reoriented_image_filename'],
                                                  self.dict_save_niftii['image_input_orientation'],
                                                  file_name)

    def press_save_and_quit(self):
        self.press_save()
        self.window_widget.close()

    def rewrite_list_points(self,list_points):
        list_points_useful_notation = ''
        for coord in list_points:
            if coord.x!=-1:
                if list_points_useful_notation:
                    list_points_useful_notation += ':'
                list_points_useful_notation = list_points_useful_notation + str(coord.x) + ',' + \
                                              str(coord.y) + ',' + str(coord.z) + ',' + str(coord.value)

        return list_points_useful_notation


class ControlButtonsTest(ControlButtonsCore):
    """
    Inherites ControlButtonsCore
    Class that displays specific button for Propseg Viewer : Skip
    """
    def __init__(self,
                 main_plot,
                 window,
                 header,
                 window_widget,
                 dict_save_niftii={},
                 bool_save_as_png=True):
        super(ControlButtonsTest, self).__init__(main_plot, window, header)
        self.bool_save_png_txt=bool_save_as_png
        self.dict_save_niftii=dict_save_niftii
        self.window_widget=window_widget
        self.output_name_file=os.getcwd()+'/dl_label_vertebrae_gt/'

        self.add_save_options()
        self.add_skip_button()
        self.add_save_button()
        self.add_classical_buttons()

    def add_skip_button(self):
        btn_skip = QtGui.QPushButton('Skip')
        self.layout_buttons.addWidget(btn_skip)
        btn_skip.clicked.connect(self.press_skip)

    def add_save_button(self):
        btn_save = QtGui.QPushButton('Save')
        self.layout_buttons.addWidget(btn_save)
        btn_save.clicked.connect(self.press_save)

    def add_save_options(self):
        self.rm_png_txt = QtGui.QRadioButton('txt and png')
        self.rm_niftii = QtGui.QRadioButton('niftii')
        self.layout_buttons.addWidget(self.rm_png_txt)
        self.layout_buttons.addWidget(self.rm_niftii)

        if self.bool_save_png_txt:
            self.rm_png_txt.setChecked(True)
        else:
            self.rm_niftii.setChecked(True)
        self.rm_png_txt.clicked.connect(self.switch_save_format)
        self.rm_niftii.clicked.connect(self.switch_save_format)

    def switch_save_format(self):
        self.bool_save_png_txt=not self.bool_save_png_txt

    def find_point_with_max_label(self,list_points_on_slice):
        def translate_num_labels(lab):
            if lab==50:
                return -1
            elif lab==49:
                return 0
            else:
                return lab

        point_max=list_points_on_slice[0]
        for ipoints in list_points_on_slice:
            ipvalue = translate_num_labels(ipoints.value)
            mvalue = translate_num_labels(point_max.value)
            if ipvalue>mvalue:
                point_max = ipoints
        return point_max

    def press_undo(self):
        def redundant_removal(list_points,point_to_remove):
            """
            Function that does manually the removal of the point we want to remove.
            Its necessary as the usual comparaison of Coordinates does not take into account the Coordinate.value value,
            which is necessary to distinguish between the points that are filled in automatically in mainPlot.fill_first_labels.

            Parameters
            ----------
            list_points         : current self.list_points
            point_to_remove     : point we are about to undo

            Returns
            -------
            list_points_to_keep : new list_points without the point to remove.

            """
            list_points_to_keep=[]
            for ipoints in list_points:
                if ipoints.x==point_to_remove.x and ipoints.y==point_to_remove.y and ipoints.z==point_to_remove.z and ipoints.value==point_to_remove.value:
                    pass
                else:
                    list_points_to_keep.append(ipoints)
            return list_points_to_keep
        list_points_on_slice=self.main_plot.calc_list_points_on_slice()
        if len(list_points_on_slice)>0:
            self.main_plot.list_points=redundant_removal(self.main_plot.list_points,self.find_point_with_max_label(list_points_on_slice))
            self.main_plot.draw_dots()
            self.header.update_text('update', str(len(self.main_plot.calc_list_points_on_slice())+1), self.main_plot.number_of_points)
        else:
            self.header.update_text('warning_undo_beyond_first_point')

    def press_skip(self):
        self.main_plot.add_point_to_list_points(Coordinate([-1,
                                                            -1,
                                                            self.main_plot.current_position.z,
                                                            self.main_plot.dict_translate_label[str(len(self.main_plot.calc_list_points_on_slice())+1)]
                                                            ])) #pas necessaire

    def save_all_labels_as_txt(self):
        def calc_list_different_slices_in_list_point(list_points):
            list_slices = []
            for ipoints in list_points:
                if ipoints.x != -1 and not ipoints.z in list_slices:
                    list_slices.append(ipoints.z)
            return list_slices
        def calc_dict_labels_to_write(list_slice,list_points):
            dict_label_to_write={}
            for islice in list_slice:
                list_labels_to_write = []
                for ipoints in list_points:
                    if ipoints.z==islice:
                        list_labels_to_write.append(ipoints)
                dict_label_to_write[str(islice)]=list_labels_to_write
            return dict_label_to_write
        def fill_dict_list_points_with_missing_labels(dict_labels):
            for ikey in list(dict_labels.keys()):
                for imissing_labels in range(len(dict_labels[ikey]),27):
                    dict_labels[ikey].append(Coordinate([-1,-1,int(ikey),imissing_labels]))
            return dict_labels

        list_slices=calc_list_different_slices_in_list_point(self.main_plot.list_points)
        dict_label_to_write_uncomplete=calc_dict_labels_to_write(list_slices,self.main_plot.list_points)
        dict_label_to_write_complete=fill_dict_list_points_with_missing_labels(dict_label_to_write_uncomplete)

        file_path = self.manage_output_files_paths()
        (file_name,r) = self.seperate_file_name_and_path(self.window.file_name)
        for ikey in list(dict_label_to_write_complete.keys()):
            text_file = open(file_path+file_name+"_labels_slice_" + ikey + ".txt", "w")
            text_file.write(self.rewrite_list_points(dict_label_to_write_complete[ikey]))
        if list(dict_label_to_write_complete.keys()):
            text_file.close()

    def save_txt_file(self):
        contrast,patient_name=self.extract_information_from_title(self.window.file_name)
        text_file = open(self.output_name_file+ patient_name + '_' + contrast +'_gt' + ".txt", "w")
        text_file.write(self.rewrite_list_points(self.main_plot.list_points))
        text_file.close()

    def extract_information_from_title(self,name):
        (file_name,adress)=self.seperate_file_name_and_path(name)
        (contrast,adress)=self.seperate_file_name_and_path(adress)
        (patient_name,adress)=self.seperate_file_name_and_path(adress)
        return(contrast,patient_name)

    def seperate_file_name_and_path(selfs,s):
        r=''
        if s[-1]=='/':
            s=s[:-1]
        while s!='' and s[-1]!='/':
            char = s[-1]
            r+=char
            s = s[:-1]
        return (r[::-1],s)

    def save_average_slice(self):
        contrast,patient_name=self.extract_information_from_title(self.window.file_name)
        image_array = self.main_plot.show_image_mean(nb_slice_to_average=self.main_plot.nb_slice_to_average)
        import scipy.misc
        scipy.misc.imsave(self.output_name_file+ patient_name +'_'+ contrast +'_gt'+'.png', image_array)

    def make_output_file(self):
        if not os.path.exists(self.output_name_file):
            sct.run('mkdir ' + self.output_name_file)

    def save_niftii(self):
        contrast,patient_name=self.extract_information_from_title(self.window.file_name)
        file_name = self.output_name_file+ patient_name +'_'+ contrast +'_gt'+'.nii.gz'
        self.dict_save_niftii['save_function'](self.rewrite_list_points(self.main_plot.list_points),
                                               self.dict_save_niftii['reoriented_image_filename'],
                                               self.dict_save_niftii['image_input_orientation'],
                                               file_name)

    def press_save(self):
        self.make_output_file()
        self.save_average_slice()
        self.save_txt_file()
        self.save_niftii()

    def press_save_and_quit(self):
        self.press_save()
        self.window_widget.close()

    def rewrite_list_points(self,list_points):
        list_points_useful_notation = ''
        for coord in list_points:
            if coord.x!=-1:
                if list_points_useful_notation:
                    list_points_useful_notation += ':'
                list_points_useful_notation = list_points_useful_notation + str(coord.x) + ',' + \
                                              str(coord.y) + ',' + str(coord.z) + ',' + str(coord.value)

        return list_points_useful_notation


class WindowCore(object):
    """
    Core Class that manages the qt window.
    Defines some core function to display images.
    """
    def __init__(self, list_input, visualization_parameters=None):
        self.images = self.keep_only_images(list_input)
        self.im_params = visualization_parameters
        self.str_points_final = ''

        self.mean_intensity = []
        self.std_intensity = []

    def keep_only_images(self, list_input):
        # TODO: check same space
        # TODO: check if at least one image
        from msct_image import Image
        images = []
        for im in list_input:
            if isinstance(im, Image):
                images.append(im)
            else:
                print "Error, one of the images is actually not an image..."
        return images

    def compute_offset(self):
        array_dim = [self.image_dim[0] * self.im_spacing[0], self.image_dim[1] * self.im_spacing[1],
                     self.image_dim[2] * self.im_spacing[2]]
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


class WindowPropseg(WindowCore):
    """
    Inherites Window Core.
    Defines global variables and sets layout in the whole Propseg Viewer.
    """
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

        super(WindowPropseg, self).__init__(list_images, visualization_parameters)
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
        self.dict_axis_buttons = {}
        self.closed = False

        self.number_of_slices = 0
        self.gap_inter_slice = 0

        # specialized for Click viewer
        self.list_points = []
        self.list_points_useful_notation = ''

        # compute slices to display
        self.list_slices = []

    def set_layout_and_launch_viewer(self):
        (window, system) = self.launch_main_window()
        layout_main = self.add_layout_main(window)
        self.header = self.add_header(layout_main)
        self.main_pannel = self.add_main_pannel(layout_main, self, self.header)
        self.control_buttons = self.add_control_buttons(layout_main, self)
        window.setLayout(layout_main)
        sys.exit(system.exec_())

    def launch_main_window(self):
        system = QtGui.QApplication(sys.argv)
        w = QtGui.QWidget()
        w.resize(740, 850)
        w.setWindowTitle('Propseg Viewer')

        w.setWindowIcon(QtGui.QIcon(os.path.dirname(os.path.realpath(__file__)).replace('/scripts','/documentation/logo_sct.png')))

        w.show()
        return (w, system)

    def add_layout_main(self,window):
        layout_main = QtGui.QVBoxLayout()
        layout_main.setAlignment(QtCore.Qt.AlignTop)
        window.setLayout(layout_main)
        return layout_main

    def add_header(self,layout_main):
        header = HeaderPropseg()
        layout_main.addLayout(header.layout_header)
        header.update_text('welcome')
        return (header)

    def add_main_pannel(self,layout_main, window, header):
        main_pannel = MainPannelPropseg(self.images, self.im_params, window, header)
        layout_main.addLayout(main_pannel.layout_global)
        return main_pannel

    def add_control_buttons(self,layout_main, window):
        control_buttons = ControlButtonsPropseg(self.main_pannel.main_plot, window, self.header)
        layout_main.addLayout(control_buttons.layout_buttons)
        return control_buttons


class WindowLabelVertebrae(WindowCore):
    """
    Inherites Window Core.
    Defines global variables and sets layout in the whole Label Vertebrae Viewer.
    """
    def __init__(self,
                 list_images,
                 visualization_parameters=None,
                 orientation_subplot=['ax', 'sag'],
                 input_type='centerline',
                 wanted_label=12):

        # Ajust the input parameters into viewer objects.
        if isinstance(list_images, Image):
            list_images = [list_images]
        if not visualization_parameters:
            visualization_parameters = ParamMultiImageVisualization([ParamImageVisualization()])

        super(WindowLabelVertebrae, self).__init__(list_images, visualization_parameters)
        self.set_layout_and_launch_viewer(wanted_label=wanted_label)

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
        self.dict_axis_buttons = {}
        self.closed = False

        self.number_of_slices = 0
        self.gap_inter_slice = 0

        # specialized for Click viewer
        self.list_points = []
        self.list_points_useful_notation = ''

        # compute slices to display
        self.list_slices = []

    def set_layout_and_launch_viewer(self,wanted_label):
        (window, system) = self.launch_main_window()
        layout_main = self.add_layout_main(window)
        self.header = self.add_header(layout_main)
        self.main_pannel = self.add_main_pannel(layout_main, self, self.header,wanted_label=wanted_label)
        self.control_buttons = self.add_control_buttons(layout_main, self)
        window.setLayout(layout_main)
        sys.exit(system.exec_())

    def launch_main_window(self):
        system = QtGui.QApplication(sys.argv)
        w = QtGui.QWidget()
        w.resize(740, 850)
        w.setWindowTitle('Label Vertebrae Viewer')

        w.setWindowIcon(QtGui.QIcon(os.path.dirname(os.path.realpath(__file__)).replace('/scripts','/documentation/logo_sct.png')))

        w.show()
        return (w, system)

    def add_layout_main(self, window):
        layout_main = QtGui.QVBoxLayout()
        layout_main.setAlignment(QtCore.Qt.AlignTop)
        window.setLayout(layout_main)
        return layout_main

    def add_header(self, layout_main):
        header = HeaderLabelVertebrae()
        layout_main.addLayout(header.layout_header)
        start_slice = 4
        header.update_text('welcome', str(start_slice))
        return (header)

    def add_main_pannel(self, layout_main, window, header,wanted_label):
        main_pannel = MainPannelLabelVertebrae(self.images, self.im_params, window, header,wanted_label=wanted_label)
        layout_main.addLayout(main_pannel.layout_global)
        return main_pannel

    def add_control_buttons(self, layout_main, window):
        control_buttons = ControlButtonsLabelVertebrae(self.main_pannel.main_plot, window, self.header)
        layout_main.addLayout(control_buttons.layout_buttons)
        return control_buttons


class WindowGroundTruth(WindowCore):
    """
    Inherites Window Core.
    Defines global variables and sets layout in the whole Propseg Viewer.
    """
    def __init__(self,
                 list_images,
                 visualization_parameters=None,
                 first_label=1,
                 file_name='',
                 output_path='',
                 dict_save_niftii={},
                 bool_save_as_png=True):

        # Ajust the input parameters into viewer objects.
        self.bool_save_as_png=bool_save_as_png
        (self.file_name,self.output_name)=self.choose_and_clean_file_name(file_name,output_path)
        self.first_label=int(first_label)
        self.dict_save_niftii=dict_save_niftii
        if isinstance(list_images, Image):
            list_images = [list_images]
        if not visualization_parameters:
            visualization_parameters = ParamMultiImageVisualization([ParamImageVisualization()])


        super(WindowGroundTruth, self).__init__(list_images, visualization_parameters)
        self.set_layout_and_launch_viewer()

    def choose_and_clean_file_name(self,file_name,output_path):
        return (file_name.replace('.nii.gz',''),output_path)

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
        self.dict_axis_buttons = {}
        self.closed = False

        self.number_of_slices = 0
        self.gap_inter_slice = 0

        # specialized for Click viewer
        self.list_points = []
        self.list_points_useful_notation = ''

        # compute slices to display
        self.list_slices = []

    def set_layout_and_launch_viewer(self):
        (window, system) = self.launch_main_window()
        layout_main = self.add_layout_main(window)
        self.header = self.add_header(layout_main)
        self.main_pannel = self.add_main_pannel(layout_main, self, self.header)
        self.import_existing_labels()
        self.control_buttons = self.add_control_buttons(layout_main, self,window_widget=window)
        window.setLayout(layout_main)
        sys.exit(system.exec_())

    def launch_main_window(self):
        system = QtGui.QApplication(sys.argv)
        w = QtGui.QWidget()
        w.resize(740, 850)
        w.setWindowTitle('Ground Truth Viewer')

        w.setWindowIcon(QtGui.QIcon(os.path.dirname(os.path.realpath(__file__)).replace('/scripts','/documentation/logo_sct.png')))

        w.show()
        return (w, system)

    def add_layout_main(self,window):
        layout_main = QtGui.QVBoxLayout()
        layout_main.setAlignment(QtCore.Qt.AlignTop)
        window.setLayout(layout_main)
        return layout_main

    def add_header(self,layout_main):
        header = HeaderGroundTruth()
        layout_main.addLayout(header.layout_header)
        header.update_text('welcome',nbpt='1')
        return (header)

    def add_main_pannel(self,layout_main, window, header):
        main_pannel = MainPannelGroundTruth(self.images, self.im_params, window, header,first_label=self.first_label)
        layout_main.addLayout(main_pannel.layout_global)
        return main_pannel

    def add_control_buttons(self,layout_main, window,window_widget):
        control_buttons = ControlButtonsGroundTruth(self.main_pannel.main_plot,
                                                    window,
                                                    self.header,
                                                    window_widget,
                                                    dict_save_niftii=self.dict_save_niftii,
                                                    bool_save_as_png=self.bool_save_as_png)
        layout_main.addLayout(control_buttons.layout_buttons)
        return control_buttons

    def seperate_file_name_and_path(selfs,s):
        r=''
        while s!='' and s[-1]!='/':
            char = s[-1]
            r+=char
            s = s[:-1]
        return (r[::-1],s)

    def import_existing_labels(self):
        def get_txt_files_in_output_directory(file_name,output_name):
            if output_name:
                (n,path)=self.seperate_file_name_and_path(self.file_name)
                output_file_name=path+output_name
            else:
                output_file_name=file_name
                output_file_name+='_ground_truth/'

            if os.path.exists(output_file_name):
                return (list(filter(lambda x: '.txt' in x,os.listdir(output_file_name))),output_file_name)
            else:
                return ([],output_file_name)
        def extract_coordinates(output_file_name,txt_file,file_name,output_name):
            if output_name:
                (n,path)=self.seperate_file_name_and_path(self.file_name)
                output_file_name=path+output_name+'/'
            else:
                output_file_name=file_name
                output_file_name+='_ground_truth/'
            file=open(output_file_name+txt_file,"r")
            list_coordinates = []
            for line in file:
                coordinates=''
                for char in line:
                    if char==':':
                        list_coordinates.append(coordinates)
                        coordinates=''
                    else:
                        coordinates+=char
                list_coordinates.append(coordinates)
            return list_coordinates
        def make_dict_labels():
            dict_labels={'50':Coordinate([-1,-1,-1,50]),
                        '49': Coordinate([-1, -1, -1, 49]),
                        '1': Coordinate([-1, -1, -1, 1]),
                        '3': Coordinate([-1, -1, -1, 3]),
                        '4': Coordinate([-1, -1, -1, 4]),
                        }
            for ii in range (5,27):
                dict_labels[str(ii)]=Coordinate([-1,-1,-1,ii])
            return dict_labels
        def complete_dict_labels(dict_labels,list_coordinates):
            def update_max_label(current_label,max_label):
                if int(current_label)==50:
                    current_label=-1
                elif int(current_label)==49:
                    current_label=0
                else:
                    current_label=int(current_label)

                if current_label>max_label:
                    max_label=current_label
                return max_label
            def remove_points_beyond_last_selected_label(dict_labels,max_label):
                for ikey in list(dict_labels.keys()):
                    if ikey=='49':
                        if max_label==-1:
                            del dict_labels[ikey]
                    else:
                        if max_label<int(ikey) and ikey!='50':
                            del dict_labels[ikey]
                return dict_labels
            def turn_string_coord_into_list_coord(coordinates):
                list_pos=[]
                pos=''
                for char in coordinates:
                    if char ==',':
                        list_pos.append(pos)
                        pos=''
                    else:
                        pos+=char
                list_pos.append(pos)
                return list_pos
            max_label=-5
            for coordinates in list_coordinates:
                list_pos=turn_string_coord_into_list_coord(coordinates)
                if list_pos[0]!='-1':
                    max_label=update_max_label(list_pos[3],max_label)
                dict_labels[list_pos[3]]=Coordinate([int(list_pos[0]),int(list_pos[1]),int(list_pos[2]),int(list_pos[3])])
                dict_labels=remove_points_beyond_last_selected_label(dict_labels,max_label)
            return dict_labels

        list_txt,path=get_txt_files_in_output_directory(self.file_name,self.output_name)
        for ilabels in list_txt:
            dict_labels=make_dict_labels()
            list_coordinates=extract_coordinates(path,ilabels,self.file_name,self.output_name)
            dict_labels=complete_dict_labels(dict_labels,list_coordinates)
            for ikey in list(dict_labels.keys()):
                self.main_pannel.main_plot.list_points.append(dict_labels[ikey])
        self.main_pannel.main_plot.draw_dots()
        self.main_pannel.second_plot.draw_lines()
        if self.main_pannel.main_plot.calc_list_points_on_slice():
            self.header.update_text('update',str(len(self.main_pannel.main_plot.calc_list_points_on_slice())+1))
            sct.printv('Output file you have chosen contained results of a previous labelling.\n'
                       'This data has been imported.',type='info')


class WindowTest(WindowCore):
    """
    Inherites Window Core.
    Defines global variables and sets layout in the whole Propseg Viewer.
    """
    def __init__(self,
                 list_images,
                 visualization_parameters=None,
                 first_label=1,
                 file_name='',
                 output_path='',
                 dict_save_niftii={},
                 bool_save_as_png=True):

        # Ajust the input parameters into viewer objects.
        self.bool_save_as_png=bool_save_as_png
        (self.file_name,self.output_name)=self.choose_and_clean_file_name(file_name,output_path)
        self.first_label=int(first_label)
        self.dict_save_niftii=dict_save_niftii
        if isinstance(list_images, Image):
            list_images = [list_images]
        if not visualization_parameters:
            visualization_parameters = ParamMultiImageVisualization([ParamImageVisualization()])


        super(WindowTest, self).__init__(list_images, visualization_parameters)
        self.set_layout_and_launch_viewer()

    def choose_and_clean_file_name(self,file_name,output_path):
        return (file_name.replace('.nii.gz',''),output_path)

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
        self.dict_axis_buttons = {}
        self.closed = False

        self.number_of_slices = 0
        self.gap_inter_slice = 0

        # specialized for Click viewer
        self.list_points = []
        self.list_points_useful_notation = ''

        # compute slices to display
        self.list_slices = []

    def set_layout_and_launch_viewer(self):
        (window, system) = self.launch_main_window()
        layout_main = self.add_layout_main(window)
        self.header = self.add_header(layout_main)
        self.main_pannel = self.add_main_pannel(layout_main, self, self.header)
        self.import_existing_labels()
        self.control_buttons = self.add_control_buttons(layout_main, self,window_widget=window)
        window.setLayout(layout_main)
        sys.exit(system.exec_())

    def launch_main_window(self):
        system = QtGui.QApplication(sys.argv)
        w = QtGui.QWidget()
        w.resize(740, 850)
        w.setWindowTitle('Ground Truth Viewer')

        w.setWindowIcon(QtGui.QIcon(os.path.dirname(os.path.realpath(__file__)).replace('/scripts','/documentation/logo_sct.png')))

        w.show()
        return (w, system)

    def add_layout_main(self,window):
        layout_main = QtGui.QVBoxLayout()
        layout_main.setAlignment(QtCore.Qt.AlignTop)
        window.setLayout(layout_main)
        return layout_main

    def add_header(self,layout_main):
        header = HeaderGroundTruth()
        layout_main.addLayout(header.layout_header)
        header.update_text('welcome',nbpt='1')
        return (header)

    def add_main_pannel(self,layout_main, window, header):
        main_pannel = MainPannelTest(self.images, self.im_params, window, header,first_label=self.first_label)
        layout_main.addLayout(main_pannel.layout_global)
        return main_pannel

    def add_control_buttons(self,layout_main, window,window_widget):
        control_buttons = ControlButtonsTest(self.main_pannel.main_plot,
                                                    window,
                                                    self.header,
                                                    window_widget,
                                                    dict_save_niftii=self.dict_save_niftii,
                                                    bool_save_as_png=self.bool_save_as_png)
        layout_main.addLayout(control_buttons.layout_buttons)
        return control_buttons

    def seperate_file_name_and_path(selfs,s):
        r=''
        while s!='' and s[-1]!='/':
            char = s[-1]
            r+=char
            s = s[:-1]
        return (r[::-1],s)

    def import_existing_labels(self):
        def extract_information_from_title(name):
            (file_name, adress) = seperate_file_name_and_path(name)
            (contrast, adress) = seperate_file_name_and_path(adress)
            (patient_name, adress) = seperate_file_name_and_path(adress)
            return (contrast, patient_name)
        def seperate_file_name_and_path(s):
            r = ''
            if s[-1] == '/':
                s = s[:-1]
            while s != '' and s[-1] != '/':
                char = s[-1]
                r += char
                s = s[:-1]
            return (r[::-1], s)
        def get_txt_files_in_output_directory(file_name,output_name):
            output_file_name='dl_label_vertebrae_gt/'
            (contrast, patient_name)=extract_information_from_title(file_name)
            if os.path.exists(output_file_name+patient_name+'_'+contrast+'_gt.txt'):
                return ([output_file_name+patient_name+'_'+contrast+'_gt.txt'],output_file_name)
            else:
                return ([],output_file_name)
        def extract_coordinates(output_file_name,txt_file,file_name,output_name):
            if output_name:
                (n,path)=self.seperate_file_name_and_path(self.file_name)
                output_file_name=path+output_name+'/'
            else:
                output_file_name=file_name
                output_file_name+='_ground_truth/'
            file=open(txt_file,"r")
            list_coordinates = []
            for line in file:
                coordinates=''
                for char in line:
                    if char==':':
                        list_coordinates.append(coordinates)
                        coordinates=''
                    else:
                        coordinates+=char
                list_coordinates.append(coordinates)
            return list_coordinates
        def make_dict_labels():
            dict_labels={'50':Coordinate([-1,-1,-1,50]),
                        '49': Coordinate([-1, -1, -1, 49]),
                        '1': Coordinate([-1, -1, -1, 1]),
                        '3': Coordinate([-1, -1, -1, 3]),
                        '4': Coordinate([-1, -1, -1, 4]),
                        }
            for ii in range (5,27):
                dict_labels[str(ii)]=Coordinate([-1,-1,-1,ii])
            return dict_labels
        def complete_dict_labels(dict_labels,list_coordinates):
            def update_max_label(current_label,max_label):
                if int(current_label)==50:
                    current_label=-1
                elif int(current_label)==49:
                    current_label=0
                else:
                    current_label=int(current_label)

                if current_label>max_label:
                    max_label=current_label
                return max_label
            def remove_points_beyond_last_selected_label(dict_labels,max_label):
                for ikey in list(dict_labels.keys()):
                    if ikey=='49':
                        if max_label==-1:
                            del dict_labels[ikey]
                    else:
                        if max_label<int(ikey) and ikey!='50':
                            del dict_labels[ikey]
                return dict_labels
            def turn_string_coord_into_list_coord(coordinates):
                list_pos=[]
                pos=''
                for char in coordinates:
                    if char ==',':
                        list_pos.append(pos)
                        pos=''
                    else:
                        pos+=char
                list_pos.append(pos)
                return list_pos
            max_label=-5
            for coordinates in list_coordinates:
                list_pos=turn_string_coord_into_list_coord(coordinates)
                if list_pos[0]!='-1':
                    max_label=update_max_label(list_pos[3],max_label)
                dict_labels[list_pos[3]]=Coordinate([int(list_pos[0]),int(list_pos[1]),int(list_pos[2]),int(list_pos[3])])
                dict_labels=remove_points_beyond_last_selected_label(dict_labels,max_label)
            return dict_labels

        list_txt,path=get_txt_files_in_output_directory(self.file_name,self.output_name)
        for ilabels in list_txt:
            dict_labels=make_dict_labels()
            list_coordinates=extract_coordinates(path,ilabels,self.file_name,self.output_name)
            dict_labels=complete_dict_labels(dict_labels,list_coordinates)
            for ikey in list(dict_labels.keys()):
                self.main_pannel.main_plot.list_points.append(dict_labels[ikey])
        self.main_pannel.main_plot.draw_dots()
        if self.main_pannel.main_plot.calc_list_points_on_slice():
            self.header.update_text('update',str(len(self.main_pannel.main_plot.calc_list_points_on_slice())+1))
            sct.printv('Output file you have chosen contained results of a previous labelling.\n'
                       'This data has been imported.',type='info')


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
    def __init__(self, id='0', mode='image', cmap='gray', interp='nearest', vmin='0', vmax='99', vmean='98',
                 vmode='percentile', alpha='1.0'):
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
