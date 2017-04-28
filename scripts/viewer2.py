import sys
import webbrowser
from copy import copy
from time import time

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
        dic = {'1': 50,
               '2': 49,
               '3': 1,
               '4': 3, }
        for ii in range(5, 27):
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
        return 0 <= target_point.x < self.image_dim[0] and 0 <= target_point.y < self.image_dim[
            1] and 0 <= target_point.z < self.image_dim[2]


class ImagePlotMain(ImagePlot):
    """
    Inherites ImagePlot
    Class used for displaying the main (right) picture in Propseg Viewer
    Defines the action on mouse events, draw dots, and manages the list of results : list_points.
    """
    def __init__(self, ax, images, viewer, canvas, view, line_direction='hv', im_params=None, secondary_plot=None,
                 header=None, number_of_points=0):
        super(ImagePlotMain, self).__init__(ax, images, viewer, canvas, view, line_direction, im_params, header)
        self.secondary_plot = secondary_plot
        self.plot_points, = self.axes.plot([], [], '.r', markersize=10)
        self.show_image(self.im_params, current_point=None)
        self.number_of_points = number_of_points
        self.calculate_list_slices()
        self.update_slice(Coordinate([self.list_slices[0], self.current_position.y, self.current_position.z]))
        self.bool_is_mode_auto = True
        # print(self.list_slices)

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
        if self.secondary_plot:
            self.secondary_plot.change_intensity(event)

    def refresh(self):
        self.figs[-1].figure.canvas.draw()

    def draw_dots(self):
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


class ImagePlotSecond(ImagePlot):
    """
    Inherites ImagePlot
    Class used for displaying the secondary (left) picture in Propseg Viewer
    Defines the action on mouse events, draw lines and update the slice on the main picture.
    """
    def __init__(self, ax, images, viewer, canvas, main_single_plot, view, line_direction='hv', im_params=None,
                 header=None):
        super(ImagePlotSecond, self).__init__(ax, images, viewer, canvas, view, line_direction, im_params, header)
        self.main_plot = main_single_plot
        self.current_position = self.main_plot.current_position
        self.list_previous_lines = []

        self.show_image(self.im_params, current_point=None)
        self.current_line = self.calc_line('v',
                                           self.current_position)  # add_line is used in stead of draw_line because in draw_line we also remove the previous line.
        self.axes.add_line(self.current_line)
        self.refresh()

    def calc_line(self, line_direction, line_position, line_color='white'):
        def calc_dic_line_coor(current_position, view):
            if view == 'ax':
                return {'v': [[current_position.y, current_position.y], [-10000, 10000]],
                        'h': [[-10000, 10000], [current_position.z, current_position.z]]}
            elif view == 'cor':
                return {'v': [[current_position.x, current_position.x], [-10000, 10000]],
                        'h': [[-10000, 10000], [current_position.z, current_position.z]]}
            elif view == 'sag':
                return {'v': [[current_position.x, current_position.x], [-10000, 10000]],
                        'h': [[-10000, 10000], [current_position.y, current_position.y]]}

        dic_line_coor = calc_dic_line_coor(line_position, self.view)
        line = Line2D(dic_line_coor[line_direction][1], dic_line_coor[line_direction][0], color=line_color)
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
                print(self.list_points)
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


class HeaderCore(object):
    """
    Core Class for Header
    Defines Layouts and some basic messages.
    """
    def __init__(self):
        self.define_layout_header()
        self.add_lb_status()
        self.add_lb_warning()
        self.dic_message_labels = self.define_dic_message_labels()

    def define_dic_message_labels(self):
        dic = {'1': 'Please click on anterior base \n'
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
        return dic

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
        self.layout_header.setContentsMargins(0, 30, 0, 80)

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
                self.lb_status.setText('You have maid ' + str(nbpt) + ' points.')
                self.lb_status.setStyleSheet("color:black")
            else:
                self.lb_status.setText('You have maid ' + str(nbpt) + ' points out of ' + str(nbfin) + '.')
                self.lb_status.setStyleSheet("color:black")

        else:
            self.lb_warning.setText(key + ' : Unknown key')
            self.lb_warning.setStyleSheet("color:red")


class Header(HeaderCore):
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
            self.lb_status.setText(self.dic_message_labels[nbpt])
            self.lb_status.setStyleSheet("color:black")

        elif (key == 'warning_skip_not_defined'):
            self.lb_warning.setText('This option is not used in Manual Mode. \n')
            self.lb_warning.setStyleSheet("color:red")
        elif (key == 'mode_switched'):
            self.lb_status.setText('You have switched on an other segmentation mode. \n'
                                   'All previous data have been erased.')
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
        self.main_plot = ImagePlotMain(axis, self.images, self, view='ax', line_direction='', im_params=self.im_params,
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
        self.second_plot = ImagePlotSecond(axis, self.images, self, view='sag', line_direction='',
                                            im_params=self.im_params, canvas=self.canvas_second,
                                            main_single_plot=self.main_plot, header=self.header)
        self.main_plot.secondary_plot = self.second_plot

    def merge_layouts(self):
        self.layout_global.addLayout(self.layout_option_settings)
        self.layout_global.addLayout(self.layout_central)


class MainPannel(MainPannelCore):
    """
    Inherites MainPannelCore
    Class that defines specific main image plot and secondary image plot for Propseg Viewer.
    """
    def __init__(self, images, im_params, window, header):
        super(MainPannel, self).__init__(images, im_params, window, header)

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
        self.main_plot = ImagePlotMain(axis, self.images, self, view='ax', line_direction='', im_params=self.im_params,
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
        self.second_plot = ImagePlotSecond(axis, self.images, self, view='sag', line_direction='',
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
    def __init__(self, images, im_params, window, header):
        super(MainPannelLabelVertebrae, self).__init__(images, im_params, window, header)

        self.number_of_points = 1
        self.add_main_view()
        self.add_controller_pannel()
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

    def add_controller_pannel(self):
        def update_slider_label():
            if not self.main_plot.list_points:
                slider_real_value = slider_maximum - int(slider_maximum * self.slider_label.value() / 100)
                self.header.update_text('welcome', str(slider_real_value))
                self.main_plot.current_label = slider_real_value
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
        init_label = slider_maximum - 3
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
        self.layout_buttons.setContentsMargins(10, 80, 15, 160)

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

    def press_save_and_quit(self):
        def rewrite_list_points(list_points):
            list_points_useful_notation = ''
            for coord in list_points:
                if coord.x != -1:  # check either the point has been placed or skipped.
                    if list_points_useful_notation:
                        list_points_useful_notation += ':'
                    list_points_useful_notation = list_points_useful_notation + str(coord.x) + ',' + \
                                                  str(coord.y) + ',' + str(coord.z) + ',' + str(coord.value)
            return list_points_useful_notation

        self.window.str_points_final = rewrite_list_points(self.main_plot.list_points)

    def press_undo(self):
        if self.main_plot.list_points:
            del self.main_plot.list_points[-1]
            self.main_plot.draw_dots()
            self.header.update_text('update', len(self.main_plot.list_points), self.main_plot.number_of_points)
        else:
            self.header.update_text('warning_undo_beyond_first_point')


class ControlButtons(ControlButtonsCore):
    """
    Inherites ControlButtonsCore
    Class that displays specific button for Propseg Viewer : Skip
    """
    def __init__(self, main_plot, window, header):
        super(ControlButtons, self).__init__(main_plot, window, header)
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

    def start(self):
        return self.list_points_useful_notation


class Window(WindowCore):
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
            w.setWindowTitle('Propseg Viewer')
            w.show()
            return (w, system)

        def add_layout_main(window):
            layout_main = QtGui.QVBoxLayout()
            layout_main.setAlignment(QtCore.Qt.AlignTop)
            window.setLayout(layout_main)
            return layout_main

        def add_header(layout_main):
            header = Header()
            layout_main.addLayout(header.layout_header)
            header.update_text('welcome')
            return (header)

        def add_main_pannel(layout_main, window, header):
            main_pannel = MainPannel(self.images, self.im_params, window, header)
            layout_main.addLayout(main_pannel.layout_global)
            return main_pannel

        def add_control_buttons(layout_main, window):
            control_buttons = ControlButtons(self.main_pannel.main_plot, window, self.header)
            layout_main.addLayout(control_buttons.layout_buttons)
            return control_buttons

        (window, system) = launch_main_window()
        layout_main = add_layout_main(window)
        self.header = add_header(layout_main)
        self.main_pannel = add_main_pannel(layout_main, self, self.header)
        self.control_buttons = add_control_buttons(layout_main, self)
        window.setLayout(layout_main)
        sys.exit(system.exec_())


class WindowLabelVertebrae(WindowCore):
    """
    Inherites Window Core.
    Defines global variables and sets layout in the whole Label Vertebrae Viewer.
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

        super(WindowLabelVertebrae, self).__init__(list_images, visualization_parameters)
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
        w.setWindowTitle('Label Vertebrae Viewer')
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

    def add_main_pannel(self, layout_main, window, header):
        main_pannel = MainPannelLabelVertebrae(self.images, self.im_params, window, header)
        layout_main.addLayout(main_pannel.layout_global)
        return main_pannel

    def add_control_buttons(self, layout_main, window):
        control_buttons = ControlButtonsLabelVertebrae(self.main_pannel.main_plot, window, self.header)
        layout_main.addLayout(control_buttons.layout_buttons)
        return control_buttons


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
