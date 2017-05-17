import webbrowser
from copy import copy
from time import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
import scripts.sct_utils as sct
from matplotlib.backends.backend_qt4agg import \
    FigureCanvasQTAgg as FigureCanvas
from scripts.msct_types import Coordinate


class ImagePlot(object):
    """
    Core class for displaying pictures.
    Defines the data to display, image parameters, on_scroll event, manages intensity and manages the click (in or out the picture,...).
    """
    def __init__(self, ax, images, viewer, canvas, view, line_direction='hv', im_params=None, header=None):
        """
        Parameters
        ----------

        :param ax:
        :param images:
        :param viewer:
        :param canvas:
        :param view:
        :param line_direction:
        :param im_params:
        :param header:
        """
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
        self.current_position = Coordinate([int(self.images[0].data.shape[0] / 2),
                                            int(self.images[0].data.shape[1] / 2),
                                            int(self.images[0].data.shape[2] / 2)])
        self.list_points = []
        self.header = header
        self.dict_translate_label = self.define_translate_dict()

        self.remove_axis_number()
        self.connect_mpl_events()
        self.setup_intensity()

    def define_translate_dict(self):
        """
        Defines dictionary to translate the software labels which range is [1;27] into anatomical labels which range is:
        {50;49} U {1} U [3,26]
        It does not matter if the dictionnary is a bit too long. The number of possible labels is still 27.
        Returns
        -------
        dict        Dictionary that links the label position, from top to down, to the label definition according to
                    https://github.com/neuropoly/spinalcordtoolbox/issues/1205
        """
        dict = {'1': 50,
                '2': 49,
                '3': 1,
                '4': 3}
        for ii in range(5, 30):
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
            first_percentile = np.percentile(flattened_volume[flattened_volume > 0], 0)
            last_percentile = np.percentile(flattened_volume[flattened_volume > 0], 99)
            mean_intensity = np.percentile(flattened_volume[flattened_volume > 0], 98)
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
        dict = {'1': 'Please click on anterior base \n of pontomedullary junction (label=50) \n',
                '2': 'Please click on pontomedullary groove \n (label=49) \n',

                '3': 'Please click on top of C1 vertebrae \n (label=1) \n',
                '4': 'Please click on posterior edge of \n C2/C3 intervertebral disk (label=3) \n',
                '5': 'Please click on posterior edge of \n C3/C4 intervertebral disk (label=4) \n',
                '6': 'Please click on posterior edge of \n C4/C5 intervertebral disk (label=5) \n',
                '7': 'Please click on posterior edge of \n C5/C6 intervertebral disk (label=6) \n',
                '8': 'Please click on posterior edge of \n C6/C7 intervertebral disk (label=7) \n',
                '9': 'Please click on posterior edge of \n C7/T1 intervertebral disk (label=8) \n',

                '10': 'Please click on posterior edge of \n T1/T2 intervertebral disk (label=9) \n',
                '11': 'Please click on posterior edge of \n T2/T3 intervertebral disk (label=10) \n',
                '12': 'Please click on posterior edge of \n T3/T4 intervertebral disk (label=11) \n',
                '13': 'Please click on posterior edge of \n T4/T5 intervertebral disk (label=12) \n',
                '14': 'Please click on posterior edge of \n T5/T6 intervertebral disk (label=13) \n',
                '15': 'Please click on posterior edge of \n T6/T7 intervertebral disk (label=14) \n',
                '16': 'Please click on posterior edge of \n T7/T8 intervertebral disk (label=15) \n',
                '17': 'Please click on posterior edge of \n T8/T9 intervertebral disk (label=16) \n',
                '18': 'Please click on posterior edge of \n T9/T10 intervertebral disk (label=17) \n',
                '19': 'Please click on posterior edge of \n T10/T11 intervertebral disk (label=18) \n',
                '20': 'Please click on posterior edge of \n T11/T12 intervertebral disk (label=19) \n',
                '21': 'Please click on posterior edge of \n T12/L1 intervertebral disk (label=20) \n',

                '22': 'Please click on posterior edge of \n L1/L2 intervertebral disk (label=21) \n',
                '23': 'Please click on posterior edge of \n L2/L3 intervertebral disk (label=22) \n',
                '24': 'Please click on posterior edge of \n L3/L4 intervertebral disk (label=23) \n',
                '25': 'Please click on posterior edge of \n L4/S1 intervertebral disk (label=24) \n',

                '26': 'Please click on posterior edge of \n S1/S2 intervertebral disk (label=25) \n',
                '27': 'Please click on posterior edge of \n S2/S3 intervertebral disk (label=26) \n'}
        return dict

    def add_lb_status(self):
        self.lb_status = QtGui.QLabel('Label Status')
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
        self.layout_header.setContentsMargins(0, 30, 0, 0)

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
        self.layout_buttons.setContentsMargins(10, 30, 15, 30)

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

    def rewrite_list_points(self, list_points):
        list_points_useful_notation = ''
        for coord in list_points:
            if list_points_useful_notation:
                list_points_useful_notation += ':'
            list_points_useful_notation = list_points_useful_notation + ','.join([str(coord.x),
                                                                                  str(coord.y),
                                                                                  str(coord.z),
                                                                                  str(coord.value)])
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
            image.data = np.pad(image.data,
                                ((self.offset[0], self.offset[0]),
                                 (self.offset[1], self.offset[1]),
                                 (self.offset[2], self.offset[2])),
                                'constant',
                                constant_values=(0, 0))


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
