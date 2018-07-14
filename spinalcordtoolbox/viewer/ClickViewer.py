class ClickViewer(Viewer):
    """
    This class is a visualizer for volumes (3D images) and ask user to click on axial slices.
    Assumes SAL orientation
    orientation_subplot: list of two views that will be plotted next to each other. The first view is the main one (right) and the second view is the smaller one (left). Orientations are: ax, sag, cor.
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
        super(ClickViewer, self).__init__(list_images, visualization_parameters)

        self.declaration_global_variables_general(orientation_subplot)

        self.compute_offset()
        self.pad_data()

        self.current_point = Coordinate([int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2), int(self.images[0].data.shape[2] / 2)])

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
        self.windows[1].axes.set_title('Select the slice \n '
                                       'to inspect. \n')

        """ Connect buttons to user actions"""
        for window in self.windows:
            window.connect()

        """ Create Buttons"""
        self.create_button_save_and_quit()
        self.create_button_redo()
        self.create_button_help()

        """ Compute slices to display """
        self.calculate_list_slices()

        """ Variable to check if all slices have been processed """
        self.setup_intensity()

        """ Manage closure of viewer"""
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

    def declaration_global_variables_general(self,orientation_subplot):
        self.help_web_adress='https://sourceforge.net/p/spinalcordtoolbox/wiki/Home/'
        self.orientation = {'ax': 1, 'cor': 2, 'sag': 3}
        self.primary_subplot = orientation_subplot[0]
        self.secondary_subplot = orientation_subplot[1]
        self.dic_axis_buttons={}
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
            return('v')
        else:
            return('h')

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

    def check_point_is_valid(self,target_point):
        if(self.is_point_in_image(target_point)):
            return True
        else:
            self.update_title_text_general('warning_selected_point_not_in_image')
            return False

    def update_title_text_general(self,key):
        if(key=='ready_to_save_and_quit'):
            title_obj = self.windows[0].axes.set_title('You can save and quit. \n')
            plt.setp(title_obj, color='g')

        elif(key=='warning_all_slices_are_done_already'):
            title_obj = self.windows[0].axes.set_title('You have processed all slices \n'
                                                       'If you made a mistake please use \'Redo\' \n'
                                                       'Otherwise, you can save and quit. \n')
            plt.setp(title_obj, color='g')

        elif(key=='warning_redo_beyond_first_dot'):
            title_obj = self.windows[0].axes.set_title('Please, place your first dot. \n')
            plt.setp(title_obj, color='r')

        elif(key=='warning_skip_not_defined'):
            title_obj = self.windows[0].axes.set_title('This option is not used in Manual Mode. \n')
            plt.setp(title_obj, color='r')

        elif(key=='warning_selected_point_not_in_image'):
            title_obj = self.windows[0].axes.set_title('The point you selected in not in the image. Please try again.')
            plt.setp(title_obj, color='r')

        self.windows[0].draw()

    def is_there_next_slice(self):
        if self.current_slice < len(self.list_slices):
            return True
        else:
            self.update_title_text('ready_to_save_and_quit')
            return False

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
            point[self.orientation[self.primary_subplot]-1] = self.list_slices[self.current_slice]
            for window in self.windows:
                if window is plot:
                    window.update_slice(point, data_update=False)
                else:
                    window.update_slice(point, data_update=True)
                    self.draw_points(window, self.current_point.x)
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

    def create_button_redo(self):
        ax = plt.axes([0.70, 0.90, 0.1, 0.075])
        self.dic_axis_buttons['redo']=ax
        button_help = Button(ax, 'Redo')
        self.fig.canvas.mpl_connect('button_press_event', self.press_redo)

    def create_button_save_and_quit(self):
        ax = plt.axes([0.81, 0.90, 0.1, 0.075])
        self.dic_axis_buttons['save_and_quit']=ax
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
            self.closed=True
            plt.close('all')

    def press_redo(self, event):
        if event.inaxes == self.dic_axis_buttons['redo']:
            if self.current_slice>0:
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
        super(ClickViewer, self).start()
        return self.list_points_useful_notation

    def are_all_slices_done(self):
        if self.current_slice < len(self.list_slices):
            return False
        else:
            self.update_title_text('warning_all_slices_are_done_already')
            return True

    def press_help(self, event):
        if event.inaxes == self.dic_axis_buttons['help']:
            import webbrowser
            webbrowser.open(self.help_web_adress, new=0, autoraise=True)

    def create_button_help(self):
        ax = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.dic_axis_buttons['help']=ax
        button_help = Button(ax, 'Help')
        self.fig.canvas.mpl_connect('button_press_event', self.press_help)

class ClickViewerPropseg(ClickViewer):

    def __init__(self,
                 list_images,
                 visualization_parameters=None,
                 orientation_subplot=['ax', 'sag'],
                 input_type='centerline'):

        ClickViewer.__init__(self,list_images,
                 visualization_parameters=visualization_parameters,
                 orientation_subplot=orientation_subplot,
                 input_type=input_type)

        self.declaration_global_variables_specific()
        self.update_title_text('init')

        """ Create Buttons"""
        self.create_button_skip()
        self.create_button_auto_manual()

    def on_release(self, event, plot=None):
        """
        This subplot refers to the secondary window. It captures event "release"
        :param event:
        :param plot:
        :return:
        """
        if event.button == 1 and event.inaxes == plot.axes and plot.view == self.orientation[self.secondary_subplot]:
            point = [self.current_point.x, self.current_point.y, self.current_point.z]
            if not self.bool_enable_custom_points:
                point[self.orientation[self.primary_subplot] - 1] = self.list_slices[self.current_slice]
            for window in self.windows:
                if window is plot:
                    window.update_slice(point, data_update=False)
                else:
                    window.update_slice(point, data_update=True)
                    self.draw_points(window, self.current_point.x)
        return

    def declaration_global_variables_specific(self):
        self.bool_enable_custom_points = False
        self.help_web_adress= 'https://sourceforge.net/p/spinalcordtoolbox/wiki/correction_PropSeg/attachment/propseg_viewer.png'

    def create_button_skip(self):
        ax = plt.axes([0.59, 0.90, 0.1, 0.075])
        self.dic_axis_buttons['skip']=ax
        button_help = Button(ax, 'Skip')
        self.fig.canvas.mpl_connect('button_press_event', self.press_skip)

    def press_skip(self, event):
        if event.inaxes == self.dic_axis_buttons['skip']:
            if not self.bool_enable_custom_points:
                if not self.are_all_slices_done():
                    self.current_slice += 1
                    if self.is_there_next_slice():
                        self.show_next_slice(self.windows[0],
                                         [self.current_point.x, self.current_point.y, self.current_point.z])
            else:
                self.update_title_text('warning_skip_not_defined')

    def create_button_auto_manual(self):
        ax = plt.axes([0.08, 0.90, 0.15, 0.075])
        self.dic_axis_buttons['choose_mode']=ax
        self.button_choose_auto_manual = Button(ax, 'Mode Auto')
        self.fig.canvas.mpl_connect('button_press_event', self.press_choose_mode)

    def update_title_text(self,key):

        if(key=='way_automatic_next_point'):
            title_obj = self.windows[0].axes.set_title('Please select a new point on slice ' +
                                                       str(self.list_slices[self.current_slice]) + '/' +
                                                       str(self.image_dim[
                                                               self.orientation[self.primary_subplot] - 1] - 1) + ' (' +
                                                       str(self.current_slice + 1) + '/' +
                                                        str(len(self.list_slices)) + ') \n'
                                                                                     'You may save and quit at any time \n')
            plt.setp(title_obj, color='k')

        elif(key=='way_custom_next_point'):
            title_obj = self.windows[0].axes.set_title(
                'You have made '+str(len(self.list_points))+ ' points. \n'
                                                             'You can save and quit at any time. \n')
            plt.setp(title_obj, color='k')

        elif(key=='way_custom_start'):
            title_obj = self.windows[0].axes.set_title('You have chosen Manual Mode\n '
                                                       'All previous data have been erased\n'
                                                       'Please choose the slices on the left pannel\n')
            plt.setp(title_obj, color='k')

        elif(key=='way_auto_start'):
            title_obj = self.windows[0].axes.set_title('You have chosen Auto Mode \n '
                                                       'All previous data have been erased \n '
                                                       'Please select a new point on slice \n ')
            plt._setp(title_obj,color='k')

        elif(key=='init'):
            title_obj = self.windows[0].axes.set_title('Mode Auto \n '
                                                       'Please click in the center of the spinal cord \n'
                                                       'If it is not visible yet, you may skip the first slices \n')
            plt._setp(title_obj,color='k')

        elif(key=='skipped_all_remaining_slices'):
            title_obj = self.windows[0].axes.set_title('You have skipped all remaining slices \n '
                                                       'You may now save and quit. \n')
            plt._setp(title_obj, color='g')


        else:
            self.update_title_text_general(key)

        self.windows[0].draw()

    def reset_useful_global_variables(self):
        self.windows[0].update_slice(0)
        # specialized for Click viewer
        self.list_points = []
        self.list_points_useful_notation = ''

        # compute slices to display
        self.list_slices = []

        self.current_slice = 0
        self.number_of_slices = 0
        self.gap_inter_slice = 0

        self.current_point = Coordinate([0, int(self.images[0].data.shape[1] / 2), int(self.images[0].data.shape[2] / 2)])
        self.calculate_list_slices()

        self.draw_points(self.windows[0],self.current_point.x)

    def press_choose_mode(self,event):
        if event.inaxes == self.dic_axis_buttons['choose_mode']:
            self.reset_useful_global_variables()
            self.bool_enable_custom_points=not self.bool_enable_custom_points

            if(self.bool_enable_custom_points):
                self.button_choose_auto_manual.label.set_text('Mode Manual')
                self.update_title_text('way_custom_start')
            else:
                self.button_choose_auto_manual.label.set_text('Mode Auto')
                self.update_title_text('way_auto_start')

    def on_press_main_window(self,event,plot):
        if self.bool_enable_custom_points:
            self.save_target_point_custom_way(event, plot)
        else:
            if not self.are_all_slices_done():
                self.save_target_point_not_custom_way(event,plot)

    def save_target_point_custom_way(self,event,plot):
        target_point = self.set_custom_target_points(event)
        if self.check_point_is_valid(target_point):
            self.list_points.append(target_point)
            point = [target_point.x,target_point.y,target_point.z]
            self.add_dot_to_current_slice(plot, point)

    def save_target_point_not_custom_way(self,event,plot):
        target_point = self.set_not_custom_target_points(event)
        if self.check_point_is_valid(target_point):
            self.list_points.append(target_point)
            point = [self.current_point.x, self.current_point.y, self.current_point.z]
            self.current_slice += 1
            if self.is_there_next_slice():
                self.show_next_slice(plot,point)

    def add_dot_to_current_slice(self,plot,point):
        self.draw_points(self.windows[0], self.current_point.x)
        self.windows[0].update_slice(point, data_update=True)
        self.update_title_text('way_custom_next_point')
        plot.draw()

    def show_next_slice(self,plot,point):
        point[self.orientation[self.secondary_subplot] - 1] = self.list_slices[self.current_slice]
        self.current_point = Coordinate(point)
        self.windows[1].update_slice([point[2], point[0], point[1]], data_update=False)
        self.windows[0].update_slice(self.list_slices[self.current_slice])
        self.update_title_text('way_automatic_next_point')
        plot.draw()

    def on_press_secondary_window(self,event,plot):
        is_in_axes = False
        for window in self.windows:
            if event.inaxes == window.axes:
                is_in_axes = True
        if not is_in_axes:
            return

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

    def press_redo(self, event):
        if event.inaxes == self.dic_axis_buttons['redo']:
            if (len(self.list_points) > 0   or self.current_slice>0):
                if not self.bool_enable_custom_points:
                    self.current_slice += -1
                self.remove_last_dot()
                self.update_ui_after_redo()
            else:
                self.update_title_text('warning_redo_beyond_first_dot')

    def show_previous_slice_in_custom(self):
        point = self.list_points[-1]
        self.windows[1].update_slice([point.x, point.y, point.z], data_update=False)
        self.windows[0].update_slice(point.x)
        self.current_point=Coordinate([point.x, point.y, point.z])
        self.draw_points(self.windows[0], point.x)
        return point


    def update_ui_after_redo(self):
        if not self.bool_enable_custom_points:
            self.update_title_text('way_automatic_next_point')
            self.show_next_slice(self.windows[0], [self.current_point.x, self.current_point.y, self.current_point.z])
            self.draw_points(self.windows[0], self.current_point.x)
        else:
            self.update_title_text('way_custom_next_point')
            if self.list_points :
                #self.show_previous_slice_in_custom() Option to jump to last slice when redoing in custom mode.
                self.draw_points(self.windows[0], self.current_point.x)
            else:
                self.draw_points(self.windows[0], self.current_point.x)

class ClickViewerLabelVertebrae(ClickViewer):

    def __init__(self,
                 list_images,
                 visualization_parameters=None,
                 orientation_subplot=['ax', 'sag'],
                 input_type='centerline'):

        ClickViewer.__init__(self,list_images,
                 visualization_parameters=visualization_parameters,
                 orientation_subplot=orientation_subplot,
                 input_type=input_type)

        self.update_title_text('init')
        self.bool_ignore_warning_about_leaving=False
        self.help_web_adress='https://sourceforge.net/p/spinalcordtoolbox/wiki/sct_label_vertebrae/attachment/label_vertebrae_viewer.png'
        self.windows[1].axes.set_title(' \n')

        """ Create Buttons"""
        self.create_button_redo()

    def update_title_text(self,key):

        if(key=='init'):
            title_obj = self.windows[0].axes.set_title( 'Please click on posterior edge of  \n'
                  'C2/C3 intervertebral disk (label=3) \n')
            plt._setp(title_obj,color='k')

        elif(key=='redo_done'):
            title_obj = self.windows[0].axes.set_title( 'Previous dot erased \n'
                                                        'Please click on posterioer edge of \n'
                                                        'intervertebral disc C2-C3 \n')
            plt._setp(title_obj,color='k')

        elif(key=='impossible_to_leave'):
            title_obj = self.windows[0].axes.set_title('Please confirm : You have not drawn the dot \n'
                                                       'If you leave now, the software will crash\n')
            plt._setp(title_obj, color='r')

        else:
            self.update_title_text_general(key)

        self.windows[0].draw()

    def on_press_main_window(self,event,plot):
        if not self.are_all_slices_done():
            target_point = self.set_target_point(event)
            if self.check_point_is_valid(target_point):
                self.list_points.append(target_point)
                self.current_slice = 1
                self.add_dot_to_current_slice(plot, target_point)
                self.update_title_text('ready_to_save_and_quit')

    def add_dot_to_current_slice(self,plot,point):
        self.draw_points(self.windows[0], self.current_point.z)
        self.windows[0].update_slice(point, data_update=True)
        plot.draw()

    def on_motion(self,event,plot):
        pass

    def on_press_secondary_window(self,event,plot):
        pass
        '''
        is_in_axes = False
        for window in self.windows:
            if event.inaxes == window.axes:
                is_in_axes = True
        if not is_in_axes:
            return

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
        '''

    def on_press(self, event, plot=None):
        if event.inaxes and plot.view == self.orientation[self.primary_subplot]:
            self.on_press_main_window(event,plot)
        elif event.inaxes and plot.view == self.orientation[self.secondary_subplot]:
            self.on_press_secondary_window(event,plot)

    def draw_points(self, window, current_slice):
        if window.view == self.orientation[self.primary_subplot]:
            x_data, y_data = [], []
            for pt in self.list_points:
                if pt.z == current_slice:
                    y_data.append(pt.y + self.offset[1])
                    x_data.append(pt.x + self.offset[0])
            self.plot_points.set_xdata(y_data)
            self.plot_points.set_ydata(x_data)
            self.fig.canvas.draw()

    def set_target_point(self,event):
        if self.primary_subplot == 'ax':
            return( Coordinate([int(self.list_slices[0]), int(event.ydata) - self.offset[1], int(event.xdata) - self.offset[2], 1]))
        elif self.primary_subplot == 'cor':
            return ( Coordinate([int(event.ydata) - self.offset[0], int(self.list_slices[0]), int(event.xdata) - self.offset[2], 1]) )
        elif self.primary_subplot == 'sag':
            return ( Coordinate([int(event.ydata) - self.offset[0], int(event.xdata) - self.offset[1], int(self.list_slices[0]), 1]) )

    def press_save_and_quit(self, event):
        if event.inaxes == self.dic_axis_buttons['save_and_quit']:
            if len(self.list_points)>0 or self.bool_ignore_warning_about_leaving:
                self.save_data()
                self.closed = True
                plt.close('all')
            else:
                self.update_title_text('impossible_to_leave')
                self.bool_ignore_warning_about_leaving=True

class ClickViewerRegisterToTemplate(ClickViewer):

    def __init__(self,
                 list_images,
                 visualization_parameters=None,
                 orientation_subplot=['ax', 'sag'],
                 input_type='centerline',
                 first_label=4,
                 last_label=10):

        ClickViewer.__init__(self,list_images,
                 visualization_parameters=visualization_parameters,
                 orientation_subplot=orientation_subplot,
                 input_type=input_type)

        self.number_of_dots_final=2
        self.current_dot_number=0
        self.dic_message_labels=self.define_dic_message_labels()
        self.list_current_wanted_labels=self.control_list_wanted_label(first_label,last_label)
        self.update_title_text(str(self.current_dot_number))
        self.help_web_adress='https://sourceforge.net/p/spinalcordtoolbox/wiki/sct_register_to_template/attachment/sct_register_to_template.png'
        self.windows[1].axes.set_title(' \n')

        """ Create Buttons"""
        self.create_button_redo()
        self.create_button_lower_label()
        self.create_button_higher_label()

    def control_list_wanted_label(self,first_label,last_label):
        a=self.correct_label_choice(first_label)
        b=self.correct_label_choice(last_label)
        if b>a:
            return [a,b]
        elif b<a:
            sct.printv('Warning : You can not have the first label higher than the last one : parameter corrected.',
                       True, 'warning')
            return [b,a]
        else:
            sct.printv('Warning : You can not have the same first and last label : parameter ignored.', True, 'warning')
            return [4,10]

    def correct_label_choice(self,i):
        if i>20:
            sct.printv('Warning : You can not choose this label : parameter ignored.', True, 'warning')
            return 3
        elif i==2 or i==1:
            return 3
        else:
            return i+1

    def define_dic_message_labels(self):
        dic={'1':'Please click on anterior base \n'
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

    def update_title_text(self,key):

        if(key=='0'):
            title_obj = self.windows[0].axes.set_title(self.dic_message_labels[str(self.list_current_wanted_labels[self.current_dot_number])])
            plt._setp(title_obj,color='k')

        elif(key=='1'):
            title_obj = self.windows[0].axes.set_title(self.dic_message_labels[str(self.list_current_wanted_labels[self.current_dot_number])])
            plt._setp(title_obj,color='k')

        elif(key=='redo_done'):
            title_obj = self.windows[0].axes.set_title( 'Previous dot erased \n')
            plt._setp(title_obj,color='k')

        elif(key=='cant_go_higher'):
            title_obj = self.windows[0].axes.set_title( 'You can\'t choose a higher label \n')
            plt._setp(title_obj,color='r')

        elif(key=='cant_go_lower'):
            title_obj = self.windows[0].axes.set_title( 'You can\'t choose a lower label \n')
            plt._setp(title_obj,color='r')

        else:
            self.update_title_text_general(key)

        self.windows[0].draw()

    def on_press_main_window(self,event,plot):
        if not self.are_all_slices_done():
            target_point = self.set_target_point(event)
            if self.check_point_is_valid(target_point):
                self.list_points.append(target_point)
                self.current_dot_number += 1
                self.add_dot_to_current_slice(plot, target_point)
                self.is_there_next_slice()

    def on_motion(self,event,plot):
        pass

    def on_press_secondary_window(self,event,plot):
        pass
        '''
        is_in_axes = False
        for window in self.windows:
            if event.inaxes == window.axes:
                is_in_axes = True
        if not is_in_axes:
            return

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
        '''

    def on_press(self, event, plot=None):
        if event.inaxes and plot.view == self.orientation[self.primary_subplot]:
            self.on_press_main_window(event,plot)
        elif event.inaxes and plot.view == self.orientation[self.secondary_subplot]:
            self.on_press_secondary_window(event,plot)

    def define_translate_dic(self):
        dic={'1':50,
             '2':49,
             '3':1,
             '4':3,}
        for ii in range (5,len(self.dic_message_labels)+1):
            dic[str(ii)]=ii-1
        return dic

    def set_target_point(self,event):
        dic_translate_labels=self.define_translate_dic()
        if self.primary_subplot == 'ax':
            return( Coordinate([int(self.list_slices[0]),
                                int(event.ydata) - self.offset[1],
                                int(event.xdata) - self.offset[2],
                                dic_translate_labels[str(self.list_current_wanted_labels[self.current_dot_number])]
            ] ) )
        elif self.primary_subplot == 'cor':
            return ( Coordinate([int(event.ydata) - self.offset[0],
                                 int(self.list_slices[0]),
                                 int(event.xdata) - self.offset[2],
                                 dic_translate_labels[str(self.list_current_wanted_labels[self.current_dot_number])]
                                 ]))
        elif self.primary_subplot == 'sag':
            return ( Coordinate([int(event.ydata) - self.offset[0],
                                 int(event.xdata) - self.offset[1],
                                 int(self.list_slices[0]),
                                 dic_translate_labels[str(self.list_current_wanted_labels[self.current_dot_number])]
                                 ]))

    def are_all_slices_done(self):
        if self.current_dot_number < self.number_of_dots_final:
            return False
        else:
            self.update_title_text('warning_all_slices_are_done_already')
            return True

    def update_second_label_if_necessary(self):
        if self.list_current_wanted_labels[0]>=self.list_current_wanted_labels[1]:
            self.list_current_wanted_labels[1]=self.list_current_wanted_labels[0]+1

    def is_there_next_slice(self):
        if self.current_dot_number < self.number_of_dots_final:
            self.update_second_label_if_necessary()
            self.update_title_text(str(self.current_dot_number))
            return True
        else:
            self.update_title_text('ready_to_save_and_quit')
            return False

    def add_dot_to_current_slice(self,plot,point):
        self.draw_points(self.windows[0], self.current_point.z)
        self.windows[0].update_slice(point, data_update=True)
        plot.draw()

    def draw_points(self, window, current_slice):
        if window.view == self.orientation[self.primary_subplot]:
            x_data, y_data = [], []
            for pt in self.list_points:
                if pt.z == current_slice:
                    y_data.append(pt.y + self.offset[1])
                    x_data.append(pt.x + self.offset[0])
            self.plot_points.set_xdata(y_data)
            self.plot_points.set_ydata(x_data)
            self.fig.canvas.draw()

    def press_redo(self, event):
        if event.inaxes == self.dic_axis_buttons['redo']:
            if self.current_dot_number>0:
                self.current_dot_number += -1
                self.windows[0].update_slice(self.list_slices[self.current_slice])
                self.remove_last_dot()
                self.update_ui_after_redo()
                self.update_title_text(str(self.current_dot_number))
            else:
                self.update_title_text('warning_redo_beyond_first_dot')

    def is_it_possible_to_get_lower(self):
        if not self.current_dot_number:
            if self.list_current_wanted_labels[0]<20:
                return True
            else:
                self.update_title_text('cant_go_lower')
                return False
        else:
            if self.list_current_wanted_labels[1]<21:
                return True
            else:
                self.update_title_text('cant_go_lower')
                return False

    def is_it_possible_to_get_higher(self):
        if not self.current_dot_number:
            if self.list_current_wanted_labels[0]>3:
                return True
            else:
                self.update_title_text('cant_go_higher')
                return False
        else:
            if self.list_current_wanted_labels[1]>self.list_current_wanted_labels[0]+1:
                return True
            else:
                self.update_title_text('cant_go_higher')
                return False

    def create_button_higher_label(self):
        ax = plt.axes([0.08, 0.90, 0.15, 0.075])
        self.dic_axis_buttons['higher_label']=ax
        self.button_choose_auto_manual = Button(ax, 'Pick \n '
                                                    'Higher Label')
        self.fig.canvas.mpl_connect('button_press_event', self.press_higher_label)

    def press_higher_label(self,event):
        if event.inaxes == self.dic_axis_buttons['higher_label']:
            if self.is_it_possible_to_get_higher():
                self.list_current_wanted_labels[self.current_dot_number]+= -1
                self.update_title_text('0')

    def create_button_lower_label(self):
        ax = plt.axes([0.25, 0.90, 0.15, 0.075])
        self.dic_axis_buttons['lower_label']=ax
        self.button_choose_auto_manual = Button(ax, 'Pick \n '
                                                    'Lower Label')
        self.fig.canvas.mpl_connect('button_press_event', self.press_lower_label)

    def press_lower_label(self,event):
        if event.inaxes == self.dic_axis_buttons['lower_label']:
            if self.is_it_possible_to_get_lower():
                self.list_current_wanted_labels[self.current_dot_number]+= +1
                self.update_title_text('0')

class ClickViewerGroundTruth(ClickViewer):

    def __init__(self,
                 list_images,
                 first_label=50,
                 visualization_parameters=None,
                 orientation_subplot=['ax', 'sag'],
                 input_type='centerline'):

        ClickViewer.__init__(self,list_images,
                 visualization_parameters=visualization_parameters,
                 orientation_subplot=orientation_subplot,
                 input_type=input_type)



        if self.check_first_label(first_label):
            self.first_label=self.correct_label_choice(first_label)

        self.dic_message_labels=self.define_dic_message_labels()

        self.current_dot_number=1
        self.number_of_slices_to_mean=3

        self.number_of_dots_final = len(self.dic_message_labels)
        self.dic_translate_labels=self.define_translate_dic()
        self.update_title_text(str(self.current_dot_number))
        self.update_title_text('new_number_of_slice_to_mean')

        """ Create Buttons"""
        self.create_button_redo()
        self.create_button_skip()
        self.create_button_mean_more()
        self.create_button_mean_less()
        self.create_button_reset_mean()

        self.skip_until_first_slice()
        self.update_title_text('current_dot_to_draw')
        self.show_image_mean()

    def correct_label_choice(self,i):
        if i==50:
            return 1
        elif i==49:
            return 2
        elif i==2 or i==1:
            return 3
        else:
            return i+1

    def show_image_mean(self):
        if self.check_averaging_possible():
            imMean = self.calc_mean_slices()
            self.windows[0].figs[0].set_data(imMean)
            self.windows[0].figs[0].figure.canvas.draw()

    def check_averaging_possible(self):
        data=self.images[0].data
        if self.current_point.z-(self.number_of_slices_to_mean-1)/2 <1 or self.current_point.z+(self.number_of_slices_to_mean-1)/2+1 > data.shape[2]-1:
            self.number_of_slices_to_mean+= -2
            self.show_image_mean()
            self.update_title_text('warning_averaging_excedes_data_shape')
            return False
        else:
            return True

    def calc_mean_slices(self):
        data=self.images[0].data
        dataRacc=data[:,:,self.current_point.z-(self.number_of_slices_to_mean-1)/2:self.current_point.z+(self.number_of_slices_to_mean-1)/2+1]
        imMean=np.empty([data.shape[0],data.shape[1]])
        for ii in range (0,data.shape[0]):
            for jj in range (0,data.shape[1]):
                imMean[ii,jj]=np.mean(dataRacc[ii,jj,:])
        return imMean

    def check_first_label(self,i):
        if i in range (1,27) or i in range (49,51):
            return True
        else:
            sct.printv('Warning : You have selected a wrong number for \'-start\' : starting from label 50.',True,'warning')
            self.first_label=1
            return False

    def skip_until_first_slice(self):
        self.list_points.append(
            Coordinate([-1, -1, -1, self.dic_translate_labels[str(self.current_dot_number)]]))
        for ilabels in range(1, self.first_label-1):
            self.current_dot_number += 1
            self.list_points.append(
                Coordinate([-1, -1, -1, self.dic_translate_labels[str(self.current_dot_number)]]))
        self.current_dot_number += 1

    def define_dic_message_labels(self):
        dic={'1':'Please click on anterior base \n'
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

    def update_title_text(self,key):

        if(key=='current_dot_to_draw'):
            title_obj = self.windows[0].axes.set_title(self.dic_message_labels[str(self.current_dot_number)]
                                                       + ' ( ' + str(self.current_dot_number) + ' / ' + str(self.number_of_dots_final) + ' )\n')
            plt._setp(title_obj,color='k')

        elif(key=='redo_done'):
            title_obj = self.windows[0].axes.set_title( 'Previous dot erased \n'
                                                        'Please click at intervertebral disc C2-C3 \n')
            plt._setp(title_obj,color='k')

        elif(key=='confirm_to_quit'):
            title_obj = self.windows[0].axes.set_title( 'All unprocessed labels have been skipped \n'
                                                        'Please confirm you wish to leave \n')
            plt._setp(title_obj,color='r')

        elif(key=='all_remaining_labels_skipped'):
            title_obj = self.windows[0].axes.set_title( 'All unprocessed labels have been skipped \n'
                                                        'You may now save and quit \n')
            plt._setp(title_obj,color='g')

        elif(key=='warning_cannot_mean_fewer_slices'):
            title_obj = self.windows[1].axes.set_title( 'You can\'t average\n fewer slices. \n')
            plt._setp(title_obj,color='r')

        elif(key=='warning_cannot_mean_more_slices'):
            title_obj = self.windows[1].axes.set_title( 'You can\'t average\n more slices. \n')
            plt._setp(title_obj,color='r')

        elif(key=='new_number_of_slice_to_mean'):
            if(self.number_of_slices_to_mean > 1 ):
                title_obj = self.windows[1].axes.set_title( 'Select sagital slice \n'
                                                            'Averaging across ' + str(self.number_of_slices_to_mean) + ' slices. \n')
            else:
                title_obj = self.windows[1].axes.set_title( 'The main picture does not \n'
                                                            'average slices. \n')
            plt._setp(title_obj,color='k')

        elif(key=='warning_averaging_excedes_data_shape'):
            title_obj = self.windows[0].axes.set_title( 'You are too close to the border \n'
                                                        'The number of slice you can average is ' + str(self.number_of_slices_to_mean) +'.\n')
            plt._setp(title_obj,color='r')

        elif(key=='reset_average_parameters'):
            title_obj = self.windows[0].axes.set_title( 'You have reseted the parameters\n'
                                                        'The main picture is the average of ' + str(self.number_of_slices_to_mean) +' slices.\n')
            plt._setp(title_obj,color='k')

        else:
            self.update_title_text_general(key)

        self.windows[0].draw()

    def on_release(self, event, plot=None):
        if event.button == 1 and event.inaxes == plot.axes and plot.view == self.orientation[self.secondary_subplot]:
            point = [self.current_point.x, self.current_point.y, self.current_point.z]
            self.update_pictures_in_windows(plot,point)
            self.show_image_mean()
            self.draw_points(self.windows[0],self.current_point.z)

    def update_pictures_in_windows(self,plot,point):
        for window in self.windows:
            if window is plot:
                window.update_slice(point, data_update=False)
            else:
                window.update_slice(point, data_update=True)
                self.draw_points(window, self.current_point.x)

    def create_button_skip(self):
        ax = plt.axes([0.59, 0.90, 0.1, 0.075])
        self.dic_axis_buttons['skip']=ax
        button_help = Button(ax, 'Skip')
        self.fig.canvas.mpl_connect('button_press_event', self.press_skip)

    def press_skip(self, event):
        if event.inaxes == self.dic_axis_buttons['skip']:
            if not self.are_all_slices_done():
                self.current_dot_number += 1
                self.list_points.append( Coordinate([-1,-1,-1,self.dic_translate_labels[str(self.current_dot_number ) ] ] ) )
                self.is_there_next_slice()
            else:
                self.update_title_text('warning_all_slices_are_done_already')

    def on_press_main_window(self,event,plot):
        if not self.are_all_slices_done():
            target_point = self.set_target_point(event)
            if self.check_point_is_valid(target_point):
                self.list_points.append(target_point)
                self.current_dot_number += 1
                self.add_dot_to_current_slice(plot, target_point)
                self.is_there_next_slice()

    def add_dot_to_current_slice(self,plot,point):
        self.draw_points(self.windows[0], self.current_point.z)
        self.windows[0].update_slice(point, data_update=True)
        plot.draw()

    def on_press_secondary_window(self,event,plot):
        is_in_axes = False
        for window in self.windows:
            if event.inaxes == window.axes:
                is_in_axes = True
        if not is_in_axes:
            return

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
        if event.inaxes and plot.view == self.orientation[self.primary_subplot]:
            self.on_press_main_window(event,plot)
        elif event.inaxes and plot.view == self.orientation[self.secondary_subplot]:
            self.on_press_secondary_window(event,plot)

    def draw_points(self, window, current_slice):
        if window.view == self.orientation[self.primary_subplot]:
            x_data, y_data = [], []
            for pt in self.list_points:
                if pt.z == current_slice:
                    y_data.append(pt.y + self.offset[1])
                    x_data.append(pt.x + self.offset[0])
            self.plot_points.set_xdata(y_data)
            self.plot_points.set_ydata(x_data)
            self.fig.canvas.draw()

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
            self.draw_points(self.windows[0],self.current_point.z)
        return

    def define_translate_dic(self):
        dic={'1':50,
             '2':49,
             '3':1,
             '4':3,}
        for ii in range (5,len(self.dic_message_labels)+1):
            dic[str(ii)]=ii-1
        return dic

    def set_target_point(self,event):
        dic_translate_labels=self.define_translate_dic()
        if self.primary_subplot == 'ax':
            return( Coordinate([int(self.current_point.x),
                                int(event.ydata) - self.offset[1],
                                int(event.xdata) - self.offset[2],
                                dic_translate_labels[str(self.current_dot_number)]
            ] ) )
        elif self.primary_subplot == 'cor':
            return ( Coordinate([int(event.ydata) - self.offset[0],
                                 int(self.current_point.y),
                                 int(event.xdata) - self.offset[2],
                                 dic_translate_labels[str(self.current_dot_number)]
                                 ]))
        elif self.primary_subplot == 'sag':
            return ( Coordinate([int(event.ydata) - self.offset[0],
                                 int(event.xdata) - self.offset[1],
                                 int(self.current_point.z),
                                 dic_translate_labels[str(self.current_dot_number)]
                                 ]))

    def are_all_slices_done(self):
        if self.current_dot_number < self.number_of_dots_final:
            return False
        else:
            self.update_title_text('warning_all_slices_are_done_already')
            return True

    def is_there_next_slice(self):
        if self.current_dot_number < self.number_of_dots_final:
            self.update_title_text('current_dot_to_draw')
            return True
        else:
            self.update_title_text('ready_to_save_and_quit')
            return False

    def press_redo(self, event):
        if event.inaxes == self.dic_axis_buttons['redo']:
            if self.current_dot_number>1:
                self.current_dot_number += -1
                self.list_points=self.list_points[0:len(self.list_points)-1]
                self.draw_points(self.windows[0],self.current_point.z)
                self.update_title_text('current_dot_to_draw')
            else:
                self.update_title_text('warning_redo_beyond_first_dot')

    def skip_all_remaining_labels(self):
        for ilabels in range (self.current_dot_number,self.number_of_dots_final):
            self.current_dot_number += 1
            self.list_points.append(Coordinate([-1, -1, -1, self.dic_translate_labels[str(self.current_dot_number)]]))

    def check_all_labels_are_done(self):
        if self.current_dot_number==self.number_of_dots_final:
            return True
        else:
            self.skip_all_remaining_labels()
            self.update_title_text('confirm_to_quit')

    def print_useful_points(self):
        sct.printv('Labels positions are : ')
        sct.printv(self.list_points_useful_notation)

    def press_save_and_quit(self, event):
        if event.inaxes == self.dic_axis_buttons['save_and_quit']:
            if self.check_all_labels_are_done():
                self.save_data()
                self.print_useful_points()
                self.closed = True
                plt.close('all')

    def create_button_mean_more(self):
        ax = plt.axes([0.08, 0.90, 0.15, 0.075])
        self.dic_axis_buttons['mean_more']=ax
        self.button_choose_auto_manual = Button(ax, 'Average \n '
                                                    'more slices')
        self.fig.canvas.mpl_connect('button_press_event', self.press_mean_more)

    def press_mean_more(self,event):
        if event.inaxes == self.dic_axis_buttons['mean_more']:
            if self.number_of_slices_to_mean < 9 :
                self.number_of_slices_to_mean+= 2
                self.update_title_text('new_number_of_slice_to_mean')
                self.show_image_mean()
            else:
                self.update_title_text('warning_cannot_mean_more_slices')

    def create_button_mean_less(self):
        ax = plt.axes([0.25, 0.90, 0.15, 0.075])
        self.dic_axis_buttons['mean_less']=ax
        self.button_choose_auto_manual = Button(ax, 'Average \n '
                                                    'fewer slices')
        self.fig.canvas.mpl_connect('button_press_event', self.press_mean_less)

    def press_mean_less(self,event):
        if event.inaxes == self.dic_axis_buttons['mean_less']:
            if self.number_of_slices_to_mean >1 :
                self.number_of_slices_to_mean+= -2
                self.update_title_text('new_number_of_slice_to_mean')
                self.show_image_mean()
            else:
                self.update_title_text('warning_cannot_mean_fewer_slices')

    def create_button_reset_mean(self):
        ax = plt.axes([0.12, 0.82, 0.24, 0.075])
        self.dic_axis_buttons['reset_mean']=ax
        self.button_choose_auto_manual = Button(ax, 'Reset \n'
                                                    ' average settings')
        self.fig.canvas.mpl_connect('button_press_event', self.press_reset_settings)

    def press_reset_settings(self, event):
        if event.inaxes == self.dic_axis_buttons['reset_mean']:
            ''' Reset central image '''
            point = [self.current_point.x, self.current_point.y, self.current_point.z]
            point[self.orientation[self.primary_subplot] - 1] = self.list_slices[self.current_slice]
            self.update_pictures_in_windows(self.windows[1],point)

            self.current_point = Coordinate(
                [int(self.images[0].data.shape[0] / 2), int(self.images[0].data.shape[1] / 2),
                 int(self.images[0].data.shape[2] / 2)])
            self.number_of_slices_to_mean=3
            self.show_image_mean()
            self.draw_points(self.windows[0],self.current_point.z)
            self.update_title_text('reset_average_parameters')