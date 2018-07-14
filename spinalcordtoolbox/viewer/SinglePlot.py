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
            (my_cmap,my_interpolation,my_alpha)=self.set_image_parameters(im_params,i,cm)
            my_cmap.set_under('b', alpha=0)
            self.figs.append(self.axes.imshow(data_to_display, aspect=self.aspect_ratio, alpha=my_alpha))
            self.figs[-1].set_cmap(my_cmap)
            self.figs[-1].set_interpolation(my_interpolation)

        self.axes.set_axis_bgcolor('black')
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        self.draw_line(display_cross)


    def draw_line(self,display_cross):
        from matplotlib.lines import Line2D
        
        self.line_horizontal = Line2D(self.cross_to_display[1][1], self.cross_to_display[1][0], color='white')
        self.line_vertical = Line2D(self.cross_to_display[0][1], self.cross_to_display[0][0], color='white')
        if 'h' in display_cross:
            self.axes.add_line(self.line_horizontal)
        if 'v' in display_cross:
            self.axes.add_line(self.line_vertical)

    def set_image_parameters(self,im_params,i,cm):
        if str(i) in im_params.images_parameters:
            from copy import copy
            return(copy(cm.get_cmap(im_params.images_parameters[str(i)].cmap)),im_params.images_parameters[str(i)].interp,float(im_params.images_parameters[str(i)].alpha))
        else:
            return (cm.get_cmap('gray'), 'nearest', 1.0)

    def set_data_to_display(self,image):
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
                sct.printv(event.button)

            self.update_xy_lim(x_center=xdata, y_center=ydata,
                               x_scale_factor=scale_factor, y_scale_factor=scale_factor,
                               zoom=True)

        return
