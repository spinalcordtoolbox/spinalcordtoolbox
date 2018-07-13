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
