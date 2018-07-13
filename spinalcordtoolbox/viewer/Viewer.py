class Viewer(object):
    def __init__(self, list_input, visualization_parameters=None):
        self.images = self.keep_only_images(list_input)
        self.im_params = visualization_parameters

        """ Initialisation of plot """
        self.fig = plt.figure(figsize=(8, 8))
        self.fig.subplots_adjust(bottom=0.1, left=0.1)
        self.fig.patch.set_facecolor('lightgrey')

        """ Pad the image so that it is square in axial view (useful for zooming) """
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

    def keep_only_images(self,list_input):
        # TODO: check same space
        # TODO: check if at least one image
        images=[]
        for im in list_input:
            if isinstance(im, Image):
                images.append(im)
            else:
                sct.printv("Error, one of the images is actually not an image...")
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
