#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

import sys
import os
import json
import logging
import warnings
import datetime
import io
from string import Template

warnings.filterwarnings("ignore")

import numpy as np

import skimage
import skimage.io
import skimage.exposure

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as color

import sct_utils as sct
from spinalcordtoolbox.image import Image
import spinalcordtoolbox.reports.slice as qcslice

import portalocker

logger = logging.getLogger("sct.{}".format(__file__))


class QcImage(object):
    """
    Class used to create a .png file from a 2d image produced by the class "Slice"
    """
    _labels_regions = {'PONS': 50, 'MO': 51,
                       'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
                       'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14, 'T8': 15, 'T9': 16,
                       'T10': 17, 'T11': 18, 'T12': 19,
                       'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24,
                       'S1': 25, 'S2': 26, 'S3': 27, 'S4': 28, 'S5': 29,
                       'Co': 30}
    _color_bin_green = ["#ffffff", "#00ff00"]
    _color_bin_red = ["#ffffff", "#ff0000"]
    _labels_color = ["#04663c", "#ff0000", "#50ff30",
                     "#ed1339", "#ffffff", "#e002e8",
                     "#ffee00", "#00c7ff", "#199f26",
                     "#563691", "#848545", "#ce2fe1",
                     "#2142a6", "#3edd76", "#c4c253",
                     "#e8618a", "#3128a3", "#1a41db",
                     "#939e41", "#3bec02", "#1c2c79",
                     "#18584e", "#b49992", "#e9e73a",
                     "#3b0e6e", "#6e856f", "#637394",
                     "#36e05b", "#530a1f", "#8179c4",
                     "#e1320c", "#52a4df", "#000ab5",
                     "#4a4242", "#0b53a5", "#b49c19",
                     "#50e7a9", "#bf5a42", "#fa8d8e",
                     "#83839a", "#320fef", "#82ffbf",
                     "#360ee7", "#551960", "#11371e",
                     "#e900c3", "#a21360", "#58a601",
                     "#811c90", "#235acf", "#49395d",
                     "#9f89b0", "#e08e08", "#3d2b54",
                     "#7d0434", "#fb1849", "#14aab4",
                     "#a22abd", "#d58240", "#ac2aff"]
    # _seg_colormap = plt.cm.autumn

    def __init__(self, qc_report, interpolation, action_list, stretch_contrast=True,
                 stretch_contrast_method='contrast_stretching'):
        """

        Parameters
        ----------
        qc_report : QcReport
            The QC report object
        interpolation : str
            Type of interpolation used in matplotlib
        action_list : list of functions
            List of functions that generates a specific type of images
        stretch_contrast : adjust image so as to improve contrast
        stretch_contrast_method: {'contrast_stretching', 'equalized'}: Method for stretching contrast
        """
        self.qc_report = qc_report
        self.interpolation = interpolation
        self.action_list = action_list
        self._stretch_contrast = stretch_contrast
        self._stretch_contrast_method = stretch_contrast_method

    """
    action_list contain the list of images that has to be generated.
    It can be seen as "figures" of matplotlib to be shown
    Ex: if 'colorbar' is in the list, the process will generate a color bar in the "img" folder
    """

    def listed_seg(self, mask, ax):
        """Create figure with red segmentation. Common scenario."""
        img = np.rint(np.ma.masked_where(mask < 1, mask))
        ax.imshow(img,
                  cmap=color.ListedColormap(self._color_bin_red),
                  norm=color.Normalize(vmin=0, vmax=1),
                  interpolation=self.interpolation,
                  alpha=1,
                  aspect=float(self.aspect_mask))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def template(self, mask, ax):
        """Show template statistical atlas"""
        values = mask
        values[values < 0.5] = 0
        color_white = color.colorConverter.to_rgba('white', alpha=0.0)
        color_blue = color.colorConverter.to_rgba('blue', alpha=0.7)
        color_cyan = color.colorConverter.to_rgba('cyan', alpha=0.8)
        cmap = color.LinearSegmentedColormap.from_list('cmap_atlas',
                                                       [color_white, color_blue, color_cyan], N=256)
        ax.imshow(values,
                  cmap=cmap,
                  interpolation=self.interpolation,
                  aspect=self.aspect_mask)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def no_seg_seg(self, mask, ax):
        """Create figure with image overlay. Notably used by sct_registration_to_template"""
        ax.imshow(mask, cmap='gray', interpolation=self.interpolation, aspect=self.aspect_mask)
        self._add_orientation_label(ax)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def sequential_seg(self, mask, ax):
        values = np.ma.masked_equal(np.rint(mask), 0)
        ax.imshow(values,
                  cmap=self._seg_colormap,
                  interpolation=self.interpolation,
                  aspect=self.aspect_mask)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def label_vertebrae(self, mask, ax):
        """Draw vertebrae areas, then add text showing the vertebrae names"""
        from matplotlib import colors
        import scipy.ndimage
        img = np.rint(np.ma.masked_where(mask < 1, mask))
        ax.imshow(img,
                  cmap=colors.ListedColormap(self._labels_color),
                  norm=colors.Normalize(vmin=0, vmax=len(self._labels_color)),
                  interpolation=self.interpolation,
                  alpha=1,
                  aspect=float(self.aspect_mask))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        a = [0.0]
        data = mask
        for index, val in np.ndenumerate(data):
            if val not in a:
                a.append(val)
                index = int(val)
                if index in self._labels_regions.values():
                    color = self._labels_color[index]
                    y, x = scipy.ndimage.measurements.center_of_mass(np.where(data == val, data, 0))
                    # Draw text with a shadow
                    x += 10
                    label = list(self._labels_regions.keys())[list(self._labels_regions.values()).index(index)]
                    ax.text(x, y, label, color='black', clip_on=True)
                    x -= 0.5
                    y -= 0.5
                    ax.text(x, y, label, color=color, clip_on=True)

    def highlight_pmj(self, mask, ax):
        """Hook to show a rectangle where PMJ is on the slice"""
        import matplotlib.patches as patches
        y, x = np.where(mask == 50)
        img = np.full_like(mask, np.nan)
        ax.imshow(img, cmap='gray', alpha=0, aspect=float(self.aspect_mask))
        ax.text(x, y, 'X', color='lime', clip_on=True)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # def colorbar(self):
    #     fig = plt.figure(figsize=(9, 1.5))
    #     ax = fig.add_axes([0.05, 0.80, 0.9, 0.15])
    #     colorbar.ColorbarBase(ax, cmap=self._seg_colormap, orientation='horizontal')
    #     return '{}_colorbar'.format(self.qc_report.img_base_name)

    def __call__(self, func):
        """wrapped function (f).

        In this case, it is the "mosaic" or "single" methods of the class "Slice"

        Parameters
        ----------
        func : function
            The wrapped function
        """

        def wrapped_f(sct_slice, *args):
            """

            Parameters
            ----------
            sct_slice : spinalcordtoolbox.report.slice:Slice
            args : list

            Returns
            -------

            """
            self.qc_report.slice_name = sct_slice.get_name()

            # Get the aspect ratio (height/width) based on pixel size. Consider only the first 2 slices.
            aspect_img, self.aspect_mask = sct_slice.aspect()[:2]

            self.qc_report.make_content_path()
            logger.info('QC: %s with %s slice', func.__name__, sct_slice.get_name())

            img, mask = func(sct_slice, *args)

            if self._stretch_contrast:
                def equalized(a):
                    """
                    Perform histogram equalization using CLAHE

                    Notes:

                    - Image value range is preserved
                    - Workaround for adapthist artifact by padding (#1664)
                    """
                    winsize = 16
                    min_, max_ = a.min(), a.max()
                    b = (np.float32(a) - min_) / (max_ - min_)
                    b[b >= 1] = 1  # 1+eps numerical error may happen (#1691)

                    h, w = b.shape
                    h1 = (h + (winsize - 1)) // winsize * winsize
                    w1 = (w + (winsize - 1)) // winsize * winsize
                    if h != h1 or w != w1:
                        b1 = np.zeros((h1, w1), dtype=b.dtype)
                        b1[:h, :w] = b
                        b = b1
                    c = skimage.exposure.equalize_adapthist(b, kernel_size=(winsize, winsize))
                    if h != h1 or w != w1:
                        c = c[:h, :w]
                    return np.array(c * (max_ - min_) + min_, dtype=a.dtype)

                def contrast_stretching(a):
                    p2, p98 = np.percentile(a, (2, 98))
                    return skimage.exposure.rescale_intensity(a, in_range=(p2, p98))

                func_stretch_contrast = {'equalized': equalized,
                                         'contrast_stretching': contrast_stretching}

                img = func_stretch_contrast[self._stretch_contrast_method](img)

            fig = Figure()
            FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.imshow(img, cmap='gray', interpolation=self.interpolation, aspect=float(aspect_img))
            self._add_orientation_label(ax)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self._save(fig, self.qc_report.qc_params.abs_bkg_img_path(), dpi=self.qc_report.qc_params.dpi)

            for action in self.action_list:
                logger.debug('Action List %s', action.__name__)
                if self._stretch_contrast and action.__name__ in ("no_seg_seg",):
                    print("Mask type %s" % mask.dtype)
                    mask = func_stretch_contrast[self._stretch_contrast_method](mask)
                fig = Figure()
                FigureCanvas(fig)
                ax = fig.add_subplot(111)
                action(self, mask, ax)
                self._save(fig, self.qc_report.qc_params.abs_overlay_img_path(), dpi=self.qc_report.qc_params.dpi)

            self.qc_report.update_description_file(img.shape)

        return wrapped_f

    def _add_orientation_label(self, ax):
        """
        Add orientation labels on the figure
        :param fig: MPL figure handler
        :return:
        """
        if self.qc_report.qc_params.orientation == 'Axial':
            # If mosaic of axial slices, display orientation labels
            ax.text(12, 6, 'A', color='yellow', size=6)
            ax.text(12, 28, 'P', color='yellow', size=6)
            ax.text(0, 18, 'L', color='yellow', size=6)
            ax.text(24, 18, 'R', color='yellow', size=6)

    def _save(self, fig, img_path, format='png', bbox_inches='tight', pad_inches=0.00, dpi=300):
        """
        Save the current figure into an image.
        :param fig: Figure handler
        :param img_path: str: path of the folder where the image is saved
        :param format: str: image format
        :param bbox_inches: str
        :param pad_inches: float
        :param dpi: int: Output resolution of the image
        :return:
        """
        logger.debug('Save image %s', img_path)
        fig.savefig(img_path,
                    format=format,
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                    transparent=True,
                    dpi=dpi)


class Params(object):
    """Parses and stores the variables that will included into the QC details

    Assuming BIDS convention, we derive the value of the dataset, subject and contrast from the `input_file`
    by splitting it into `[dataset]/[subject]/[contrast]/input_file`
    """

    def __init__(self, input_file, command, args, orientation, dest_folder, dpi=300):
        """
        Parameters
        :param input_file: str: the input nifti file name
        :param command: str: command name
        :param args: str: the command's arguments
        :param orientation: str: The anatomical orientation
        :param dest_folder: str: The absolute path of the QC root
        :param dpi: int: Output resolution of the image
        """
        path_in, file_in, ext_in = sct.extract_fname(os.path.abspath(input_file))
        # abs_input_path = os.path.dirname(os.path.abspath(input_file))
        abs_input_path, contrast = os.path.split(path_in)
        abs_input_path, subject = os.path.split(abs_input_path)
        _, dataset = os.path.split(abs_input_path)
        if isinstance(args, list):
            args = sct.list2cmdline(args)

        self.fname_in = file_in+ext_in
        self.dataset = dataset
        self.subject = subject
        self.cwd = os.getcwd()
        self.contrast = contrast
        self.command = command
        self.sct_version = sct.__version__
        self.args = args
        self.orientation = orientation
        self.dpi = dpi
        self.root_folder = dest_folder
        self.mod_date = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H%M%S.%f')
        self.qc_results = os.path.join(dest_folder, 'qc_results.json')
        self.bkg_img_path = os.path.join(dataset, subject, contrast, command, self.mod_date, 'bkg_img.png')
        self.overlay_img_path = os.path.join(dataset, subject, contrast, command, self.mod_date, 'overlay_img.png')

    def abs_bkg_img_path(self):
        return os.path.join(self.root_folder, self.bkg_img_path)

    def abs_overlay_img_path(self):
        return os.path.join(self.root_folder, self.overlay_img_path)


class QcReport(object):
    """This class generates the quality control report.

    It will also setup the folder structure so the report generator only needs to fetch the appropriate files.
    """

    def __init__(self, qc_params, usage):
        """
        Parameters
        :param qc_params: arguments of the "-param-qc" option in Terminal
        :param usage: str: description of the process
        """
        self.tool_name = qc_params.command
        self.slice_name = qc_params.orientation
        self.qc_params = qc_params
        self.usage = usage

        import spinalcordtoolbox
        pardir = os.path.dirname(os.path.dirname(spinalcordtoolbox.__file__))
        self.assets_folder = os.path.join(pardir, 'assets')

        self.img_base_name = 'bkg_img'
        self.description_base_name = "qc_results"

    def make_content_path(self):
        """Creates the whole directory to contain the QC report

        :return: return "root folder of the report" and the "furthest folder path" containing the images
        """
        # make a new or update Qc directory
        target_img_folder = os.path.dirname(self.qc_params.abs_bkg_img_path())

        try:
            os.makedirs(target_img_folder)
        except OSError as err:
            if not os.path.isdir(target_img_folder):
                raise err

    def update_description_file(self, dimension):
        """Create the description file with a JSON structure

        :param: dimension 2-tuple, the dimension of the image frame (w, h)
        """

        output = {
            'python': sys.executable,
            'cwd': self.qc_params.cwd,
            'cmdline': "{} {}".format(self.qc_params.command, self.qc_params.args),
            'command': self.qc_params.command,
            'sct_version': self.qc_params.sct_version,
            'dataset': self.qc_params.dataset,
            'subject': self.qc_params.subject,
            'contrast': self.qc_params.contrast,
            'fname_in': self.qc_params.fname_in,
            'orientation': self.qc_params.orientation,
            'background_img': self.qc_params.bkg_img_path,
            'overlay_img': self.qc_params.overlay_img_path,
            'dimension': '%dx%d' % dimension,
            'moddate': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        logger.debug('Description file: %s', self.qc_params.qc_results)
        results = []
        if os.path.isfile(self.qc_params.qc_results):
            results = json.load(open(self.qc_params.qc_results, 'r'))
        results.append(output)
        with portalocker.Lock(qc_report_file, "w") as lck_qc_file:
            json.dump(results, lck_qc_file, indent=2)
        self._update_html_assets(results)

    def _update_html_assets(self, json_data):
        """Update the html file and assets"""
        assets_path = os.path.join(os.path.dirname(__file__), 'assets')
        dest_path = self.qc_params.root_folder

        with io.open(os.path.join(assets_path, 'index.html')) as template_index:
            template = Template(template_index.read())
            output = template.substitute(sct_json_data=json.dumps(json_data))
            io.open(os.path.join(dest_path, 'index.html'), 'w').write(output)

        for path in ['css', 'js', 'imgs', 'fonts']:
            src_path = os.path.join(assets_path, '_assets', path)
            dest_full_path = os.path.join(dest_path, '_assets', path)
            if not os.path.exists(dest_full_path):
                os.makedirs(dest_full_path)
            for file_ in os.listdir(src_path):
                if not os.path.isfile(os.path.join(dest_full_path, file_)):
                    sct.copy(os.path.join(src_path, file_),
                             dest_full_path)


def add_entry(src, process, args, path_qc, plane, background=None, foreground=None,
              qcslice=None,
              qcslice_operations=[],
              qcslice_layout=None,
              dpi=300,
              stretch_contrast_method='contrast_stretching'):
    """
    Starting point to QC report creation.

    :param src: Path to input file (only used to populate report metadata)
    :param process:
    :param args:
    :param path_qc:
    :param plane:
    :param background:
    :param foreground:
    :param qcslice: spinalcordtoolbox.reports.slice:Axial
    :param qcslice_operations:
    :param qcslice_layout:
    :param dpi: int: Output resolution of the image
    :param stretch_contrast_method: Method for stretching contrast. See QcImage
    :return:
    """

    qc_param = Params(src, process, args, plane, path_qc, dpi)
    report = QcReport(qc_param, '')

    if qcslice is not None:
        @QcImage(report, 'none', qcslice_operations, stretch_contrast_method=stretch_contrast_method)
        def layout(qslice):
            return qcslice_layout(qslice)

        layout(qcslice)
    else:
        report.make_content_path()

        def normalized(img):
            return np.uint8(skimage.exposure.rescale_intensity(img, out_range=np.uint8))

        skimage.io.imsave(qc_param.abs_overlay_img_path(), normalized(foreground))

        if background is None:
            qc_param.bkg_img_path = qc_param.overlay_img_path
        else:
            skimage.io.imsave(qc_param.abs_bkg_img_path(), normalized(background))

        report.update_description_file(foreground.shape[:2])

    sct.printv('Successfully generated the QC results in %s' % qc_param.qc_results)
    sct.printv('Use the following command to see the results in a browser:')
    try:
        from sys import platform as _platform
        if _platform == "linux" or _platform == "linux2":
            sct.printv('xdg-open "{}/index.html"'.format(path_qc), type='info')
        elif _platform == "darwin":
            sct.printv('open "{}/index.html"'.format(path_qc), type='info')
        else:
            sct.printv('open file "{}/index.html"'.format(path_qc), type='info')
    except ImportError:
        print("WARNING! Platform undetectable.")


def generate_qc(fname_in1, fname_in2=None, fname_seg=None, args=None, path_qc=None, process=None):
    """
    Generate a QC entry allowing to quickly review results. This function is called by SCT scripts (e.g. sct_propseg).

    :param fname_in1: str: File name of input image #1 (mandatory)
    :param fname_in2: str: File name of input image #2
    :param fname_seg: str: File name of input segmentation
    :param args: args from parent function
    :param path_qc: str: Path to save QC report
    :param process: str: Name of SCT function. e.g., sct_propseg
    :return: None
    """
    dpi = 300
    # Get QC specifics based on SCT process
    if process in ['sct_register_multimodal', 'sct_register_to_template']:
        # axial orientation, switch between two input images
        plane = 'Axial'
        qcslice_type = qcslice.Axial([Image(fname_in1), Image(fname_in2), Image(fname_seg)])
        qcslice_operations = [QcImage.no_seg_seg]
        qcslice_layout = lambda x: x.mosaic()[:2]
    elif process in ['sct_propseg', 'sct_deepseg_sc', 'sct_deepseg_gm']:
        # axial orientation, switch between the image and the segmentation
        plane = 'Axial'
        qcslice_type = qcslice.Axial([Image(fname_in1), Image(fname_seg)])
        qcslice_operations = [QcImage.listed_seg]
        qcslice_layout = lambda x: x.mosaic()
    elif process in ['sct_warp_template']:
        # axial orientation, switch between the image and the linear segmentation (in blue)
        plane = 'Axial'
        qcslice_type = qcslice.Axial([Image(fname_in1), Image(fname_seg)])
        qcslice_operations = [QcImage.template]
        qcslice_layout = lambda x: x.mosaic()
    elif process in ['sct_label_vertebrae']:
        # sagittal orientation, display vertebral labels
        plane = 'Sagittal'
        dpi = 100  # bigger picture is needed for this special case, hence reduce dpi
        qcslice_type = qcslice.Sagittal([Image(fname_in1), Image(fname_seg)], p_resample=None)
        qcslice_operations = [QcImage.label_vertebrae]
        qcslice_layout = lambda x: x.single()
    elif process in ['sct_detect_pmj']:
        # sagittal orientation, display PMJ box
        plane = 'Sagittal'
        qcslice_type = qcslice.Sagittal([Image(fname_in1), Image(fname_seg)], p_resample=None)
        qcslice_operations = [QcImage.highlight_pmj]
        qcslice_layout = lambda x: x.single()
    else:
        sct.log.error('Unrecognised process.')

    add_entry(
        src=fname_in1,
        process=process,
        args=args,
        path_qc=path_qc,
        plane=plane,
        dpi=dpi,
        qcslice=qcslice_type,
        qcslice_operations=qcslice_operations,
        qcslice_layout=qcslice_layout,
        stretch_contrast_method='equalized',
    )
