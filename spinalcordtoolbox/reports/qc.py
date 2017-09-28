# -*- coding: utf-8 -*-
import json
import logging
import os
import subprocess

import warnings
warnings.filterwarnings("ignore")

import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.colorbar as colorbar
import matplotlib.colors as color
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


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
    _seg_colormap = plt.cm.autumn

    def __init__(self, qc_report, interpolation, action_list):
        """

        Parameters
        ----------
        qc_report : QcReport
            The QC report object
        interpolation : str
            Type of interpolation used in matplotlib
        action_list : list of functions
            List of functions that generates a specific type of images
        """
        self.qc_report = qc_report
        self.interpolation = interpolation
        self.action_list = action_list

    """
    action_list contain the list of images that has to be generated.
    It can be seen as "figures" of matplotlib to be shown
    Ex: if 'colorbar' is in the list, the msct_qc process will generate a color bar in the "img" folder
    """

    def listed_seg(self, mask):
        img = np.rint(np.ma.masked_where(mask < 1, mask))
        fig = plt.imshow(img,
                         cmap=color.ListedColormap(self._labels_color),
                         norm=color.Normalize(vmin=0, vmax=len(self._labels_color)),
                         interpolation=self.interpolation,
                         alpha=1,
                         aspect=float(self.aspect_mask))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    def no_seg_seg(self, mask):
        values = np.ma.masked_equal(np.rint(mask), 0)
        fig = plt.imshow(values,
                         cmap=plt.cm.gray,
                         interpolation=self.interpolation,
                         aspect=self.aspect_mask)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    def sequential_seg(self, mask):
        values = np.ma.masked_equal(np.rint(mask), 0)
        fig = plt.imshow(values,
                         cmap=self._seg_colormap,
                         interpolation=self.interpolation,
                         aspect=self.aspect_mask)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    def label_vertebrae(self, mask):
        self.listed_seg(mask)
        ax = plt.gca()
        a = [0.0]
        data = mask
        for index, val in np.ndenumerate(data):
            if val not in a:
                a.append(val)
                index = int(val)
                if index in self._labels_regions.values():
                    color = self._labels_color[index]
                    x, y = ndimage.measurements.center_of_mass(np.where(data == val, data, 0))
                    label = self._labels_regions.keys()[list(self._labels_regions.values()).index(index)]
                    ax.text(y, x, label, color='black', weight='heavy', clip_on=True)
                    ax.text(y, x, label, color=color, clip_on=True)

    def colorbar(self):
        fig = plt.figure(figsize=(9, 1.5))
        ax = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        colorbar.ColorbarBase(ax, cmap=self._seg_colormap, orientation='horizontal')
        return '{}_colorbar'.format(self.qc_report.img_base_name)

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
            aspect_img, self.aspect_mask = sct_slice.aspect()
            self.qc_report.make_content_path()
            logger.info('QC: %s with %s slice', func.__name__, sct_slice.get_name())

            img, mask = func(sct_slice, *args)

            plt.figure(1)
            fig = plt.imshow(img, cmap=plt.cm.gray, interpolation=self.interpolation, aspect=float(aspect_img))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            self._save(self.qc_report.qc_params.abs_bkg_img_path())

            for action in self.action_list:
                logger.debug('Action List %s', action.__name__)
                plt.clf()
                plt.figure(1)
                action(self, mask)
                self._save(self.qc_report.qc_params.abs_overlay_img_path())
            plt.close()

            self.qc_report.update_description_file(img.shape)

        return wrapped_f

    def _save(self, img_path, format='png', bbox_inches='tight', pad_inches=0.00):
        """ Save the current figure into an image.

        Parameters
        ----------
        img_path : str
            path of the folder where the image is saved
        format : str
            image format
        bbox_inches : str
        pad_inches : float

        """
        logger.debug('Save image %s', img_path)
        plt.savefig(img_path,
                    format=format,
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                    transparent=True,
                    dpi=600)


class Params(object):
    """Parses and stores the variables that will included into the QC details

    We derive the value of the contrast and subject name from the `input_file` path,
    by splitting it into `[subject]/[contrast]/input_file`
    """

    def __init__(self, input_file, command, args, orientation, dest_folder):
        """
        Parameters
        ----------
        input_file : str
            the input nifti file name
        command : str
            the command name
        args : str
            the command's arguments
        orientation : str
            The anatomical orientation
        dest_folder : str
            The absolute path of the QC root
        """
        abs_input_path = os.path.dirname(os.path.abspath(input_file))
        abs_input_path, contrast = os.path.split(abs_input_path)
        _, subject = os.path.split(abs_input_path)

        self.subject = subject
        self.contrast = contrast
        self.command = command
        self.args = args
        self.orientation = orientation
        self.root_folder = dest_folder
        self.mod_date = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H%M%S')

        self.qc_results = os.path.join(dest_folder, 'qc_results.json')
        self.bkg_img_path = os.path.join(subject, contrast, command, self.mod_date, 'bkg_img.png')
        self.overlay_img_path = os.path.join(subject, contrast, command, self.mod_date, 'overlay_img.png')

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
        ----------
        tool_name : str
            name of the sct tool being used. Is used to name the image file.
        qc_params : Params
            arguments of the "-param-qc" option in Terminal
        cmd_args : list of str
            the commands of the process being used to generate the images
        usage : str
             description of the process
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
        """Creates the description file with a JSON structure

        Parameters
        ----------
        dimension : (int, int)
            The dimension of the image frame
        """
        # get path of the toolbox
        try:
            cmd = 'git rev-parse --short HEAD'.split()
            git_hash = subprocess.check_output(cmd).strip('"')
        except (subprocess.CalledProcessError, OSError):
            git_hash = "N/A"

        output = {
            'command': self.qc_params.command,
            'version': git_hash,
            'args': ' '.join(self.qc_params.args),
            'subject': self.qc_params.subject,
            'contrast': self.qc_params.contrast,
            'orientation': self.qc_params.orientation,
            'background_img': self.qc_params.bkg_img_path,
            'overlay_img': self.qc_params.overlay_img_path,
            'dimension': '%dx%d' % dimension,
            'moddate': datetime.datetime.now().isoformat(' ')
        }
        logger.debug('Description file: %s', self.qc_params.qc_results)
        results = []
        if os.path.isfile(self.qc_params.qc_results):
            results = json.load(open(self.qc_params.qc_results, 'r'))
        results.append(output)
        json.dump(results, open(self.qc_params.qc_results, "w"), indent=2)
