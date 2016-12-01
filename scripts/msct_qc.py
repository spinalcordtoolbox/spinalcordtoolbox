#!/usr/bin/env python
#########################################################################################
#
# Qc class implementation
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Frederic Cloutier Samson Lam Erwan Marchand Thierno  Barry Nguyen Kenny
# Modified: 2016-11-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import os
import json
import time
from __builtin__ import staticmethod

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import matplotlib
from msct_image import Image
from scipy import ndimage
import abc
import isct_generate_report
import commands
import sct_utils as sct
import glob
import shutil


class Qc_Params(object):
    """
    Parses and stores the variables that can be provides as arguments in the Terminal
    These variables are used as parameters to define how some images should be generated
    and how the report should be generated.
    Other variables not provided by the Terminal are in another class.
    """
    def __init__(self, params_qc=None):
        """
        :param params_qc: arguments provided to the -param-qc parameter in the Terminal
        """
        # list of parameters that can be provided by the args
        # by default, root folder is in the previous folder
        self.report_root_folder = os.path.join(os.getcwd(), "..")
        self.nb_column = 10
        self.show_report = False

        # having the generate_report forces to make make the following condition verification
        # "if qcParams is None or qcParams.generate_report is True:"
        # it must be done after parsing arguments.
        self.generate_report = True
        self.threshold = 0.5

        # settings up the parameters
        if params_qc is not None or params_qc == "":
            self.parse_params(params_qc)

        # must be done after parsing the params
        self.validate_root_folder()

    def parse_params(self, params_qc):
        """
        Converts the parameters, verification of arg -param-qc outside because this class is for params only
        :param params_qc: list of the arguments following the -param-qc paramter in the terminal
        :return:
        """
        # iterates through the list to parse each pairs of {parameter="value"}
        for paramStep in params_qc:
            # params[0] is left side of the '=' and params[1] is right side of the '='
            params = paramStep.split('=')
            if len(params) > 1:
                # Parameter where the report should be created/updated
                if params[0] == "ofolder":
                    self.report_root_folder = params[1]
                # Parameter defining how many columns should be created in the picture
                elif params[0] == "ncol":
                    self.nb_column = int(params[1])
                # We assume default parameters will not be provided for the conditions containing 'and'
                # If default value is provided it will appear as warning.
                elif params[0] == "autoview" and int(params[1]) == 1:
                    self.show_report = True
                elif params[0] == "generate" and int(params[1]) == 0:
                    self.generate_report = False
                elif params[0] == "thresh":
                    self.threshold = float(params[1])
                else:
                    sct.printv("-param-qc parameter '{}' is not valid...".format(params[0]), 1, type='warning')

    def validate_root_folder(self):
        # By default, the root folder will be one folder back, because we assume
        # that user will usually run from data structure like sct_example_data
        if not os.path.exists(self.report_root_folder):
            self.report_root_folder = os.getcwd()

    @staticmethod
    def get_qc_params_description(list_params):
        """
        Generates the help description for -param-qc

        :param self:
        :param list_params: string of the available parameters of the param-qc arg.
        :return: the description text to show
        """
        param_description = "Specify conditions under which the Qc report should be generated."

        for param in list_params:
            if param == "ofolder":
                param_description += "\n:ofolder={path to create 'qc' folder, default: '../'.}"
            elif param == "ncol":
                param_description += "\n:ncol={number of slices per row, default: 10.}"
            elif param == "autoview":
                param_description += "\n:autoview={open report after generation (yes=1, no=0), default: 0.}"
            elif param == "generate":
                param_description += "\n:generate={generate report (yes=1, no=0), default: 1.}"
            elif param == "thresh":
                param_description += "\n:thresh={threshold of the colorbar (0.0 to 1.0), default: 0.5.}"
            else:
                param_description += ""

        param_description += "\nExample of use: '-param-qc ofolder=../../,ncol=8,autoview=1'."
        return param_description


class Qc_Report(object):
    """
    This class contains all the necessary methods and variables to tell how the report should be generated.
    It will also setup the folder structure so the report generator only needs to fetch the appropriate files.
    """
    def __init__(self, tool_name, qc_params, cmd_args, usage):
        """
        :param tool_name: name of the sct tool being used. Is used to name the image file.
        :param qc_params: arguments of the "-param-qc" option in Terminal
        :param cmd_args:  the commands of the process being used to generate the images
        :param usage:     description of the process
        """
        # the class Qc_Params verification done here to prevent from having to be sure it's not none outside
        if qc_params is None:
            qc_params = Qc_Params()
        self.qc_params = qc_params

        # used to create folder
        self.tool_name = tool_name

        # os.path.relpath to get the current directory to use in the naming instead of using the contrast_type
        # if desired, contrast_type can be an input parameter to give more freedom to the user to specify a name
        # requires to change a bit the code
        head, tail = os.path.split(os.path.relpath(".", "../.."))
        self.subject_name = head  # assume subject always two folder back, find better way to do it
        self.contrast_type = os.path.relpath(".", "..")

        # used for description
        self.cmd_args = cmd_args
        self.usage = usage

        # Get timestamp, will be used for folder structure and name of files
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        self.img_base_name = '{0}_{1}_{2}'.format(tool_name, self.contrast_type, timestamp)

        # can be used instead of the return value of the function mkdir, 
        # workaround for mkdir to save description file and to use it
        self.report_leaf_folder = None
        self.description_base_name = "description_{}".format(self.timestamp)

        #slice name axial or sagital
        self.slice_name = None

    def set_used_slice_name(self, name):
        self.slice_name = name

    def generate_report_for_text(self, file_output=None):
        """
        Generate report. Class object must already be instanced before executing this method.
        This method is mostly used when no action list is required (eg extract_metric).
        """
        rootFolderPath, leafNodeFullPath = self.mkdir()
        if file_output:
            shutil.copy(os.path.join(".",file_output), self.report_leaf_folder)
        else:
            txts= glob.glob1(".", "*.txt")
            pickles = glob.glob1(".", "*.pickle")
            elements=txts + pickles
            for i in elements:
                 shutil.copy(i, self.report_leaf_folder)

        # create description
        self.create_description_file(self.cmd_args, self.usage, None)
        # create HTML files
        self.generate_html(rootFolderPath, leafNodeFullPath)


    def mkdir(self):
        """
        Creates the whole directory to contain the QC report.

        Folder structure:
        -----------------
        .(report)
        +-- _img
        |   +-- _contrast01
        |      +-- _toolProcess01_timestamp
        |          +-- contrast01_tool01_timestamp.png
        |   +-- _contrast02
        |      +-- _toolProcess01_timestamp
        |          +-- contrast02_tool01_timestamp.png
        ...
        |
        +-- index.html

        :return: return "root folder of the report" and the "furthest folder path" containing the images
        """
        # make a new or update Qc directory
        newReportFolder = os.path.join(self.qc_params.report_root_folder, "qc")
        newImgFolder = os.path.join(newReportFolder, "img")
        newContrastFolder = os.path.join(newImgFolder, self.contrast_type)
        newToolProcessFolder = os.path.join(newContrastFolder, "{0}_{1}".format(self.tool_name, self.timestamp))

        # Only create folder when it doesn't exist and it is always done in the current terminal
        # TODO: print the created directory
        if not os.path.exists(newReportFolder):
            os.mkdir(newReportFolder)
        if not os.path.exists(newImgFolder):
            os.mkdir(newImgFolder)
        if not os.path.exists(newContrastFolder):
            os.mkdir(newContrastFolder)
        if not os.path.exists(newToolProcessFolder):
            os.mkdir(newToolProcessFolder)

        # save the leaf folder name for Description file
        self.report_leaf_folder = newToolProcessFolder

        return newReportFolder, newToolProcessFolder

    def create_description_file(self, unparsed_args, description, sct_commit):
        """
        Creates the description file with a JSON struct

        :param unparsed_args: the commands used in the Terminal of the process
        :param description:   quick description of current usage of the process
        :param sct_commit:    commit version of the code being currently used to generate the images
        :return:
        """
        if not isinstance(sct_commit, basestring):
            # get path of the toolbox
            path_script = os.path.dirname(__file__)
            path_sct = os.path.dirname(path_script)

            # fetch true commit number and branch (do not use commit.txt which is wrong)
            path_curr = os.path.abspath(os.curdir)
            os.chdir(path_sct)
            sct_commit = commands.getoutput('git rev-parse HEAD')
            if not sct_commit.isalnum():
                print 'WARNING: Cannot retrieve SCT commit'
                sct_commit = 'unknown'
                sct_branch = 'unknown'
            else:
                sct_branch = commands.getoutput('git branch --contains ' + sct_commit).strip('* ')
            # with open (path_sct+"/version.txt", "r") as myfile:
            #     version_sct = myfile.read().replace('\n', '')
            # with open (path_sct+"/commit.txt", "r") as myfile:
            #     commit_sct = myfile.read().replace('\n', '')
            print 'SCT commit/branch: ' + sct_commit + '/' + sct_branch
            os.chdir(path_curr)
            cmd = ""
            for arg in unparsed_args:
                cmd += arg + " "
            cmd = "sct_{}".format(self.tool_name) + " " + str(cmd)
        with open(os.path.join(self.report_leaf_folder, "{}.txt".format(self.description_base_name)), "w") as outfile:
            json.dump({"command": cmd, "commit_version": sct_commit}, outfile, indent=4)
            outfile.write(description + '\n')
        outfile.close

    def generate_html(self, report_dir, leaf_node_full_path):
        """

        :param report_dir: folder containing report
        :param leaf_node_full_path: folder containing all tems (eg images, texts) for html
        :return:
        """
         # create htmls
        syntax = '{} {}'.format(self.contrast_type, os.path.basename(leaf_node_full_path))
        isct_generate_report.generate_report("{}.txt".format(self.description_base_name), syntax,
                                             report_dir, self.qc_params.show_report, self.subject_name,self.slice_name)




class Qc(object):
    """
    Class used to create a .png file from a 2d image produced by the class "Slice"
    """
    # 'NameOfVertebrae':index
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
                     "#a22abd", "#d58240", "#ac2aff"
                     ]
    _seg_colormap = cm.autumn

    # start of action_list
    """
    action_list contain the list of images that has to be generated.
    It can be seen as "figures" of matplotlib to be shown
    Ex: if 'colorbar' is in the list, the msct_qc process will generate a color bar in the "img" folder
    """
    def listed_seg(self, mask):
        img = np.rint(np.ma.masked_where(mask < 1, mask))
        plt.imshow(img, cmap=col.ListedColormap(self._labels_color), norm=
        matplotlib.colors.Normalize(vmin=0, vmax=len(self._labels_color)),
                   interpolation=self.interpolation, alpha=1, aspect=float(self.aspect_mask))
        return self.qc_report.img_base_name

    def no_seg_seg(self, mask):
        img = np.rint(np.ma.masked_where(mask == 0, mask))
        plt.imshow(img, cmap=cm.gray, interpolation=self.interpolation, aspect=self.aspect_mask)
        return self.qc_report.img_base_name

    def sequential_seg(self, mask):
        seg = mask
        seg = np.ma.masked_where(seg == 0, seg)
        plt.imshow(seg, cmap=self._seg_colormap, interpolation=self.interpolation, aspect=self.aspect_mask)
        return self.qc_report.img_base_name

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
        return '{}_label'.format(self.qc_report.img_base_name)

    def colorbar(self, mask):
        fig = plt.figure(figsize=(9, 1.5))
        ax = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=self._seg_colormap, orientation='horizontal')
        # cb.set_label('Some Units')
        return '{}_colorbar'.format(self.qc_report.img_base_name)
    # end of action_list

    def __init__(self, qc_report, interpolation='none', action_list=[listed_seg]):
        """
        :param qc_report:       object containing the info necessary for the qc report generation
        :param interpolation:   type of interpolation to be used in matplotlib
        :param action_list:     list of methods that generates a specific type of images.
        """
        self.qc_report = qc_report
        # used to save the image file
        self.interpolation = interpolation
        self.action_list = action_list

    def __call__(self, f):
        # wrapped function (f). In this case, it is the "mosaic" or "single" methods of the class "Slice"
        def wrapped_f(sct_slice, *args):
            assert isinstance(sct_slice,Slice)
            self.qc_report.set_used_slice_name(sct_slice.get_name())
            aspect_img,  self.aspect_mask = sct_slice.aspect()

            rootFolderPath, leafNodeFullPath = self.qc_report.mkdir()
            img, mask = f(sct_slice, *args)
            assert isinstance(img, np.ndarray)
            assert isinstance(mask, np.ndarray)

            # Make original plot
            plt.figure(1)
            fig = plt.imshow(img, cmap=cm.gray, interpolation=self.interpolation, aspect= float(aspect_img))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

            # saves the original color without contrast
            self.__save(leafNodeFullPath, '{}_original'.format(self.qc_report.img_base_name))

            # Make params for segmented plot and others decorations
            # Save each action in order to build up an animation
            for action in self.action_list:
                self.__save(leafNodeFullPath, action(self, mask))

            plt.close()

            # create description
            self.qc_report.create_description_file(self.qc_report.cmd_args, self.qc_report.usage, None)
            # create htmls
            self.qc_report.generate_html(rootFolderPath, leafNodeFullPath)

        return wrapped_f

    def __save(self, dirPath, name, format='png', bbox_inches='tight', pad_inches=0.00):
        """
        Saves the image
        :param dirPath:     path of the folder where the image is saved
        :param name:        name of the image file
        :param format:      file extension name of the image
        :param bbox_inches:
        :param pad_inches:
        :return:
        """
        plt.savefig('{0}/{1}.{2}'.format(dirPath, name, format), format=format, bbox_inches=bbox_inches,
                    pad_inches=pad_inches, dpi=600)


class Slice(object):
    """
    This class represents the slice objet that will be transformed in 2D image file.
    
    Parameters of the constructor
    ----------
    imageName:      Input 3D MRI to be separated into slices.
    segImageName:   Output name for the 3D MRI to be produced.
    """

    def __init__(self, imageName, segImageName):
        # type: (object, object) -> object
        image = Image(imageName)  # the original input
        image_seg = Image(segImageName)  # transformed input the one segmented
        image.change_orientation('SAL')  # reorient to SAL
        image_seg.change_orientation('SAL')  # reorient to SAL
        self.image = image
        self.image_seg = image_seg

    __metaclass__ = abc.ABCMeta

    # ..._slice:    Gets a slice cut in the desired axis at the "i" position of the data of the 3D image.
    # ..._dim:      Gets the size of the desired dimension of the 3D image.
    @staticmethod
    def axial_slice(data, i):
        return data[i, :, :]

    @staticmethod
    def axial_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nx

    @staticmethod
    def axial_aspect(image):
        nx,ny,nz,nt,px,py,pz,pt = image.dim
        return py/pz

    @staticmethod
    def sagittal_slice(data, i):
        return data[:, :, i]

    @staticmethod
    def sagittal_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nz

    @staticmethod
    def sagittal_aspect(image):
        nx,ny,nz,nt,px,py,pz,pt = image.dim
        return px / py

    @staticmethod
    def coronal_slice(data, i):
        return data[:, i, :]

    @staticmethod
    def coronal_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return ny

    @staticmethod
    def coronal_aspect(image):
        nx,ny,nz,nt,px,py,pz,pt = image.dim
        return px / pz

    @abc.abstractmethod
    def get_aspect(self, image):
        return

    @staticmethod
    def crop(matrix, center_x, center_y, radius_x, radius_y):
        """
        This method crops the unnecessary parts of the image to keep only the essential image of the slice
        """
        # Radius is actually the size of the square. It is not the same radius for a circle
        start_row = center_x - radius_x
        end_row = center_x + radius_x
        start_col = center_y - radius_y
        end_col = center_y + radius_y

        if matrix.shape[0] < end_row:
            if matrix.shape[0] < (
                end_row - start_row):  # TODO: throw/raise an exception that the matrix is smaller than the crop section
                raise OverflowError
            return Slice.crop(matrix, center_x - 1, center_y, radius_x, radius_y)
        if matrix.shape[1] < end_col:
            if matrix.shape[1] < (
                end_col - start_col):  # TODO: throw/raise an exception that the matrix is smaller than the crop section
                raise OverflowError
            return Slice.crop(matrix, center_x, center_y - 1, radius_x, radius_y)
        if start_row < 0:
            return Slice.crop(matrix, center_x + 1, center_y, radius_x, radius_y)
        if start_col < 0:
            return Slice.crop(matrix, center_x, center_y + 1, radius_x, radius_y)

        return matrix[start_row:end_row, start_col:end_col]

    @staticmethod
    def add_slice(matrix, i, column, size, patch):
        """
        This method adds a slice to the Matrix containing all the slices
        """
        startCol = (i % column) * size * 2
        endCol = startCol + patch.shape[1]
        startRow = int(math.ceil(i / column)) * size * 2
        endRow = startRow + patch.shape[0]
        matrix[startRow:endRow, startCol:endCol] = patch
        return matrix

    @staticmethod
    def nan_fill(array):
        array[np.isnan(array)] = np.interp(np.isnan(array).ravel().nonzero()[0]
                                           , (-np.isnan(array)).ravel().nonzero()[0]
                                           , array[-np.isnan(array)])
        return array

    @abc.abstractmethod
    def get_name(self):
        return

    @abc.abstractmethod
    def getSlice(self, data, i):
        """
        Abstract method to obtain a slice of a 3d matrix.
        :param data: 3d numpy.ndarray
        :param i: int
        :return: 2d numpy.ndarray
        """
        return

    @abc.abstractmethod
    def getDim(self, image):
        """
        Abstract method to obtain the depth of the 3d matrix.
        :param image: 3d numpy.ndarray
        :return: int
        """
        return

    def _axial_center(self,image):
        """
        Method to get the center of mass in the axial plan.
        :return: centers of mass in the x and y axis. 
        """
        axial_dim = self.axial_dim(image)
        centers_x = np.zeros(axial_dim)
        centers_y = np.zeros(axial_dim)
        for i in xrange(axial_dim):
            centers_x[i], centers_y[i] \
                = ndimage.measurements.center_of_mass(self.axial_slice(image.data, i))
        try:
            Slice.nan_fill(centers_x)
            Slice.nan_fill(centers_y)
        except ValueError:
            print "Oops! There are no trace of that spinal cord."  # TODO : raise error
            raise
        return centers_x, centers_y

    def mosaic(self, nb_column=10, size=15):
        """
        Method to obtain matrices of the mosaics 
       
        :return matrix0: matrix of the input 3D RMI containing the mosaics of slices' "pixels"
        :return matrix1: matrix of the transformed 3D RMI to output containing the mosaics of slices' "pixels"
        """
        # Calculates how many squares will fit in a row based on the column and the size
        # Multiply by 2 because the sides are of size*2. Central point is size +/-.
        dim = self.getDim(self.image)
        nb_column = int(np.clip(dim, 1, nb_column))
        nb_row = -(-dim//nb_column) # upside-down floor division
        matrix0 = np.ones(( size * 2 * nb_row, size * 2 * nb_column))
        matrix1 = np.zeros(( size * 2 * nb_row, size * 2 * nb_column))
        centers_x, centers_y = self.get_center()
        for i in range(dim):
            x = int(round(centers_x[i]))
            y = int(round(centers_y[i]))
            matrix0 = self.add_slice(matrix0, i, nb_column, size,
                                       self.crop(self.getSlice(self.image.data, i), x, y, size, size))
            matrix1 = self.add_slice(matrix1, i, nb_column, size,
                                       self.crop(self.getSlice(self.image_seg.data, i), x, y, size, size))

        return matrix0, matrix1

    def single(self):
        """
        Method to obtain matrices of the single slices
       
        :return matrix0: matrix of the input 3D RMI containing the slices
        :return matrix1: matrix of the transformed 3D RMI to output containing the slices
        """
        assert  self.image.data.shape == self.image_seg.data.shape

        dim = self.getDim(self.image)
        matrix0 = self.getSlice(self.image.data, dim / 2)
        matrix1 = self.getSlice(self.image_seg.data, dim / 2)
        index = self.get_center_spit()
        for j in range(len(index)):
            matrix0[j] = self.getSlice(self.image.data, int(round(index[j])))[j]
            matrix1[j] = self.getSlice(self.image_seg.data, int(round(index[j])))[j]

        return matrix0, matrix1

    def aspect(self):
        return self.get_aspect(self.image),self.get_aspect(self.image_seg)


"""
The following classes (axial, sagittal, coronal) inherits from the class "Slice" and represents a cut in an axis
"""
class axial(Slice):
    def get_name(self):
        return axial.__name__

    def get_aspect(self, image):
        return Slice.axial_aspect(image)

    def getSlice(self, data, i):
        return self.axial_slice(data, i)

    def getDim(self, image):
        return self.axial_dim(image)

    def get_center_spit(self):
        size = self.axial_dim(self.image_seg)
        return np.ones(size) * size / 2

    def get_center(self):
        return self._axial_center(self.image_seg)


class sagittal(Slice):

    def get_name(self):
        return sagittal.__name__

    def get_aspect(self, image):
        return Slice.sagittal_aspect(image)

    def getSlice(self, data, i):
        return self.sagittal_slice(data, i)

    def getDim(self, image):
        return self.sagittal_dim(image)

    def get_center_spit(self):
        x, y = self._axial_center(self.image_seg)
        return y

    def get_center(self):
        dim = self.getDim(self.image_seg)
        size_y = self.axial_dim(self.image_seg)
        size_x = self.coronal_dim(self.image_seg)
        return np.ones(dim) * size_x / 2, np.ones(dim) * size_y / 2


class coronal(Slice):

    def get_name(self):
        return coronal.__name__

    def get_aspect(self, image):
        return Slice.coronal_aspect(image)

    def getSlice(self, data, i):
        return self.coronal_slice(data, i)

    def getDim(self, image):
        return self.coronal_dim(image)

    def get_center_spit(self):
        x, y = self._axial_center(self.image_seg)
        return x

    def get_center(self):
        dim = self.getDim(self.image_seg)
        size_y = self.axial_dim(self.image_seg)
        size_x = self.sagittal_dim(self.image_seg)
        return np.ones(dim) * size_x / 2, np.ones(dim) * size_y / 2


class template_axial(axial):
    def getDim(self, image):
        return min([self.axial_dim(image), self.axial_dim(self.image_seg)])

    def get_size(self, image):
        return min(image.data.shape + self.image_seg.data.shape)//2

    def get_center(self):
        size = self.get_size(self.image)
        dim = self.getDim(self.image)
        return np.ones(dim) * size, np.ones(dim) * size

    def mosaic(self, nb_column=10):
        return super(template_axial, self).mosaic(size=self.get_size(self.image), nb_column=nb_column)

    def single(self):
        dim = self.getDim(self.image)
        matrix0 = self.getSlice(self.image.data, dim / 2)
        matrix1 = self.getSlice(self.image_seg.data, dim / 2)

        return matrix0, matrix1


class template2anat_axial(template_axial):
    def __init__(self, imageName, template2anatName, segImageName):
        super(template2anat_axial, self).__init__(imageName, template2anatName)
        self.image_seg2 = Image(segImageName)  # transformed input the one segmented
        self.image_seg2.change_orientation('SAL')  # reorient to SAL

    def get_center(self):
        return self._axial_center(self.image_seg2)


class template_sagittal(sagittal):
    def getDim(self, image):
        return min([self.sagittal_dim(image), self.sagittal_dim(self.image_seg)])

    def get_size(self, image):
        return min(image.data.shape + self.image_seg.data.shape) // 2

    def get_center(self):
        size = self.get_size(self.image)
        dim = self.getDim(self.image)
        return np.ones(dim) * size, np.ones(dim) * size

    def mosaic(self, nb_column=10):
        return super(template_sagittal, self).mosaic(size=self.get_size(self.image), nb_column=nb_column)

    def single(self):
        dim = self.getDim(self.image)
        matrix0 = self.getSlice(self.image.data, dim / 2)
        matrix1 = self.getSlice(self.image_seg.data, dim / 2)

        return matrix0, matrix1


class template2anat_sagittal(sagittal):
    def __init__(self, imageName, template2anatName, segImageName):
        super(template2anat_sagittal, self).__init__(imageName, template2anatName)
        self.image_seg2 = Image(segImageName)  # transformed input the one segmented
        self.image_seg2.change_orientation('SAL')  # reorient to SAL

    def get_center(self):
        return self._axial_center(self.image_seg2)



