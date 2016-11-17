#!/usr/bin/env python
#########################################################################################
#
# Qc class implementation
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Frederic Cloutier Samson Lam Erwan Marchand Thierno Ib Barry Nguyen Kenny
# Modified: 2016-11-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import os
import json
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
import matplotlib
from msct_image import Image
from scipy import ndimage
import abc
import subprocess
import isct_generate_report

class Qc(object):
    """
    Creates a .png file from a 2d image produced by the class "slices"
    """
    # 'NameOfVertebrae':index
    _labels_regions = {'PONS': 50, 'MO': 51,
                 'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
                 'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19,
                 'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24,
                 'S1': 25, 'S2': 26, 'S3': 27, 'S4': 28, 'S5': 29,
                 'Co': 30}
    _labels_color = [   "#49ad2e","#e7363c","#c34603",
                        "#ed1339","#f88ae4","#44673e",
                        "#ffee00","#00c7ff","#199f26",
                        "#563691","#848545","#ce2fe1",
                        "#2142a6","#3edd76","#c4c253",
                        "#e8618a","#3128a3","#1a41db",
                        "#939e41","#3bec02","#1c2c79",
                        "#727c16","#18584e","#b49992",
                        "#3b0e6e","#6e856f","#637394",
                        "#36e05b","#530a1f","#8179c4",
                        "#e1320c","#52a4df","#000ab5",
                        "#4a4242","#0b53a5","#b49c19",
                        "#50e7a9","#bf5a42","#fa8d8e",
                        "#83839a","#320fef","#82ffbf",
                        "#360ee7","#551960","#11371e",
                        "#e900c3","#a21360","#58a601",
                        "#811c90","#235acf","#49395d",
                        "#9f89b0","#e08e08","#3d2b54",
                        "#7d0434","#fb1849","#e9e73a",
                        "#a22abd","#d58240","#ac2aff"

                        ]

    def __init__(self, label = False, dpi=600, interpolation='none'):
        self.interpolation = interpolation
        self.dpi = dpi
        self.label = label
        self.folder_name = None


    def __call__(self, f):
        # wrapped function (f). In this case, it is the "mosaic" or "single" methods of the class "slices"
        def wrapped_f(slice, *args, **kargs):
            
            # Get timestamp, will be used for folder structure and name of files
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            baseFilename = '{0}_{1}'.format(slice.name, timestamp)

            # Create the directory for the 
            rootFolderPath, leafNodeFullPath = self.mkdir(slice, timestamp)
            img, mask = f(slice,*args, **kargs)
       
            assert isinstance(img, np.ndarray)
            assert isinstance(mask, np.ndarray)
       
            fig = plt.imshow(img, cmap='gray', interpolation=self.interpolation)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
           
            # saves the original color without contrast
            self.save(leafNodeFullPath, '{}_original'.format(baseFilename))
            
            ax = plt.subplot()
            mask = np.rint(np.ma.masked_where(mask < 1, mask))
            plt.imshow(img, cmap='gray', interpolation=self.interpolation)
           
            if self.label:
                self.label_vertebrae(mask, ax)
            else:
                self._labels_color = {'#ff0000'}
            plt.imshow(mask, cmap= col.ListedColormap(self._labels_color),norm =
                matplotlib.colors.Normalize(vmin=0,vmax=len(self._labels_color)),interpolation=self.interpolation, alpha=1)
          
            if self.label:
                self.label_vertebrae(mask, ax)
         
            self.save(leafNodeFullPath, baseFilename)

            plt.close()

            #msct_qc.Qc().createDescriptionFile("sct_propseg", sys.argv[1:], parser.usage.description, None)
            syntax = '{} {}'.format(slice.contrast_type, os.path.basename(leafNodeFullPath)) 
            isct_generate_report.generate_report("description",syntax, rootFolderPath)

        return wrapped_f

    def label_vertebrae(self, data, ax):
        a = [0.0]
        for index, val in np.ndenumerate(data):
            if val not in a:
                a.append(val)
                index = int(val)
                color = self._labels_color[index]
                x, y = ndimage.measurements.center_of_mass(np.where(data == val, data, 0))
                label = self._labels_regions.keys()[list(self._labels_regions.values()).index(index)]
                ax.annotate(label, xy=(y,x), xytext=(y + 25, x),color= color,
                    arrowprops=dict(facecolor= color,shrink=0.05))


    def save(self, dirPath, name, format='png', bbox_inches='tight', pad_inches=0):
        plt.savefig('{0}/{1}.{2}'.format(dirPath, name, format), format=format, bbox_inches=bbox_inches,
                    pad_inches=pad_inches, dpi=self.dpi)

    def mkdir(self, slice, timestamp):
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
        newReportFolder = os.path.join(slice.report_root_folder, "report")
        newImgFolder = os.path.join(newReportFolder, "img")
        newContrastFolder = os.path.join(newImgFolder, slice.contrast_type)
        newToolProcessFolder = os.path.join(newContrastFolder, "{0}_{1}".format(slice.tool_name, timestamp))

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
            
        return newReportFolder, newToolProcessFolder

    def createDescriptionFile(self, tool, unparsed_args, description, commit_version):
        """
        Creates the description file with a JSON struct

        Description file structure:
        -----------------
    	commit_version:	version of last commit retrieved from util
            command: 	cmd used by user
    	description:	quick description of current usage
        """
        if not isinstance(commit_version, basestring):
            pathToProject = os.path.dirname(os.path.realpath(__file__))
            currentDir = os.getcwd()
            os.chdir(pathToProject)
            commit_version = subprocess.check_output(["git", "describe"])
            os.chdir(currentDir)
            cmd = ""
            for arg in unparsed_args:
                cmd += arg + " "
            cmd = tool + " " + str(cmd)
        with open("description", "w") as outfile:
            json.dump({"command": cmd, "description": description, "commit_version": commit_version}, outfile, indent = 4)
        outfile.close

class slices(object):
    """
    This class represents the slice objet that will be transformed in 2D image file.
    
    Parameters of the constructor
    ----------
    toolName:       Name of the sct_tool being used.  
    contrastType:   Contrast parameter used for the tool. 
    imageName:      Input 3D MRI to be separated into slices.
    segImageName:   Output name for the 3D MRI to be produced.
    """
 
    def __init__(self, toolName, contrastType, imageName, segImageName, reportRootFolder=None ):
        self.name = "{0}_{1}".format(toolName, contrastType) #Output base name for the .png images of the slices.
        self.tool_name = toolName                   # used to create folder
        self.contrast_type = contrastType           # used to create Folder
        self.image = Image(imageName)               # the original input
        self.image_seg = Image(segImageName)        # transformed input the one segmented
        self.image.change_orientation('SAL')        # reorient to SAL
        self.image_seg.change_orientation('SAL')    # reorient to SAL
        self.dim = self.getDim(self.image)
        
        # By default, the root folder will be one folder back, because we assume
        # that user will usually run from data structure like sct_example_data
        if reportRootFolder==None:
            reportRootPath = os.path.join(os.getcwd(), "..")
            if os.path.exists(reportRootPath):
                self.report_root_folder = reportRootPath
            # if the folder before doesn't exist, it will create in the current directory 
            else:
                self.report_root_folder = os.getcwd()
        else:
            self.report_root_folder = reportRootFolder

    __metaclass__ = abc.ABCMeta

    # ..._slice:    Gets a slice cut in the desired axis at the "i" position of the data of the 3D image.
    # ..._dim:      Gets the size of the desired dimension of the 3D image.
    @staticmethod
    def axial_slice(data, i):
        return data[ i, :, : ]

    @staticmethod
    def axial_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nx

    @staticmethod
    def sagital_slice(data, i):
        return data[ :, :, i ]

    @staticmethod
    def sagital_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nz

    @staticmethod
    def coronal_slice(data, i):
        return data[ :, i, : ]

    @staticmethod
    def coronal_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return ny

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

        if matrix.shape[ 0 ] < end_row:
            if matrix.shape[ 0 ] < (end_row - start_row):# TODO: throw/raise an exception that the matrix is smaller than the crop section
                raise OverflowError
            return slices.crop(matrix, center_x - 1, center_y, radius_x, radius_y)
        if matrix.shape[ 1 ] < end_col:
            if matrix.shape[ 1 ] < (end_col - start_col):# TODO: throw/raise an exception that the matrix is smaller than the crop section
                raise OverflowError
            return slices.crop(matrix, center_x, center_y - 1, radius_x, radius_y)
        if start_row < 0:
            return slices.crop(matrix, center_x + 1 , center_y, radius_x , radius_y)
        if start_col < 0:
            return slices.crop(matrix, center_x, center_y + 1, radius_x, radius_y)

        return matrix[ start_row:end_row, start_col:end_col ]

    @staticmethod
    def add_slice(matrix, i, column, size, patch):
        """
        This method adds a slice to the Matrix containing all the slices
        """
        startCol = (i % column) * size * 2
        endCol = startCol + patch.shape[ 1 ]
        startRow = int(math.ceil(i / column)) * size * 2
        endRow = startRow + patch.shape[ 0 ]
        matrix[ startRow:endRow, startCol:endCol ] = patch
        return matrix

    @staticmethod
    def nan_fill(array):
        array[ np.isnan(array) ] = np.interp(np.isnan(array).ravel().nonzero()[0]
                                             , (-np.isnan(array)).ravel().nonzero()[0]
                                             , array[ -np.isnan(array) ])
        return array


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

    def _axial_center(self):
        """
        Method to get the center of mass in the axial plan.
        :return: centers of mass in the x and y axis. 
        """
        axial_dim = self.axial_dim(self.image_seg)
        centers_x = np.zeros(axial_dim)
        centers_y = np.zeros(axial_dim)
        for i in xrange(axial_dim):
            centers_x[ i ], centers_y[ i ] \
                = ndimage.measurements.center_of_mass(self.axial_slice(self.image_seg.data, i))
        try:
            slices.nan_fill(centers_x)
            slices.nan_fill(centers_y)
        except ValueError:
            print "Oops! There are no trace of that spinal cord."  # TODO : raise error
            raise
        return centers_x, centers_y

    @Qc()
    def mosaic(self, nb_column, size):
        """
        Method to obtain matrices of the mosaics 
       
        :return matrix0: matrix of the input 3D RMI containing the mosaics of slices' "pixels"
        :return matrix1: matrix of the transformed 3D RMI to output containing the mosaics of slices' "pixels"
        """

        # Calculates how many squares will fit in a row based on the column and the size
        # Multiply by 2 because the sides are of size*2. Central point is size +/-.
        matrix0 = np.ones((size * 2 * int((self.dim / nb_column) + 1),size * 2 * nb_column))
        matrix1 = np.empty((size * 2 * int((self.dim / nb_column) + 1), size * 2 * nb_column))
        centers_x, centers_y = self.get_center()
        for i in range(self.dim):
            x = int(round(centers_x[ i ]))
            y = int(round(centers_y[ i ]))
            matrix0 = slices.add_slice(matrix0, i, nb_column, size,
                                       slices.crop(self.getSlice(self.image.data, i), x, y, size, size))
            matrix1 = slices.add_slice(matrix1, i, nb_column, size,
                                       slices.crop(self.getSlice(self.image_seg.data, i), x, y, size, size))

        return matrix0, matrix1

    @Qc(label= True,interpolation='nearest')
    def single(self):
        """
        Method to obtain matrices of the single slices
       
        :return matrix0: matrix of the input 3D RMI containing the slices
        :return matrix1: matrix of the transformed 3D RMI to output containing the slices
        """
        matrix0 = self.getSlice(self.image.data, self.dim/2)
        matrix1 = self.getSlice(self.image_seg.data,self.dim/2 )
        index = self.get_center_spit()
        for j in range(len(index)):
            matrix0[j] = self.getSlice(self.image.data, int(round(index[j])))[j]
            matrix1[j] = self.getSlice(self.image_seg.data, int(round(index[j])))[j]

        return matrix0, matrix1

    def save(self, nb_column=0, size=10):
        """
        Saves the image in a mosaic if a number of columns is specified else it saves each slices as individual images 
        :param size: Define the size of the side of the square containing the image of the slice. 
        """
        if nb_column > 0:
            return self.mosaic(nb_column, size)
        else:
            return self.single()


# The following classes (axial, sagital, coronal) inherits from the class "slices" and represents a cut in an axis

class axial(slices):
    def getSlice(self, data, i):
        return self.axial_slice(data, i)

    def getDim(self, image):
        return self.axial_dim(image)

    def get_center_spit(self):
        size = self.axial_dim(self.image_seg)
        return np.ones(size)*size/2

    def get_center(self):
        return self._axial_center()


class sagital(slices):
    def getSlice(self, data, i):
        return self.sagital_slice(data, i)

    def getDim(self, image):
        return self.sagital_dim(image)

    def get_center_spit(self):
        x, y = self._axial_center()
        return y

    def get_center(self):
        size_y = self.axial_dim(self.image_seg)
        size_x = self.coronal_dim(self.image_seg)
        return np.ones(self.dim)*size_x/2, np.ones(self.dim)*size_y/2


class coronal(slices):
    def getSlice(self, data, i):
        return  self.coronal_slice(data, i)

    def getDim(self, image):
        return self.coronal_dim(image)

    def get_center_spit(self):
        x, y = self._axial_center()
        return x

    def get_center(self):
        size_y = self.axial_dim(self.image_seg)
        size_x = self.sagital_dim(self.image_seg)
        return np.ones(self.dim) * size_x / 2, np.ones(self.dim) * size_y / 2
