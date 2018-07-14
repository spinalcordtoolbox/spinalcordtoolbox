#!/usr/bin/env python
#########################################################################################
#
# Visualizer for MRI volumes
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Created: 2015-01-30
#
# Notes on how to use classes in this script.
# If you are interested into selecting manually some points in an image, you can use the following code.

# from sct_viewer import ClickViewer
# from msct_image import Image
#
# im_input = Image('my_image.nii.gz')
#
# im_input_SAL = im_input.copy()
# # SAL orientation is mandatory
# im_input_SAL.change_orientation('SAL')
# # The viewer is composed by a primary plot and a secondary plot. The primary plot is the one you will click points in.
# # The secondary plot will help you go throughout slices in another dimensions to help manual selection.
# viewer = ClickViewer(im_input_SAL, orientation_subplot=['sag', 'ax'])
# viewer.number_of_slices = X  # Change X appropriately.
# viewer.gap_inter_slice = Y  # this number should reflect image spacing
# viewer.calculate_list_slices()
# # start the viewer that ask the user to enter a few points along the spinal cord
# mask_points = viewer.start()
# sct.printv(mask_points)

#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys, io, os

from time import time

from numpy import arange, max, pad, linspace, mean, median, std, percentile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.widgets import Slider, Button, RadioButtons


from msct_parser import Parser
from msct_image import Image
from msct_types import *
import sct_utils as sct




def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Volume Viewer')
    parser.add_option(name="-i",
                      type_value=[[','], 'file'],
                      description="Images to display.",
                      mandatory=True,
                      example="anat.nii.gz")

    parser.add_option(name='-mode',
                      type_value='multiple_choice',
                      description='Display mode.'
                                  '\nviewer: standard three-window viewer.'
                                  '\naxial: one-window viewer for manual centerline.\n',
                      mandatory=False,
                      default_value='viewer',
                      example=['viewer', 'axial'])

    parser.add_option(name='-param',
                      type_value=[[':'], 'str'],
                      description='Parameters for visualization. '
                                  'Separate images with \",\". Separate parameters with \":\".'
                                  '\nid: number of image in the "-i" list'
                                  '\ncmap: image colormap'
                                  '\ninterp: image interpolation. Accepts: [\'nearest\' | \'bilinear\' | \'bicubic\' | \'spline16\' | '
                                                                            '\'spline36\' | \'hanning\' | \'hamming\' | \'hermite\' | \'kaiser\' | '
                                                                            '\'quadric\' | \'catrom\' | \'gaussian\' | \'bessel\' | \'mitchell\' | '
                                                                            '\'sinc\' | \'lanczos\' | \'none\' |]'
                                  '\nvmin:'
                                  '\nvmax:'
                                  '\nvmean:'
                                  '\nperc: ',
                      mandatory=False,
                      example=['cmap=red:vmin=0:vmax=1', 'cmap=grey'])

    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1', '2'])

    return parser

class ParamImageVisualization(object):
    def __init__(self, id='0', mode='image', cmap='gray', interp='nearest', vmin='0', vmax='99', vmean='98', vmode='percentile', alpha='1.0'):
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

def prepare(list_images):
    fname_images, orientation_images = [], []
    for fname_im in list_images:
        from sct_image import orientation
        orientation_images.append(orientation(Image(fname_im), get=True, verbose=False))
        path_fname, file_fname, ext_fname = sct.extract_fname(fname_im)
        reoriented_image_filename = 'tmp.' + sct.add_suffix(file_fname + ext_fname, "_SAL")
        sct.run('sct_image -i ' + fname_im + ' -o ' + reoriented_image_filename + ' -setorient SAL -v 0', verbose=False)
        fname_images.append(reoriented_image_filename)
    return fname_images, orientation_images




def clean(fname_images):
    for fn in fname_images:
        os.remove(fn)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.init_sct()
    parser = get_parser()

    arguments = parser.parse(sys.argv[1:])

    fname_images, orientation_images = prepare(arguments["-i"])
    list_images = [Image(fname) for fname in fname_images]

    mode = arguments['-mode']

    param_image1 = ParamImageVisualization()
    visualization_parameters = ParamMultiImageVisualization([param_image1])
    if "-param" in arguments:
        param_images = arguments['-param']
        # update registration parameters
        for param in param_images:
            visualization_parameters.addImage(param)

    if mode == 'viewer':
        # 3 views
        from spinalcordtoolbox.viewer.ThreeViewer import ThreeViewer
        viewer = ThreeViewer(list_images, visualization_parameters)
        viewer.start()
    elif mode == 'axial':
        # only one axial view
        from spinalcordtoolbox.viewer.ClickViewer import ClickViewer
        viewer = ClickViewer(list_images, visualization_parameters)
        viewer.start()
    clean(fname_images)
