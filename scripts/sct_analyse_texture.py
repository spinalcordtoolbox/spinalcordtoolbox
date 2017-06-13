#!/usr/bin/env python
#######################################################################################################################
#
# Analyse texture
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley
# Modified: 2017-06-13
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

import os
import shutil
import sys
import numpy as np

import sct_maths
from msct_image import Image
from msct_parser import Parser
from sct_image import set_orientation
from sct_utils import (add_suffix, extract_fname, printv, run,
                       slash_at_the_end, tmp_create)

def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Extraction of texture features from an image within a given mask.\n'
                                 ' It calculates the texture properties of a grey level co-occurence matrix (GLCM).'
                                 ' The textures features are those defined in the sckit-image implementation:\n'
                                 ' http://scikit-image.org/docs/dev/api/skimage.feature.html#greycoprops\n'
                                 ' This function outputs one nifti file per texture metric (contrast, dissimilarity, homogeneity, ASM, energy, correlation).')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image to analyse",
                      mandatory=True,
                      example='t2.nii.gz')
    parser.add_option(name="-s",
                      type_value="file",
                      description="Spinal cord segmentation",
                      mandatory=True,
                      example='t2_seg.nii.gz')
    parser.add_option(name="-param",
                      type_value="str",
                      description="Parameters for extraction. Separate arguments with \":\".\n"
                                  "prop: texture property/ies of a GLCM. Default='contrast,dissimilarity,homogeneity,energy,correlation,ASM'\n"
                                  "distance: List of pixel pair distance offsets. Default=1\n"
                                  "angle: List of pixel pair angles in degrees. Default=0\n",
                      mandatory=False,
                      example="prop='energy':distance=1:angles=0")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      default_value=Param().path_results,
                      example='texture_analysis_results/')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value=str(int(Param().rm_tmp)),
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value=str(Param().verbose))

    return parser

class SegmentGM:
    def __init__(self, param_seg=None, param_model=None, param_data=None, param=None):
        self.param_seg = param_seg if param_seg is not None else ParamSeg()
        self.param_model = param_model if param_model is not None else ParamModel()
        self.param_data = param_data if param_data is not None else ParamData()
        self.param = param if param is not None else Param()

        # create model:
        self.model = Model(param_model=self.param_model, param_data=self.param_data, param=self.param)

        # create tmp directory
        self.tmp_dir = tmp_create(verbose=self.param.verbose)  # path to tmp directory

        self.target_im = None  # list of slices
        self.info_preprocessing = None  # dic containing {'orientation': 'xxx', 'im_sc_seg_rpi': im, 'interpolated_images': [list of im = interpolated image data per slice]}

        self.projected_target = None  # list of coordinates of the target slices in the model reduced space
        self.im_res_gmseg = None
        self.im_res_wmseg = None

class Param:
  def __init__(self):
    self.fname_im = None
    self.fname_seg = None
    self.path_results = './'

    self.rm_tmp = '1'
    self.verbose = '0'

class ParamGLCM(object):
  def __init__(self, symmetric=True, normed=True, prop='contrast,dissimilarity,homogeneity,energy,correlation,ASM', distance='1', angle='0'):
    self.symmetric = True  # If True, the output matrix P[:, :, d, theta] is symmetric.
    self.normed = True  # If True, normalize each matrix P[:, :, d, theta] by dividing by the total number of accumulated co-occurrences for the given offset. 
                        # The elements of the resulting matrix sum to 1.
    self.prop = 'contrast,dissimilarity,homogeneity,energy,correlation,ASM'
    self.distance = [1]
    self.angle = [0]

  # update constructor with user's parameters
  def update(self, param_user):
    param_lst = param_user.split(':')
    for param in param_lst:
      obj = param.split('=')
      setattr(self, obj[0], obj[1])

def main(args=None):
  if args is None:
    args = sys.argv[1:]

  # create param object
  param = Param()
  param_glcm = ParamGLCM()

  # get parser
  parser = get_parser()
  arguments = parser.parse(args)

  # set param arguments ad inputted by user
  param.fname_im = arguments["-i"]
  param.fname_seg = arguments["-s"]

  if '-ofolder' in arguments:
    param.path_results = arguments['-ofolder']
  if '-r' in arguments:
    param.rm_tmp = bool(int(arguments['-r']))
  if '-v' in arguments:
    param.verbose = bool(int(arguments['-v']))
  if '-param' in arguments:
    param_glcm.update(arguments['-param'])

  print param_glcm.angle
    
if __name__ == "__main__":
    main()
