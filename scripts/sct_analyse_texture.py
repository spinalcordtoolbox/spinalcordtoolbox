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
import itertools
from math import radians
from skimage.feature import greycomatrix, greycoprops

import sct_maths
from msct_image import Image
from msct_parser import Parser
from sct_image import set_orientation, get_orientation
from sct_utils import (add_suffix, extract_fname, printv, run,
                       slash_at_the_end, tmp_create, Timer)

'''
TODO:
  - securiser le parser
  - optimiser le temps de calcul --> tous les angles en meme temps?
  - facon de save
  - Test difference sym et normed
  - arranger la doc
  - report github
'''


def get_parser():
    # Initialize the parser
    parser = Parser(__file__) # %%%%%
    parser.usage.set_description('Extraction of GLCM texture features from an image within a given mask.\n'
                                 ' It calculates the texture properties of a grey level co-occurence matrix (GLCM).'
                                 ' The textures features are those defined in the sckit-image implementation:\n'
                                 ' http://scikit-image.org/docs/dev/api/skimage.feature.html#greycoprops\n'
                                 ' This function outputs one nifti file per texture metric (contrast, dissimilarity, homogeneity, ASM, energy, correlation) and per orientation in the folder ./texture/')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image to analyse",
                      mandatory=True,
                      example='t2.nii.gz')
    parser.add_option(name="-s",
                      type_value="file",
                      description="Image mask",
                      mandatory=True,
                      example='t2_seg.nii.gz')
    parser.add_option(name="-param",
                      type_value="str", # %%%%%
                      description="Parameters for extraction. Separate arguments with \":\".\n"
                                  "prop: <list_stg> list of GLCM texture properties. Default="+ParamGLCM().prop+"\n"
                                  "distance: <int> distance offset. Default="+str(ParamGLCM().distance)+"\n"
                                  "angle: <list_int> list of angles (in degrees). Default="+",".join([str(a) for a in ParamGLCM().angle])+"\n",
                      mandatory=False,
                      example="prop=energy:distance=1:angle=0,90")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      default_value=Param().path_results,
                      example='texture') # %%%%%
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value=str(Param().verbose))

    return parser

class ExtractGLCM:
  def __init__(self, param=None, param_glcm=None):
    self.param = param if param is not None else Param()
    self.param_glcm = param_glcm if param_glcm is not None else ParamGLCM()

    # dct_metric{'property_distance_angle': Image}
    self.dct_metric = {}
    for m in list(itertools.product(self.param_glcm.prop.split(','), self.param_glcm.angle.split(','))):
      text_name = m[0] if m[0].upper()!='asm'.upper() else m[0].upper()
      metric_name = text_name+'_'+str(self.param_glcm.distance)+'_'+str(m[1])
      self.dct_metric[metric_name] = None

    # dct_im_seg{'im': list_of_axial_slice, 'seg': list_of_axial_masked_slice}
    self.dct_im_seg = {'im': None, 'seg': None}

    # to re-orient the data at the end if needed
    self.orientation = get_orientation(Image(self.param.fname_im))

  def extract(self):

    # fill self.dct_metric --> for each key_metric: create an Image with zero values
    self.init_dct_metric()
    
    # fill self.dct_im_seg --> extract axial slices from self.param.fname_im and self.param.fname_seg
    self.extract_slices()

    # compute texture
    self.compute_texture()

    # reorient data
    if self.orientation != 'RPI':
      self.reorient_data()

    # save data
    fname_out_lst = self.save_data_to_ofolder()

    printv('\nDone! To view results, type:', self.param.verbose)
    printv('fslview ' + self.param.fname_im + ' ' + ' -l Red-Yellow -t 0.7 '.join(fname_out_lst) + ' -l Red-Yellow -t 0.7 & \n', self.param.verbose, 'info')

  def extract_slices(self):

    # open image and re-orient it to RPI if needed
    if self.orientation == 'RPI':
      im, seg = Image(self.param.fname_im), Image(self.param.fname_seg)
    else:
      im, seg = set_orientation(Image(self.param.fname_im), 'RPI'), set_orientation(Image(self.param.fname_seg), 'RPI')
    
    # extract axial slices in self.dct_im_seg
    self.dct_im_seg['im'], self.dct_im_seg['seg'] = [im.data[:,:,z] for z in range(im.dim[2])], [seg.data[:,:,z] for z in range(im.dim[2])]
    
    del im, seg # %%%%%

  def init_dct_metric(self):

    # open image and re-orient it to RPI if needed
    im_tmp = Image(self.param.fname_im) if self.orientation == 'RPI' else set_orientation(Image(self.param.fname_im), 'RPI')

    # create Image objects with zeros values for each output image needed
    for m in self.dct_metric:
      im_2save = im_tmp.copy()
      im_2save.changeType(type='float64')
      im_2save.data *= 0
      self.dct_metric[m] = im_2save
      del im_2save

    del im_tmp # %%%%%

  def compute_texture(self):

    offset = int(self.param_glcm.distance)

    printv('\nCompute texture metrics...', self.param.verbose, 'normal')

    timer_texture = Timer(number_of_iteration=len(self.dct_im_seg['im']))
    timer_texture.start()  # %%%%%

    for im_z, seg_z,zz in zip(self.dct_im_seg['im'],self.dct_im_seg['seg'],range(len(self.dct_im_seg['im']))):
      for xx in range(im_z.shape[0]):
        for yy in range(im_z.shape[1]):
          if xx < offset or yy < offset:  # %%%%% --> to add in parser security
              continue
          if xx > (im_z.shape[0] - offset-1) or yy > (im_z.shape[1] - offset-1):
              continue # to check if the whole glcm_window is in the axial_slice
          if False in np.unique(seg_z[xx-offset : xx+offset+1, yy-offset : yy+offset+1]):
              continue # to check if the whole glcm_window is in the mask of the axial_slice

          glcm_window = im_z[xx-offset : xx+offset+1, yy-offset : yy+offset+1]
          glcm_window = glcm_window.astype(np.uint8)

          dct_glcm = {}  # %%%%% --> optimize the way to compute: all angle at the same time OR in a loop + investigate symmetric and normed param
          for a in self.param_glcm.angle.split(','): # compute the GLCM for self.param_glcm.distance and for each self.param_glcm.angle
            dct_glcm[a] = greycomatrix(glcm_window, [self.param_glcm.distance], [radians(int(a))],  symmetric = self.param_glcm.symmetric, normed = self.param_glcm.normed)
          
          for m in self.dct_metric: # compute the GLCM property (m.split('_')[0]) of the voxel xx,yy,zz
            self.dct_metric[m].data[xx,yy,zz] = greycoprops(dct_glcm[m.split('_')[2]], m.split('_')[0])[0][0]

      timer_texture.add_iteration()
    
    timer_texture.stop()

  def reorient_data():
    for m in self.dct_metric:
      self.dct_metric[m] = set_orientation(Image(self.dct_metric[m]), self.orientation)

  def save_data_to_ofolder(self):

    output_folder = slash_at_the_end(self.param.path_results) + '/texture/' # %%%%%

    # create output folder
    if not os.path.isdir(output_folder): # %%%%%
      os.makedirs(output_folder)

    # save each output image with the following template_name: property_distance_angle.nii.gz
    fname_out_lst = []
    for m in self.dct_metric:
      fname_texture_cur = output_folder + add_suffix(''.join(extract_fname(self.param.fname_im)[1:]), '_'+m)  
      self.dct_metric[m].setFileName(fname_texture_cur)
      self.dct_metric[m].save()
      fname_out.append(fname_out_lst)

    return fname_out_lst

class Param:
  def __init__(self):
    self.fname_im = None
    self.fname_seg = None
    self.path_results = './'
    self.verbose = '0'

class ParamGLCM(object):
  def __init__(self, symmetric=True, normed=True, prop='contrast,dissimilarity,homogeneity,energy,correlation,ASM', distance='1', angle='0'):
    self.symmetric = True  # If True, the output matrix P[:, :, d, theta] is symmetric.
    self.normed = True  # If True, normalize each matrix P[:, :, d, theta] by dividing by the total number of accumulated co-occurrences for the given offset. The elements of the resulting matrix sum to 1.
    self.prop = 'contrast,dissimilarity,homogeneity,energy,correlation,ASM' # The property formulae are detailed here: http://scikit-image.org/docs/dev/api/skimage.feature.html#greycoprops
    self.distance = 1 # Size of the window: distance = 1 --> a reference pixel and its immediate neighbour
    self.angle = '0,45,90,135' # Rotation angles for co-occurrence matrix

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
  if '-v' in arguments:
    param.verbose = bool(int(arguments['-v']))
  if '-param' in arguments:
    param_glcm.update(arguments['-param'])

  # create the GLCM constructor
  glcm = ExtractGLCM(param=param, param_glcm=param_glcm)
  # run the extraction
  glcm.extract()
    
if __name__ == "__main__":
    main()