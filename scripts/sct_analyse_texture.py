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
    parser = Parser(__file__)
    parser.usage.set_description('Extraction of texture features from an image within a given mask.\n'
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
                      type_value="str",
                      description="Parameters for extraction. Separate arguments with \":\".\n"
                                  "prop: <list_stg> list of GLCM texture property. Default="+ParamGLCM().prop+"\n"
                                  "distance: <int> distance offset. Default="+str(ParamGLCM().distance)+"\n"
                                  "angle: <list_int> list of angles in degrees. Default="+",".join([str(a) for a in ParamGLCM().angle])+"\n",
                      mandatory=False,
                      example="prop=energy:distance=1:angle=0,90")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      default_value=Param().path_results,
                      example='texture_analysis_results/')
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

    self.dct_metric = {}
    for m in list(itertools.product(self.param_glcm.prop.split(','), self.param_glcm.angle.split(','))):
      text_name = m[0] if m[0].upper()!='asm'.upper() else m[0].upper()
      metric_name = text_name+'_'+str(self.param_glcm.distance)+'_'+str(m[1])
      self.dct_metric[metric_name] = None

    self.dct_im = {'im': None, 'seg': None}

    self.orientation = get_orientation(Image(self.param.fname_im))

  def extract(self):

    # init output images
    self.init_metric_im()
    
    # extract img and seg slices in a dict
    self.extract_slices()

    # compute texture
    self.compute_texture()

    # reorient data
    if get_orientation(Image(self.param.fname_im)) != 'RPI':
      self.reorient_data()

    # save data
    fname_out_lst = self.save_data_to_ofolder()

    printv('\nDone! To view results, type:', self.param.verbose)
    printv('fslview ' + self.param.fname_im + ' ' + ' -l Red-Yellow -t 0.7 '.join(fname_out_lst) + ' -l Red-Yellow -t 0.7 & \n', self.param.verbose, 'info')

  def extract_slices(self):
    im = Image(self.param.fname_im) if get_orientation(Image(self.param.fname_im)) == 'RPI' else set_orientation(Image(self.param.fname_im), 'RPI')
    seg = Image(self.param.fname_seg) if get_orientation(Image(self.param.fname_seg)) == 'RPI' else set_orientation(Image(self.param.fname_seg), 'RPI')
    
    self.dct_im['im'] = [im.data[:,:,z] for z in range(im.dim[2])]
    self.dct_im['seg'] = [seg.data[:,:,z] for z in range(im.dim[2])]

    del im, seg

  def init_metric_im(self):

    im_tmp = Image(self.param.fname_im) if get_orientation(Image(self.param.fname_im)) == 'RPI' else set_orientation(Image(self.param.fname_im), 'RPI')

    for m in self.dct_metric:
      im_2save = im_tmp.copy()
      im_2save.changeType(type='float64')
      im_2save.data *= 0
      self.dct_metric[m] = im_2save
      del im_2save

    del im_tmp

  def compute_texture(self):

    offset = int(self.param_glcm.distance)

    printv('\nCompute texture...', self.param.verbose, 'normal')

    timer_texture = Timer(number_of_iteration=len(self.dct_im['im']))
    timer_texture.start()

    for im_z, seg_z,z in zip(self.dct_im['im'],self.dct_im['seg'],range(len(self.dct_im['im']))):
      for im_x in range(im_z.shape[0]):
        for im_y in range(im_z.shape[1]):
          if im_x < offset or im_y < offset:
              continue
          if im_x > (im_z.shape[0] - offset-1) or im_y > (im_z.shape[1] - offset-1):
              continue
          if False in np.unique(seg_z[im_x-offset : im_x+offset+1, im_y-offset : im_y+offset+1]):
              continue

          glcm_window = im_z[im_x-offset : im_x+offset+1, im_y-offset : im_y+offset+1]
        
          glcm_window = glcm_window.astype(np.uint8)

          dct_glcm = {}
          for a in self.param_glcm.angle.split(','):
            dct_glcm[a] = greycomatrix(glcm_window, [self.param_glcm.distance], [radians(int(a))],  symmetric = self.param_glcm.symmetric, normed = self.param_glcm.normed)
          
          for m in self.dct_metric:
            self.dct_metric[m].data[im_x,im_y,z] = greycoprops(dct_glcm[m.split('_')[2]], m.split('_')[0])[0][0]

      print '\nHEY'
      for m in self.dct_metric:
        print m
        print np.mean(self.dct_metric[m].data[:,:,z])
      timer_texture.add_iteration()
    
    timer_texture.stop()

  def reorient_data():
    for m in self.dct_metric:
      self.dct_metric[m] = set_orientation(Image(self.dct_metric[m]), self.orientation)

  def save_data_to_ofolder(self):

    output_folder = slash_at_the_end(self.param.path_results) + '/texture/'
    if not os.path.isdir(output_folder):
      os.makedirs(output_folder)

    fname_out = []
    for m in self.dct_metric:
      fname_texture_cur = output_folder + add_suffix(''.join(extract_fname(self.param.fname_im)[1:]), '_'+m)  
      self.dct_metric[m].setFileName(fname_texture_cur)
      self.dct_metric[m].save()
      fname_out.append(fname_texture_cur)

    return fname_out

class Param:
  def __init__(self):
    self.fname_im = None
    self.fname_seg = None
    self.path_results = './'

    self.verbose = '0'

class ParamGLCM(object):
  def __init__(self, symmetric=True, normed=True, prop='contrast,dissimilarity,homogeneity,energy,correlation,ASM', distance='1', angle='0'):
    self.symmetric = True  # If True, the output matrix P[:, :, d, theta] is symmetric.
    self.normed = True  # If True, normalize each matrix P[:, :, d, theta] by dividing by the total number of accumulated co-occurrences for the given offset. 
                        # The elements of the resulting matrix sum to 1.
    self.prop = 'contrast,dissimilarity,homogeneity,energy,correlation,ASM'
    self.distance = 1
    self.angle = '0,45,90,135'

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

  glcm = ExtractGLCM(param=param, param_glcm=param_glcm)
  glcm.extract()
    
if __name__ == "__main__":
    main()
