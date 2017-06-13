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
from sct_image import set_orientation, get_orientation
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
                      example="prop=energy:distance=1:angles=0")
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

class ExtractGLCM:
  def __init__(self, param=None, param_glcm=None):
    self.param = param if param is not None else Param()
    self.param_glcm = param_glcm if param_glcm is not None else ParamGLCM()

    # create tmp directory
    self.tmp_dir = tmp_create(verbose=self.param.verbose)  # path to tmp directory

    self.dct_metric = {}
    for m in self.param_glcm.prop.split(','):
      self.dct_metric[m] = None

    self.dct_im = {'im': None, 'seg': None}

  def extract(self):
    # self.copy_data_to_tmp()
    # # go to tmp directory
    # os.chdir(self.tmp_dir)

    # init output images
    self.init_metric_im()
    
    # extract img and seg slices in a dict
    self.extract_slices()

  def extract_slices(self):
    im = Image(self.param.fname_im) if get_orientation(Image(self.param.fname_im)) == 'RPI' else set_orientation(Image(self.param.fname_im), 'RPI')
    seg = Image(self.param.fname_seg) if get_orientation(Image(self.param.fname_seg)) == 'RPI' else set_orientation(Image(self.param.fname_seg), 'RPI')
    
    self.dct_im['im'] = [im.data[:,:,z] for z in range(im.dim[2])]
    self.dct_im['seg'] = [seg.data[:,:,z] for z in range(im.dim[2])]

  def init_metric_im(self):

    for m in self.dct_metric:
      nb_channel = len(self.param_glcm.distance.split(','))*len(self.param_glcm.angle.split(','))

      im_tmp = Image(self.param.fname_im) if get_orientation(Image(self.param.fname_im)) == 'RPI' else set_orientation(Image(self.param.fname_im), 'RPI')
      im_2save = im_tmp.copy()
      im_2save.data = np.zeros((im_2save.dim[0], im_2save.dim[1], im_2save.dim[2], nb_channel))
      im_2save.dim = tuple([im_2save.dim[d] if d!=3 else nb_channel for d in range(len(im_2save.dim))])
      
      self.dct_metric[m] = im_2save

#######################
    #     for i in range(energy.shape[0]):
    #         for j in range(energy.shape[1]):
    #             if i <1 or j <1:
    #                 continue
    #             if i > (energy.shape[0] - 2) or j > (energy.shape[1] - 2):
    #                 continue
    #             if False in np.unique(wm[i-1: i+2, j-1 : j+2]):
    #                 continue

    #             glcm_window = img[i-1: i+2, j-1 : j+2]
                
    #             glcm_window = glcm_window.astype(np.uint8)
    #             glcm = greycomatrix(glcm_window, [1], [0],  symmetric = True, normed = True)

    #             for t, t_stg in zip([energy, contrast, dissimilarity, homogeneity, ASM, correlation], texture_stg):
    #                 t[i,j]= greycoprops(glcm, t_stg)
#######################


        # self.target_im, self.info_preprocessing = pre_processing(self.param_seg.fname_im, self.param_seg.fname_seg, self.param_seg.fname_level, new_res=self.param_data.axial_res, square_size_size_mm=self.param_data.square_size_size_mm, denoising=self.param_data.denoising, verbose=self.param.verbose, rm_tmp=self.param.rm_tmp)

        # printv('\nRegister target image to model data...', self.param.verbose, 'normal')
        # # register target image to model dictionary space
        # path_warp = self.register_target()

        # if self.param_data.normalization:
        #     printv('\nNormalize intensity of target image...', self.param.verbose, 'normal')
        #     self.normalize_target()

        # printv('\nProject target image into the model reduced space...', self.param.verbose, 'normal')
        # self.project_target()

        # printv('\nCompute similarities between target slices and model slices using model reduced space...', self.param.verbose, 'normal')
        # list_dic_indexes_by_slice = self.compute_similarities()

        # printv('\nLabel fusion of model slices most similar to target slices...', self.param.verbose, 'normal')
        # self.label_fusion(list_dic_indexes_by_slice)

        # printv('\nWarp back segmentation into image space...', self.param.verbose, 'normal')
        # self.warp_back_seg(path_warp)

        # printv('\nPost-processing...', self.param.verbose, 'normal')
        # self.im_res_gmseg, self.im_res_wmseg = self.post_processing()

        # if (self.param_seg.path_results != './') and (not os.path.exists('../' + self.param_seg.path_results)):
        #     # create output folder
        #     printv('\nCreate output folder ...', self.param.verbose, 'normal')
        #     os.chdir('..')
        #     os.mkdir(self.param_seg.path_results)
        #     os.chdir(self.tmp_dir)

        # if self.param_seg.fname_manual_gmseg is not None:
        #     # compute validation metrics
        #     printv('\nCompute validation metrics...', self.param.verbose, 'normal')
        #     self.validation()

        # if self.param_seg.ratio is not '0':
        #     printv('\nCompute GM/WM CSA ratio...', self.param.verbose, 'normal')
        #     self.compute_ratio()

        # # go back to original directory
        # os.chdir('..')
        # printv('\nSave resulting GM and WM segmentations...', self.param.verbose, 'normal')
        # self.fname_res_gmseg = self.param_seg.path_results + add_suffix(''.join(extract_fname(self.param_seg.fname_im)[1:]), '_gmseg')
        # self.fname_res_wmseg = self.param_seg.path_results + add_suffix(''.join(extract_fname(self.param_seg.fname_im)[1:]), '_wmseg')

        # self.im_res_gmseg.setFileName(self.fname_res_gmseg)
        # self.im_res_wmseg.setFileName(self.fname_res_wmseg)

        # self.im_res_gmseg.save()
        # self.im_res_wmseg.save()

  def copy_data_to_tmp(self):
    # copy input image
    if self.param.fname_im is not None:
      shutil.copy(self.param.fname_im, self.tmp_dir)
      self.param.fname_im = ''.join(extract_fname(self.param.fname_im)[1:])
    else:
      printv('ERROR: No input image', self.param.verbose, 'error')

    # copy seg image
    if self.param.fname_seg is not None:
      shutil.copy(self.param.fname_seg, self.tmp_dir)
      self.param.fname_seg = ''.join(extract_fname(self.param.fname_seg)[1:])
    else:
      printv('ERROR: No segmentation image', self.param.verbose, 'error')

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

  glcm = ExtractGLCM(param=param, param_glcm=param_glcm)
  glcm.extract()

    
if __name__ == "__main__":
    main()
