#!/usr/bin/env python
#######################################################################################################################
#
# Analyze lesions
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
from skimage.measure import label, regionprops
import pandas as pd

from sct_maths import binarise
from msct_image import Image
from msct_parser import Parser
from sct_image import set_orientation, get_orientation
from sct_utils import (add_suffix, extract_fname, printv, run,
                       slash_at_the_end, Timer, tmp_create)

'''
TODO:
  - volume
  - if texture ou image input: prop.max_intensity, prop.mean_intensity, prop.min_intensity
  - compute_ref_feature --> attention texture errode --> erroder aussi mask?
'''


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('TODO')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Lesion mask to analyse",
                      mandatory=True,
                      example='t2_lesion.nii.gz')
    parser.add_option(name="-ref",
                      type_value="file",
                      description="Reference image for feature extraction",
                      mandatory=False,
                      example='t2.nii.gz')
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      default_value=Param().path_results,
                      example='./')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files.",
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

class AnalyzeLeion:
  def __init__(self, param=None):
    self.param = param if param is not None else Param()

    # create tmp directory
    self.tmp_dir = tmp_create(verbose=self.param.verbose)  # path to tmp directory

    self.fname_label = None

    data_dct = {}
    column_lst = ['label', 'volume', ]
    if self.param.fname_ref is not None:
      for feature in ['mean', 'std']:
        column_lst.append(feature+'_'+extract_fname(self.param.fname_ref)[1])
    for column in column_lst:
      data_dct[column] = None
    self.data_pd = pd.DataFrame(data=data_dct,index=range(0),columns=column_lst)

  def analyze(self):
    self.ifolder2tmp()

    # Binarize the input image if needed
    self.binarize()

    # Label connected regions of the masked image
    self.label_lesion()

    # Compute lesion volume
    self.compute_volume()

    # Compute mean, median, min, max value in each labeled lesion
    if self.param.fname_ref is not None:
      self.compute_ref_feature()

  def compute_ref_feature(self):
    printv('\nCompute reference image features...', self.param.verbose, 'normal')
    im_label_data, im_ref_data = Image(self.fname_label).data, Image(self.param.fname_ref).data

    for lesion_label in [l for l in np.unique(im_label_data.data) if l]:
      im_label_data_cur = im_label_data == lesion_label
      mean_cur, std_cur  = np.mean(im_ref_data[np.where(im_label_data_cur)]), np.std(im_ref_data[np.where(im_label_data_cur)])

      label_idx = self.data_pd[self.data_pd.label==lesion_label].index
      self.data_pd.loc[label_idx, 'mean_'+extract_fname(self.param.fname_ref)[1]] = mean_cur
      self.data_pd.loc[label_idx, 'std_'+extract_fname(self.param.fname_ref)[1]] = std_cur
      printv('Mean+/-std of lesion #'+str(lesion_label)+' in '+extract_fname(self.param.fname_ref)[1]+' file: '+str(round(mean_cur,2))+'+/-'+str(round(std_cur,2)), type='info')
    
    print self.data_pd

  def compute_volume(self):
    printv('\nCompute lesion volumes...', self.param.verbose, 'normal')
    im = Image(self.fname_label)
    im_data = im.data
    px, py, pz = im.dim[3:6]

    for lesion_label in [l for l in np.unique(im.data) if l]:
      volume_cur = 0.0
      im_data_cur = im_data == lesion_label
      for zz in range(im.dim[2]):
        volume_cur += np.sum(im_data_cur[:,:,zz]) * px * py * pz
      
      self.data_pd.loc[self.data_pd[self.data_pd.label==lesion_label].index, 'volume'] = volume_cur
      printv('Volume of lesion #'+str(lesion_label)+' : '+str(volume_cur)+' mm^3', type='info')

  def label_lesion(self):
    printv('\nLabel connected regions of the masked image...', self.param.verbose, 'normal')
    im = Image(self.param.fname_im)
    im_2save = im.copy()
    im_2save.data = label(im.data, connectivity=2)
    self.fname_label = add_suffix(self.param.fname_im, '_label')
    im_2save.setFileName(self.fname_label)
    im_2save.save()

    self.data_pd['label'] = [l for l in np.unique(im_2save.data) if l]

  def binarize(self):
    im = Image(self.param.fname_im)
    if len(np.unique(im.data))>2: # if the image is not binarized
      printv('\nBinarize lesion file...', self.param.verbose, 'normal')
      im_2save = im.copy()
      im_2save.data = binarise(im.data)
      im_2save.setFileName(self.param.fname_im)
      im_2save.save()

    elif list(np.unique(im.data))==[0]:
      printv('WARNING: Empty masked image', self.param.verbose, 'warning')

  def ifolder2tmp(self):
    # copy input image
    if self.param.fname_im is not None:
      shutil.copy(self.param.fname_im, self.tmp_dir)
      self.param.fname_im = ''.join(extract_fname(self.param.fname_im)[1:])
    else:
      printv('ERROR: No input image', self.param.verbose, 'error')

    # copy ref image
    if self.param.fname_ref is not None:
      shutil.copy(self.param.fname_ref, self.tmp_dir)
      self.param.fname_ref = ''.join(extract_fname(self.param.fname_ref)[1:])

    os.chdir(self.tmp_dir) # go to tmp directory

class Param:
  def __init__(self):
    self.fname_im = None
    self.fname_ref = None
    self.path_results = './'
    self.verbose = '1'
    self.rm_tmp = True

def main(args=None):
  if args is None:
    args = sys.argv[1:]

  # create param object
  param = Param()

  # get parser
  parser = get_parser()
  arguments = parser.parse(args)

  # set param arguments ad inputted by user
  param.fname_im = arguments["-i"]

  if '-ref' in arguments:
    param.fname_ref = arguments["-ref"]

  if '-ofolder' in arguments:
    param.path_results = slash_at_the_end(arguments["-ofolder"], slash=1)
  if not os.path.isdir(param.path_results) and os.path.exists(param.path_results):
      sct.printv("ERROR output directory %s is not a valid directory" % param.path_results, 1, 'error')
  if not os.path.exists(param.path_results):
      os.makedirs(param.path_results)

  if '-r' in arguments:
    param.rm_tmp = bool(int(arguments['-r']))
  if '-v' in arguments:
    param.verbose = bool(int(arguments['-v']))

  # create the Lesion constructor
  lesion_obj = AnalyzeLeion(param=param)
  # run the analyze
  lesion_obj.analyze()

    
if __name__ == "__main__":
    main()