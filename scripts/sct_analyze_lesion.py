#!/usr/bin/env python

# Analyze lesions
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley
# Modified: 2017-06-13
#
# About the license: see the file LICENSE.TXT

import os
import shutil
import sys
import numpy as np
import itertools
from math import radians, pi, sqrt
from skimage.measure import label, regionprops
import pandas as pd

from sct_maths import binarise
from msct_image import Image
from msct_parser import Parser
from sct_image import set_orientation, get_orientation
from sct_utils import (add_suffix, extract_fname, printv, run,
                       slash_at_the_end, Timer, tmp_create, get_absolute_path)
from sct_straighten_spinalcord import smooth_centerline
from msct_types import Centerline


'''
TODO:
  - vertebra --> volume?
  - wm et gm --> comment thresholder
  - donner un volume et pas de percentage
  - comment input atlas et template
  - comment presenter les resultats
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
    parser.add_option(name="-s",
                      type_value="file",
                      description="Spinal cord centerline or segmentation file for angle correction",
                      mandatory=False,
                      example='t2_seg.nii.gz')
    parser.add_option(name="-ref",
                      type_value="file",
                      description="Reference image for feature extraction",
                      mandatory=False,
                      example='t2.nii.gz')
    parser.add_option(name="-atlas_folder",
                      type_value="str",
                      description="Folder containing the atlas registered to the anatomical image",
                      mandatory=False,
                      example="./label/atlas")
    parser.add_option(name="-template_folder",
                      type_value="str",
                      description="Folder containing the template registered to the anatomical image",
                      mandatory=False,
                      example="./label/template")
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
    column_lst = ['label', 'volume', 'si_length', 'ax_nominal_diameter']
    if self.param.fname_ref is not None:
      for feature in ['mean', 'std']:
        column_lst.append(feature+'_'+extract_fname(self.param.fname_ref)[1])
    for column in column_lst:
      data_dct[column] = None
    self.data_pd = pd.DataFrame(data=data_dct,index=range(0),columns=column_lst)

    self.orientation = None

    self.angles = None

    if self.param.path_template is not None:
      self.data_template_pd = pd.DataFrame(data={'area': None, 'ratio':None},
                                  index=range(0), columns=['area', 'ratio'])
    else:
      self.data_template_pd = None

  def analyze(self):
    self.ifolder2tmp()

    # Orient input image(s) to RPI
    self.orient_rpi()

    # Binarize the input image if needed
    self.binarize()

    # Label connected regions of the masked image
    self.label_lesion()

    # Compute angle for CSA correction
    self.angle_correction()

    # Compute lesion volume, equivalent diameter, (S-I) length, max axial nominal diameter
    self.measure_without_ref_registration()

    # # Compute mean, median, min, max value in each labeled lesion
    # if self.param.fname_ref is not None:
    #   self.compute_ref_feature()

    # Compute ratio vol_lesion_vertbra / vol_lesion_tot
    if self.param.path_template is not None:
      self.measure_template_ratio()

    # # Compute ratio vol_lesion_gm / vol_lesion_tot and vol_lesion_wm / vol_lesion_tot
    # if self.param.path_template is not None:
    #   self.measure_gm_wm_ratio()

    print self.data_template_pd

  def _measure_gm_wm_ratio(self, im_lesion, im_template_lst, vox_tot, idx_pd):
    printv('\nCompute lesions ratio GM/WM...', self.param.verbose, 'normal')

    for idx,area_name,im_cur in zip(range(idx_pd,idx_pd+3), ['GM', 'WM'], im_template_lst):
      vox_cur = np.count_nonzero(im_lesion.data[np.where(im_cur.data > 0.4)])
      print vox_cur, vox_tot, area_name

      self.data_template_pd.loc[idx, 'area'] = area_name
      self.data_template_pd.loc[idx, 'ratio'] = vox_cur * 100.0 / vox_tot

  def _measure_vertebra_ratio(self, im_lesion, im_vert, vox_tot):
    printv('\nCompute lesions ratio across vertebrae...', self.param.verbose, 'normal')

    vert_lst = [v for v in list(np.unique(im_vert.data)) if v]

    for vv,vert in enumerate(vert_lst):
      v_idx = vv+1
      vox_cur = np.count_nonzero(im_lesion.data[np.where(im_vert.data==vv)])

      self.data_template_pd.loc[vv, 'area'] = 'C'+str(v_idx) if v_idx < 8 else 'T'+str(v_idx-7)
      self.data_template_pd.loc[vv, 'ratio'] = vox_cur * 100.0 / vox_tot

    self.data_template_pd.loc[vv+1, 'area'] = ' '
    self.data_template_pd.loc[vv+1, 'ratio'] = np.nan

    return vv+2

  def measure_template_ratio(self):

    im_lesion = Image(self.param.fname_im)

    # vol_tot = np.sum(self.data_pd['volume'].values.tolist())
    vox_tot = np.count_nonzero(im_lesion.data)

    im_vert = Image(self.param.path_template+'PAM50_levels.nii.gz')
    idx_pd = self._measure_vertebra_ratio(im_lesion, im_vert, vox_tot)
    
    im_gm = Image(self.param.path_template+'PAM50_gm.nii.gz')
    im_wm = Image(self.param.path_template+'PAM50_wm.nii.gz')
    self._measure_gm_wm_ratio(im_lesion, [im_gm, im_wm], vox_tot, idx_pd)


  def compute_ref_feature(self):
    printv('\nCompute reference image features...', self.param.verbose, 'normal')
    im_label_data, im_ref_data = Image(self.fname_label).data, Image(self.param.fname_ref).data

    for lesion_label in [l for l in np.unique(im_label_data.data) if l]:
      im_label_data_cur = im_label_data == lesion_label
      im_label_data_cur[np.where(im_ref_data==0)] = 0 # if the ref object is eroded compared to the labeled object
      mean_cur, std_cur  = np.mean(im_ref_data[np.where(im_label_data_cur)]), np.std(im_ref_data[np.where(im_label_data_cur)])

      label_idx = self.data_pd[self.data_pd.label==lesion_label].index
      self.data_pd.loc[label_idx, 'mean_'+extract_fname(self.param.fname_ref)[1]] = mean_cur
      self.data_pd.loc[label_idx, 'std_'+extract_fname(self.param.fname_ref)[1]] = std_cur
      printv('Mean+/-std of lesion #'+str(lesion_label)+' in '+extract_fname(self.param.fname_ref)[1]+' file: '+str(round(mean_cur,2))+'+/-'+str(round(std_cur,2)), type='info')
    
  def _measure_volume(self, im_data, p_lst):

    volume_cur = 0.0
    for zz in range(im_data.shape[2]):
      volume_cur += np.sum(im_data[:,:,zz]) * np.cos(self.angles[zz]) * p_lst[0] * p_lst[1] * p_lst[2]
    
    printv('  Volume : '+str(round(volume_cur,2))+' mm^3', type='info')
    return volume_cur

  def _measure_length(self, im_data, p_z):

    length_cur = np.sum([np.cos(self.angles[zz]) * p_z for zz in list(np.unique(np.where(im_data)[2]))])

    printv('  (S-I) length : '+str(round(length_cur,2))+' mm', type='info')
    return length_cur

  def _measure_diameter(self, im_data, p_lst):
    
    area_lst = []
    for zz in range(im_data.shape[2]):
      area_lst.append(np.sum(im_data[:,:,zz]) * np.cos(self.angles[zz]) * p_lst[0] * p_lst[1])
    diameter_cur = sqrt(max(area_lst)/(4*pi))
    
    printv('  Max. axial nominal diameter : '+str(round(diameter_cur,2))+' mm', type='info')
    return diameter_cur

  def measure_without_ref_registration(self):
    im = Image(self.fname_label)
    im_data = im.data
    p_lst = im.dim[3:6]    

    for lesion_label in [l for l in np.unique(im.data) if l]:
      im_data_cur = im_data == lesion_label
      printv('\nMeasures on lesion #'+str(lesion_label)+'...', self.param.verbose, 'normal')

      label_idx = self.data_pd[self.data_pd.label==lesion_label].index
      self.data_pd.loc[label_idx, 'volume'] = self._measure_volume(im_data_cur, p_lst)
      self.data_pd.loc[label_idx, 'si_length'] = self._measure_length(im_data_cur, p_lst[2])
      self.data_pd.loc[label_idx, 'ax_nominal_diameter'] = self._measure_diameter(im_data_cur, p_lst)

  def _normalize(self, vect):
      norm = np.linalg.norm(vect)
      return vect / norm

  def angle_correction(self):

    if self.param.fname_seg is not None:
      im_seg = Image(self.param.fname_seg)
      data_seg = im_seg.data
      X, Y, Z = (data_seg > 0).nonzero()
      min_z_index, max_z_index = min(Z), max(Z)

      # fit centerline, smooth it and return the first derivative (in physical space)
      x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(self.param.fname_seg, algo_fitting='hanning', type_window='hanning', window_length=80, nurbs_pts_number=3000, phys_coordinates=True, verbose=self.param.verbose, all_slices=False)
      centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

      # average centerline coordinates over slices of the image
      x_centerline_deriv_rescorr, y_centerline_deriv_rescorr, z_centerline_deriv_rescorr = centerline.average_coordinates_over_slices(im_seg)[3:]

      # compute Z axis of the image, in physical coordinate
      axis_Z = im_seg.get_directions()[2]

      # Empty arrays in which angle for each z slice will be stored
      self.angles = np.zeros(im_seg.dim[2])

      # for iz in xrange(min_z_index, max_z_index + 1):
      for zz in range(im_seg.dim[2]):
        if zz >= min_z_index and zz <= max_z_index:
          # in the case of problematic segmentation (e.g., non continuous segmentation often at the extremities), display a warning but do not crash
          try: # normalize the tangent vector to the centerline (i.e. its derivative)
            tangent_vect = self._normalize(np.array([x_centerline_deriv_rescorr[zz], y_centerline_deriv_rescorr[zz], z_centerline_deriv_rescorr[zz]]))

          except IndexError:
            sct.printv('WARNING: Your segmentation does not seem continuous, which could cause wrong estimations at the problematic slices. Please check it, especially at the extremities.', type='warning')

          # compute the angle between the normal vector of the plane and the vector z
          self.angles[zz] = np.arccos(np.vdot(tangent_vect, axis_Z))

    else:
      self.angles = np.zeros(Image(self.param.fname_im).dim[2])

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

  def orient(self, fname, orientation):

    im = Image(fname)
    im = set_orientation(im, orientation)
    im.setFileName(fname)
    im.save() 

  def orient_rpi(self):
    printv('\nOrient input image(s) to RPI orientation...', self.param.verbose, 'normal')

    self.orientation = get_orientation(Image(self.param.fname_im))

    self.orient(self.param.fname_im, 'RPI')
    if self.param.fname_seg is not None:
      self.orient(self.param.fname_seg, 'RPI')
    if self.param.fname_ref is not None:
      self.orient(self.param.fname_ref, 'RPI')

  def ifolder2tmp(self):
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

    # copy ref image
    if self.param.fname_ref is not None:
      shutil.copy(self.param.fname_ref, self.tmp_dir)
      self.param.fname_ref = ''.join(extract_fname(self.param.fname_ref)[1:])

    os.chdir(self.tmp_dir) # go to tmp directory

class Param:
  def __init__(self):
    self.fname_im = None
    self.fname_seg = None
    self.fname_ref = None
    self.path_results = './'
    # self.path_atlas = None
    self.path_template = None
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

  if '-s' in arguments:
    param.fname_seg = arguments["-s"]
  if '-ref' in arguments:
    param.fname_ref = arguments["-ref"]

  # if '-atlas_folder' in arguments:
  #   param.path_atlas = slash_at_the_end(arguments["-atlas_folder"], slash=1)
  # if not os.path.isdir(param.path_atlas) and os.path.exists(param.path_atlas):
  #   sct.printv("ERROR output directory %s is not a valid directory" % param.path_atlas, 1, 'error')

  if '-template_folder' in arguments:
    param.path_template = slash_at_the_end(arguments["-template_folder"], slash=1)
  if not os.path.isdir(param.path_template) and os.path.exists(param.path_template):
    sct.printv("ERROR output directory %s is not a valid directory" % param.path_template, 1, 'error')

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